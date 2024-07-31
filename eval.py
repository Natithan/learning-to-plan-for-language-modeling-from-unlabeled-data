# region preamble
import csv
import json
import os
import sys
import warnings
from collections import OrderedDict
from copy import copy, deepcopy
import wandb

from args import scores_iso_codes
from constants import FIXED_CODE, DEFAULT_PICKLE_DIR, EOT_STRING
from criticize_text_generation.format_generations import predict_and_transform_json
from criticize_text_generation.scripts.criticize.criticize import eval_latent_PPL_alt
from criticize_text_generation.scripts.criticize.fit_critic import get_best_model, eval_latent_PPL
from eval_generations import obtain_codes_for_generations, evaluate_metric_for_generations, get_sentence_embedder, \
    load_kmeans_model
from tune_cclm import get_quantization_config
from util import tn, Namespace, is_in_sorted_list
import numpy as np
import re
from dataset import WikiDataset, MyDataset, get_kmeans_path_from_args
from Levenshtein import distance as levenshtein_distance

# print("Importing torch ..."); s = time();
import torch
import torch.nn.functional as F
# print(f"Imported torch in {time()-s:.2f}s")
from torch.nn import CrossEntropyLoss
from functools import partial
from tqdm import tqdm as std_tqdm; tqdm = partial(std_tqdm, dynamic_ncols=True)
from models.muzero import is_embedding_space_prediction
import pickle
from os.path import join as jn
# endregion

def intify(code): # Needed when resuming eval in wandb: the old state gets stored as int while in the code it is stored as tensor
    if type(code) == int:
        return code
    else:
        return code.item()

def make_article_chunks_wrapper(article_and_extra_args):
    """
    Helper function to create chunks for a single article.
    This will be called in parallel for each article.
    """
    article = article_and_extra_args[0]
    max_seq_len, tokenizer, stride, cc_type, eval, code_type = article_and_extra_args[1]
    return WikiDataset.make_article_chunks(article, max_seq_len, tokenizer, stride=stride,
                                  cc_type=cc_type, eval=eval, code_type=code_type)


def eval_ppl_parallel(args, ds, logger, lm, planner):
    fixed_or_uniform = args.fixed_code or args.uniform_mix
    prep_models_for_eval(lm, planner)
    split = get_eval_split(args)
    if not args.cc_type == 'none':
        if not (args.fixed_code or args.uniform_mix):
            with torch.no_grad():
                if planner is None:
                    planner = lm
                ds.add_planner_codes_or_scores(planner, args, logger, [split])
    articles = ds.split2coded_articles[split][:1000]
    maybe_log_accuracy_parallel(args, articles, logger)
    ct_kwarg = {'code_type': f'planner_{"scores" if scores_iso_codes(args) else "codes"}'} if not fixed_or_uniform else {}
    mac_kwargs = {
        'max_seq_len': ds.max_seq_len,
        'tokenizer': ds.tokenizer,
        'stride': args.eval_stride,
        'cc_type': args.cc_type,
        'eval': True,
    }
    if not fixed_or_uniform:
        mac_kwargs |= ct_kwarg
    chunks = [c for article in tqdm(articles, desc="Making article chunks for parallel eval") for c in ds.make_article_chunks(article, **mac_kwargs)]

    BATCH_SIZE = args.parallel_eval_batch_size
    n_chunks = len(chunks)
    total_nll = 0
    nll_count = 0

    if args.eval_save_per_token_nll:
        all_per_token_nlls = []

    with torch.no_grad():
        t = tqdm(range(0, n_chunks, BATCH_SIZE), desc="Parallel nll eval")
        for i in t:
            chunks_batch = [(ds.expand_chunk(chunk, **ct_kwarg) if args.cc_type in ['none','adapter'] else chunk) for chunk in chunks[i:i+BATCH_SIZE]]
            # list of dicts of tensors to dict of tensors with batch dim
            batch = {k: torch.stack([c[k] for c in chunks_batch]).to(lm.device) for k in chunks_batch[0]}
            INPUT_KEY = 'input_ids' if args.cc_type in ['adapter','none'] else 'input_ids_and_codes'
            if args.cc_type == 'adapter':
                lm_args = {'input_ids': batch[INPUT_KEY], 'attention_mask': batch['attention_mask']}
                if scores_iso_codes(args):
                    if args.uniform_mix:
                        lm_args |= {'planner_scores': torch.zeros(batch['input_ids'].shape[0], batch['input_ids'].shape[1], args.cluster_count).to(lm.device)}
                    else:
                        lm_args |= {'planner_scores': batch['planner_scores']}
                else:
                    lm_args |= {'codes': batch['planner_codes'] if not fixed_or_uniform else torch.zeros_like(batch['codes'])+FIXED_CODE}
            elif args.cc_type == 'none':
                lm_args = {'input_ids': batch[INPUT_KEY], 'attention_mask': batch['attention_mask']}
            else:
                lm_args = {'input_ids': batch[INPUT_KEY], 'attention_mask': batch['attention_mask'], 'code_mask': batch['code_mask']}
                if args.fixed_code:
                    raise NotImplementedError("Not implemented yet")
            all_logits = lm(**lm_args).logits
            label_idxs = get_last_nonpad_idx_per_row(batch['attention_mask'])
            logit_idxs = label_idxs - 1
            logits_for_eval = all_logits[torch.arange(all_logits.shape[0]), logit_idxs]
            labels = batch[INPUT_KEY][torch.arange(all_logits.shape[0]), label_idxs]
            total_nll += CrossEntropyLoss(reduction='sum')(logits_for_eval, labels)
            nll_count += (labels != -100).sum().item()
            avg_nll = total_nll / nll_count
            t.set_description(f"Avg ppl: {np.exp(tn(avg_nll)):.2f}")
            t.refresh()

            if args.eval_save_per_token_nll:
                per_token_nlls = CrossEntropyLoss(reduction='none')(logits_for_eval, labels)
                all_per_token_nlls.extend(per_token_nlls.tolist())

    # logger.log_metrics({'fully_greedy_nll': total_nll / nll_count})
    relevant_prefix = \
        'fully_fixed' if args.fixed_code else \
        'uniform' if args.uniform_mix else \
        'none' if args.cc_type == 'none' else \
        'fully_greedy'
    ppl = np.exp(tn(total_nll) / nll_count)
    logger.log_metrics({f'eval/relevant_perplexity': ppl}, step=nll_count)
    logger.log_metrics({f'eval/{relevant_prefix}_perplexity': ppl}, step=nll_count)

    if args.eval_save_per_token_nll:
        dir = jn(args.checkpoint_dir, logger.experiment.id)
        import os
        os.makedirs(dir, exist_ok=True)
        with open(jn(dir, 'per_token_nlls.pkl'), 'wb') as f:
            pickle.dump(all_per_token_nlls, f)


def maybe_log_accuracy_parallel(args, articles, logger):
    if not args.cc_type == 'none':
        true_codes = [c for a in articles for c in a['codes']]
        if args.fixed_code or args.uniform_mix:
            planned_codes = [FIXED_CODE for _ in true_codes]
        else:
            if scores_iso_codes(args):
                planned_codes = [c.argmax(-1) for a in articles for c in a['planner_scores']]
            else:
                planned_codes = [c for a in articles for c in a['planner_codes']]
        accuracy = (np.array(true_codes) == np.array(planned_codes)).mean()
        logger.log_metrics({'fully_greedy_accuracy': accuracy})


def noctx_geneval(args, ds, logger, lm, planner):
    split = get_eval_split(args)
    articles = ds.load_unsentencized_articles(split)[:args.noctx_gen_num_samples]
    true_texts = [a['text'] for a in articles]
    generated, _ = get_generated_texts(args, lm, planner, logger, ds, with_context=False)
    generated_texts = [e['pred_text'] for e in generated] if type(generated[0]) == dict else generated
    metrics = set(args.noctx_metrics)
    if 'mauve' in metrics:
        import mauve
        result = mauve.compute_mauve(p_text=true_texts, q_text=generated_texts, device_id=0, max_text_length=args.mauve_max_text_length, verbose=False)
        logger.log_metrics({'eval/mauve': result.mauve})
        table = wandb.Table(data=result.divergence_curve, columns=["x","y"])
        logger.log_metrics({'eval/mauve_divergence_curve': table})
        print(f"Logged Mauve score: {result.mauve}")
        metrics.remove('mauve')
    if 'latent_ppl' in metrics:
        if args.latent_ppl_critic_path is not None:
            critic_path = args.latent_ppl_critic_path
        else:
            critic_path = ...
        if os.path.exists(critic_path):
            model = torch.load(critic_path)
        else:
            model = get_best_model(args)

        # Criticise text
        for prefix, texts in [('true',true_texts), ('generated', generated_texts)]:
            data = predict_and_transform_json(texts, args)
            latent_PPL = eval_latent_PPL_alt(model, data, 'predicted_section_names')
            logger.log_metrics({f'eval/{prefix}_latent_ppl': latent_PPL})
            print(f"Logged {prefix} latent PPL: {latent_PPL}")
        metrics.remove('latent_ppl')
    if 'plan_matching' in metrics:
        embedder = get_sentence_embedder()
        kmeans = load_kmeans_model(get_kmeans_path_from_args(args))
        n_correct = 0
        n_total = 0
        for dct in generated:
            snt_bounds, pred_codes = zip(*dct['code_locs'])
            token_sentences = [dct['pred_tokens'][snt_bounds[i]:(snt_bounds[i+1] if i+1 < len(snt_bounds) else None)] for i in range(len(snt_bounds))]
            text_sentences = [ds.tokenizer.decode(s, skip_special_tokens=True) for s in token_sentences]
            text_embeds = embedder.encode(text_sentences, convert_to_numpy=True)
            generated_codes = kmeans.predict(text_embeds).tolist()
            n_correct += sum([int(g == p) for g, p in zip(generated_codes, pred_codes)])
            n_total += len(pred_codes)
        accuracy = n_correct / n_total
        logger.log_metrics({'eval/plan_matching_accuracy': accuracy})
        print(f"Logged plan matching accuracy: {accuracy * 100:.2f}%")
        metrics.remove('plan_matching')

    assert(len(metrics) == 0)

def startfullctx_geneval(args, ds, logger, lm, planner):

    gen_texts, json_path = get_generated_texts(args, lm, planner, logger, ds, with_context=True)

    texts_with_code = obtain_codes_for_generations(gen_texts, token_limits=args.startfullctx_token_limits, tokenizer=ds.tokenizer, args=args, json_path=json_path)

    results = {t: {} for t in args.startfullctx_token_limits}
    for metric in args.startfullctx_metrics:
        for t in args.startfullctx_token_limits:
            scores = []
            for t_dict in tqdm(texts_with_code):
                if "levenshtein" in metric:
                    pred_sequence = t_dict['token_limit_to_action_sequence'][t]['pred_sequence_of_actions']
                    true_sequence = t_dict['token_limit_to_action_sequence'][t]['true_sequence_of_actions']
                elif "rouge" in metric:
                    pred_sequence = t_dict['token_limit_to_action_sequence'][t]['pred_text']
                    true_sequence = t_dict['token_limit_to_action_sequence'][t]['true_text']

                scores.append(evaluate_metric_for_generations(pred_sequence, true_sequence, metric))

            results[t][metric] = sum(scores) / len(scores)
            print(f"Token limit {t}, metric {metric}: {results[t][metric]}")

    print(results)

    # # Store results in CSV file
    # filename_without_ext = args.filename.rsplit('.', 1)[0].rsplit('/')[-1]
    # csv_filename = f'startfullctx_results/{filename_without_ext}.csv'
    # os.makedirs('startfullctx_results', exist_ok=True)
    #
    # with open(csv_filename, mode='w', newline='') as csv_file:
    #     writer = csv.writer(csv_file)
    #     headers = ['Metric'] + [str(t) for t in args.token_limits]
    #     writer.writerow(headers)
    #     for metric in args.metrics:
    #         row = [metric] + [results[t][metric] for t in args.token_limits]
    #         writer.writerow(row)

    # Instead of storing in CSV file, upload to wandb
    # log table with token limits as columns and metrics as rows
    table = wandb.Table(columns=["Token Limit"] + args.startfullctx_metrics)
    for t in args.startfullctx_token_limits:
        table.add_data(t, *[results[t][metric] for metric in args.startfullctx_metrics])
    logger.log_metrics({'startfullctx_results': table})



def get_true_contexts_and_remainders(args, device, ds, lm, logger, planner):
    ns = prep_ns(args, device, ds, lm, planner)
    ct_type = f'planner_{"scores" if scores_iso_codes(args) else "codes"}' if not (args.fixed_code or args.uniform_mix or args.cc_type == 'none') else None #'codes'
    prep_ds_for_eval(args, ds, logger, ns, planner)
    articles = get_eval_articles(args, ds, logger, ns, planner)
    chunks = []
    true = []
    for a_idx, a in enumerate(articles):
        ctx = []
        for snt_idx, s in zip(a['snt_idxs'], a['tokenized_sentences']):
            ctx.extend(s)
            if len(ctx) > args.max_seq_len:
                break
        wa_sent_idx = snt_idx - a['snt_idxs'][0]
        if (len(ctx) < args.max_seq_len) or wa_sent_idx >= len(a['tokenized_sentences']) - 1:
            continue

        true_remaining_tokens = [t for s in a['tokenized_sentences'][wa_sent_idx + 1:] for t in s]
        true_remaining_text = "".join(a['sentences'][wa_sent_idx + 1:])
        true.append({'true_tokens': true_remaining_tokens, 'true_text': true_remaining_text})

        chunk_pair = WikiDataset.get_chunks_at_art_and_snt_idxs(
            article_idx=a_idx,
            within_art_snt_idx=wa_sent_idx + 1,
            get_eval_code=False,
            max_seq_len=args.max_seq_len,
            unpacked_art=MyDataset.unpack_article(a, ds.tokenizer.bos_token_id, args.cc_type, ct_type),
            bos_id=ds.tokenizer.bos_token_id,
            cc_type=args.cc_type,
            code_type=ct_type)
        chunk = {k[4:]: chunk_pair[k] for k in chunk_pair if k.startswith('ctx_')}
        chunks.append(chunk)
    return chunks, true


def get_generated_texts(args, lm, planner, logger, ds, with_context=False, device='cuda'):
    if with_context:
        contexts, true = get_true_contexts_and_remainders(args, device, ds, lm, logger, planner)
    else:
        contexts, true = None, None
    gen_wid = args.noctx_gen_wid if not with_context else args.sfctx_gen_wid
    if gen_wid is None:
        gen_wid = logger.experiment.id
    n_samples = args.noctx_gen_num_samples if not with_context else len(contexts)
    n_tokens = args.noctx_gen_num_tokens if not with_context else args.startfullctx_gen_num_tokens
    prefix = 'no' if not with_context else 'startfull'
    sys.setrecursionlimit(2048 * 2048) # https://github.com/pltrdy/rouge/issues/19
    # if args.gen_json_path is None:
    if (args_path:= dict(args)[f'{prefix}ctx_gen_json_path']) is None:
        json_path = jn(args.checkpoint_dir, f'{prefix}ctx_generated_texts', f'{gen_wid}_n{n_samples}_l{n_tokens}.json')
    else:
        json_path = args_path
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            texts = json.load(f)
    else:
        texts = compute_and_store_generated_texts(args, lm, planner, ds.tokenizer, n_samples, n_tokens, json_path, contexts, true=true)
    return texts, json_path


def compute_and_store_generated_texts(args, lm, planner, tokenizer, n_samples, n_tokens, json_path, contexts=None, true=None):
    result = []
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    start = 0
    # for _ in range((n_samples // args.noctx_gen_batch_size)+1):
    while start < n_samples:
        n_samples_batch = min(args.noctx_gen_batch_size, n_samples - len(result))
        end = start + n_samples_batch
        new_gen = generate_text_parallel(args, lm, planner, tokenizer,n_samples_batch, n_tokens, contexts[start:end] if contexts is not None else None)
        result.extend([gen_dct | true_dct for gen_dct, true_dct in zip(new_gen, true[start:end])] if true is not None else new_gen)
        print("Generated", len(result), "samples out of target", n_samples)
        start = end

        with open(json_path, 'w') as f: # This just overwrites the json every time, but that's fine as writing to json is pretty quick anyway.
            json.dump(result, f)
            print(f"Saved {len(result)} samples to {json_path}")
    return result


def generate_text_parallel(args, lm, planner, tokenizer, n_samples, n_tokens, contexts=None, device='cuda'):
    is_nocc = args.cc_type == 'none'
    prep_models_for_eval(lm, planner)
    bos_id = tokenizer.bos_token_id
    should_plan_detector = UnigramEOSDetector(tokenizer.name_or_path)
    if args.plantoken or args.cc_type not in  ['adapter','none']:
        raise NotImplementedError("Not implemented yet")

    gs = initialize_generation_state(args, bos_id, contexts, device, is_nocc, lm, n_samples, planner)

    with torch.no_grad():
        for gen_idx in tqdm(range(n_tokens)):
            next_token = sample_next_token(args, lm, gs.nocc_lm_args if is_nocc else gs.cc_lm_args)
            update_generation_state_with_token(args, gs, lm, **{f'{"nocc_" if is_nocc else ""}next_token': next_token})
            if args.cc_type != 'none':
                idxs, new_codes = get_new_codes_for_single_args_based_code_type(gs, planner, should_plan_detector, args)
                update_generation_state_with_codes(gen_idx, args, device, gs, idxs, new_codes)

    generated_tokens = torch.stack(gs.nocc_generated_tokens if is_nocc else gs.generated_tokens).T
    texts = tokenizer.batch_decode(generated_tokens)
    if contexts is not None:
        result = [{'pred_tokens': to.tolist(), 'pred_text': te, 'ctx_tokens': co['input_ids'].tolist(), 'ctx_text': tokenizer.decode(co['input_ids'])} for to, te, co in zip(generated_tokens, texts, contexts)]
    else:
        result = [{'pred_tokens': to.tolist(), 'pred_text': te} for to, te in zip(generated_tokens, texts)]
    if 'code_locs' in gs:
        result = [{**r, 'code_locs': [(loc,code.item()) for (loc,code) in gs.code_locs[i]]} for i, r in enumerate(result)]
    return result


def initialize_generation_state(args, bos_id, contexts, device, is_nocc, lm, n_samples, planner):
    gs_kwargs = {}
    if contexts is None:
        rep_bos_id = torch.tensor([bos_id]).repeat(n_samples).to(device).unsqueeze(1)
        rep_am = torch.tensor([1]).repeat(n_samples).to(device).unsqueeze(1)
        lm_args = {'input_ids': rep_bos_id, 'attention_mask': rep_am}
    else:
        lm_args = prep_batch_from_chunk(args, contexts, device)
        lm_args.pop('label_mask')
        lm_args.pop('snt_idxs')
        if 'planner_codes' in lm_args:
            lm_args['codes'] = lm_args.pop('planner_codes')
    batch_size = lm_args['input_ids'].shape[0]
    n_ctx_tokens = 1 if contexts is None else lm_args['input_ids'].shape[1]
    if args.fixed_code:
        codes = torch.tensor(FIXED_CODE, dtype=torch.int32).to(device)[None].repeat(n_samples).to(device).unsqueeze(1)
        if contexts is not None:
            codes = codes.repeat(1, n_ctx_tokens)
        lm_args['codes'] = codes
    elif args.uniform_mix:
        lm_args['planner_scores'] = torch.zeros(batch_size, n_ctx_tokens, args.cluster_count).to(device)
    elif args.cc_type != 'none':
        planner_args = {'input_ids': lm_args['input_ids'], 'attention_mask': lm_args['attention_mask']}
        gs_kwargs['planner_args'] = planner_args
        if contexts is None:
            predicted_codes_or_scores = get_greedy_new_codes_or_scores(args, planner,
                                                                       planner_args | {'cb': get_cb(args, device, lm)})
            lm_args['codes'] = predicted_codes_or_scores[:, None]
            gs_kwargs['code_locs'] = [[(0, c)] for c in predicted_codes_or_scores]
    if is_nocc:
        gs = Namespace(nocc_lm_args=lm_args, nocc_generated_tokens=[], **gs_kwargs)
    else:
        gs = Namespace(cc_lm_args=lm_args, generated_tokens=[], **gs_kwargs)
    return gs


def update_generation_state_with_codes(gen_idx, args, device, gs, idxs, new_codes, code_mask=None):
    c_or_ps = 'planner_scores' if scores_iso_codes(args) else 'codes'
    if args.cc_type == 'insert':
        raise NotImplementedError("TODO when needed. Can refer to update_generation_state_with_codes_OLD")
    else:
        if 'code_locs' in gs:
            last_codes = torch.stack([gs.code_locs[k][-1][1] for k in range(len(gs.code_locs))])
        else:
            last_codes = gs.cc_lm_args[c_or_ps][:,-1]
        # Fill new_codes in at idxs in last_codes
        codes_to_append = last_codes.clone()
        if len(idxs) > 0:
            codes_to_append[idxs] = new_codes.to(last_codes.dtype)
        # gs.cc_lm_args['codes'] = update_tensor(gs.cc_lm_args['codes'], gs.cc_lm_args['attention_mask'] if code_mask is None else code_mask, codes_to_append, args)
        gs.cc_lm_args[c_or_ps] = update_maybe_padded_tensor(gs.cc_lm_args[c_or_ps], codes_to_append, gs.cc_lm_args['attention_mask'] if code_mask is None else code_mask, args)

    for i, idx in enumerate(idxs):
        if 'code_locs' in gs:
            gs.code_locs[idx].append((gen_idx+1, new_codes[i]))


def get_new_codes_for_single_args_based_code_type(gs, planner, should_plan_detector, args):
    needs_new_code = should_plan_detector.detect_should_plan(gs.generated_tokens)
    if args.fixed_code or args.uniform_mix:
        idxs, new_codes_or_scores = [], []
    else:
        # idxs where needs new code
        idxs = torch.where(needs_new_code)[0]
        if len(idxs) == 0:
            return idxs, []
        planner_args_spec = {k: v[idxs] for k, v in gs.planner_args.items()}
        new_codes_or_scores = get_greedy_new_codes_or_scores(args, planner, planner_args_spec)
    return idxs, new_codes_or_scores


def get_greedy_new_codes_or_scores(args, planner, planner_args_spec):
    new_codes_or_scores = planner.get_greedy_logits(**planner_args_spec)
    if not scores_iso_codes(args):
        new_codes_or_scores = new_codes_or_scores.argmax(-1)
    return new_codes_or_scores


def eval_planner_lm_combo(args, ds, logger, lm, planner, stage=None, device='cuda'):

    # Input checks
    if planner is None:
        assert args.plantoken or args.fixed_code or (args.cc_type == 'none') or args.uniform_mix# or hasattr(lm, 'planner')

    # ns is a Namespace that contains various variables that are used in the eval loop
    ns = prep_ns(args, device, ds, lm, planner)

    # prep models
    prep_models_for_eval(lm, planner)

    # prep ds
    prep_ds_for_eval(args, ds, logger, ns, planner)

    # state is used to keep track of metrics to log, as well as variables that allow resuming from an interrupted eval run
    check_start, state, start_chunk_count = init_state(args, ns.extended_prefixes, logger, ns.prefixes, stage)

    articles = get_eval_articles(args, ds, logger, ns, planner)

    if not ns.only_nocc:
        update_state_codes_all(args, articles, ns, state)
        log_accuracies_all(articles, ns, state, logger, args)

    with torch.no_grad():
        assert args.eval_stride == 1, "tqdm won't be right otherwise"
        # total_chunks = sum(len(ts) for a in articles for ts in a['tokenized_sentences']) if not args.skip_nll_eval else sum(len(a['tokenized_sentences']) for a in articles)
        print(f"Total sentences: {sum(len(a['tokenized_sentences']) for a in articles)}")
        # for art_count, article in enumerate(articles):

        # update_state_for_new_article(ns, state, args, article)
        def get_chunks(cc_type, code_type=None, include_art_idxs=False):
            kwargs = {'stride': args.eval_stride, 'cc_type': cc_type, 'eval': True} | ({} if code_type is None else {'code_type': code_type})
            art_idx, chunks = zip(*[(i,c) for i, article in enumerate(tqdm(articles, desc="Making chunks from articles")) for c in ds.make_article_chunks(article, ds.max_seq_len, ds.tokenizer, **kwargs)])
            return chunks if not include_art_idxs else (torch.tensor(art_idx).to(ns.device), chunks)

        art_idxs, chunks_with_codes = get_chunks(args.cc_type, include_art_idxs=True)
        c_or_s = 'scores' if scores_iso_codes(args) else 'codes'
        if ns.fully_greedy:
            warnings.filterwarnings("ignore", message="Creating a tensor from a list of numpy.ndarrays is extremely slow.")
            chunks_with_planner_codes = get_chunks(args.cc_type, f'planner_{c_or_s}')
        if ns.insert_cc:
            chunks_without_codes = get_chunks(cc_type=None)
            assert len(chunks_with_codes) == len(chunks_without_codes)
            raise NotImplementedError("To update to non-per-article eval when insert_cc eval is needed")
            insert_specific_args = get_insert_specific_args(args, article, ds)
        else:
            insert_specific_args = {}
        art_idxs, chunks_with_codes, chunks_with_planner_codes = maybe_filter_chunks(args, art_idxs, chunks_with_codes, chunks_with_planner_codes if ns.fully_greedy else None, ns)
        with tqdm(total=len(chunks_with_codes)) as pbar:
            batch_iterator = range(0, len(chunks_with_codes), args.eval_batch_size)
            for chunk_batch_start in batch_iterator:
                chunk_batch_end = min(chunk_batch_start + args.eval_batch_size, len(chunks_with_codes))
                chunk_batch = chunks_with_codes[chunk_batch_start:chunk_batch_end]
                chunk_art_idxs = art_idxs[chunk_batch_start:chunk_batch_end]
                BS = len(chunk_batch)
                chunk_pc_batch = chunks_with_planner_codes[chunk_batch_start:chunk_batch_end] if ns.fully_greedy else None
                # region Resuming logic
                if check_start:
                    if start_chunk_count < state['chunk_count']:
                        start_chunk_count += 1
                        continue
                    else:
                        assert start_chunk_count == state['chunk_count']
                        check_start = False
                # endregion
                state['chunk_count'] += len(chunk_batch)
                batch = prep_batch_from_chunk(args, chunk_batch, device)
                pc_batch = prep_batch_from_chunk(args, chunk_pc_batch, device) if ns.fully_greedy else None

                # snt_bounds = torch.stack([c['snt_bounds'] for c in chunk_batch]).to(device)
                snt_bounds = padstack([c['snt_bounds'] for c in chunk_batch]).to(device)
                nss = new_sent_starts(batch['attention_mask'], snt_bounds, args.cc_type)
                if args.skip_nll_eval:
                    assert nss.all()
                subbatch_size = nss.sum().item()

                snt_idxs = get_snt_idxs(articles, chunk_art_idxs, batch, snt_bounds, ns.insert_cc)

                if nss.any():

                    if not ns.only_nocc:
                        if args.log_nll_for_all_codes:
                            update_allnll_persentence_state(ns.extended_prefixes, state)

                        need_greedy_logits = args.eval_pdistr_vs_kmeansdistr or args.eval_save_greedy_predictions
                        need_mcts_logits = not ns.eval_only_policy_head
                        if need_greedy_logits or need_mcts_logits:
                            planner_args = get_planner_args(args, batch, art_count, device, ds, **insert_specific_args)

                            if need_greedy_logits:

                                greedy_logits = get_greedy_logits(args, batch, chunk_batch, device, ds, lm, ns, planner, planner_args, nss)

                                if args.eval_pdistr_vs_kmeansdistr:
                                    update_pdistr_vs_kmeansdistr_state(art_count, batch, device, ds, greedy_logits, ns, state)

                                if args.eval_save_greedy_predictions:
                                    update_prediction_list(art_count, batch, chunk_batch_start, chunk_batch_end, ds, greedy_logits, ns)
                            if need_mcts_logits:
                                assert not args.plantoken, "Not implemented for plantoken"
                                mcts_logits, policy_output = planner.get_mcts_logits(**planner_args)
                            else:
                                mcts_logits, policy_output = None, None

                        # update_state_codes(policy_output, args, batch, pc_batch, device, lm, ns, planner,planner_args, state) # This also only needs updating where nss is true, but will just be overwritten with the same value where it is false (I think :P) so should be fine

                            if not args.plantoken and not args.fixed_code: # not implemented yet if plantoken // not relevant if fixed_code
                                update_state_ranks(greedy_logits, mcts_logits, ns, state, nss)
                                # update_state_accuracies(ns, state, nss)

                    if is_geneval(args):
                        def get_subbatch(b):
                            subb = {k: v[nss] for k, v in b.items()}
                            return {k: v[:,:get_last_nonpad_idx_per_row(subb['attention_mask']).max()+1] for k, v in subb.items()}
                        subbatch = get_subbatch(batch)
                        pc_subbatch = get_subbatch(pc_batch) if ns.fully_greedy else None
                        sub_art_idxs = chunk_art_idxs[nss]
                        subsnt_bounds = snt_bounds[nss]


                        snt_subidxs = get_snt_idxs(articles, sub_art_idxs, subbatch, subsnt_bounds, ns.insert_cc)
                        # if args.subsample_geneval != 1.0 and int(snt_idx) not in snt_idxs_to_generate_at:
                        if args.subsample_geneval != 1.0 and not is_in_sorted_list(ns.snt_idxs_to_generate_at, int(snt_subidxs)): # This will error if args.subsample_geneval != 1.0 and args.eval_batch_size > 1. But I don't need that combo immediately, so I'll just let it error for now 🤡
                            continue

                        state['geneval_count'] += subbatch_size
                        # Autoregressively generate args.generate_ntokens tokens
                        if args.generate_ntokens >= 0:
                            num_tkns = args.generate_ntokens
                        else:
                            num_tkns = sum([len(ts) for si, ts in zip(article['snt_idxs'], article['tokenized_sentences']) if si >= within_art_snt_subidxs]) # remaining_tkns_in_article

                        # fill_codes = [state[f'{prefix}_code'][nss] for prefix in ns.generation_code_types]
                        fill_codes = [torch.index_select(state[f'{basename(prefix)}_{c_or_s}'], 0, snt_subidxs) for prefix in ns.generation_code_types]

                        # cclm args
                        cc_lm_args = get_lm_args(args, subbatch, pc_subbatch, ds, lm, fill_codes,
                                                 fully_fixed_idx=ns.generation_code_types.index('fully_fixed') if 'fully_fixed' in ns.generation_code_types else None,
                                                 fully_greedy_idx=ns.generation_code_types.index('fully_greedy') if ns.fully_greedy else None,
                                                 uniform_idx = ns.generation_code_types.index('uniform') if 'uniform' in ns.generation_code_types else None)
                        cc_lm_args = trim_last_column(cc_lm_args)

                        # nocc lm args
                        nocc_lm_args = get_nocc_lm_args(cc_lm_args, ns, subbatch_size) if not args.only_match_eval else None

                        # context strings
                        context_strings = get_context_string(subbatch, ds.tokenizer, args.cc_type)
                        nocc_contex_strings = ds.tokenizer.decode(nocc_lm_args['input_ids'][0][
                                                                     nocc_lm_args['attention_mask'][
                                                                         0] != 0]) if ns.insert_cc else context_strings
                        if args.plantoken:
                            og_vocab_size = ... # TODO
                            cc_generated_tokens_and_codes = lm.generate(**cc_lm_args, num_return_sequences=1, max_new_tokens =num_tkns, temperature=1.0, pad_token_id=ds.tokenizer.eos_token_id).T # Transpose just cuz the else outputs transposed version, and code below counts on that
                            cc_generated_tokens_and_codes = cc_generated_tokens_and_codes[-num_tkns:]
                            # nocc_generated_tokens = lm.generate(**nocc_lm_args, num_return_sequences=1, max_length=num_tkns, temperature=1.0).T # No nocc for plantoken baseline
                            # Tokens produced can be both code-tokens (with id >= og vocab size), or normal tokens. We want to generate until we have num_tkns non-code tokens, in a while loop
                            # cc_lm_args = {k: torch.cat([v, cc_generated_tokens_and_codes.T if k != 'attention_mask' else torch.ones_like(cc_generated_tokens_and_codes.T)], dim=1)[:,-args.max_seq_len:] for k, v in cc_lm_args.items()}
                            for k, v in cc_lm_args.items():
                                if k == 'input_ids':
                                    new_val = cc_generated_tokens_and_codes.T
                                elif k == 'attention_mask':
                                    new_val = torch.ones_like(cc_generated_tokens_and_codes.T)
                                elif k == 'code_mask':
                                    new_val = cc_generated_tokens_and_codes.T >= og_vocab_size
                                cc_lm_args[k] = torch.cat([v, new_val], dim=1)[:,-args.max_seq_len:]

                            MAX_N_EXTRA_TOKENS = num_tkns
                            n_extra_tokens_tried = 0
                            cc_generated_tokens = [g[g < og_vocab_size] for g in cc_generated_tokens_and_codes.T]
                            # while len(generated_tokens) < num_tkns and n_extra_tokens_tried < MAX_N_EXTRA_TOKENS:
                            while any(len(g) < num_tkns for g in cc_generated_tokens) and n_extra_tokens_tried < MAX_N_EXTRA_TOKENS:
                                next_token = sample_next_token(args, lm, cc_lm_args)
                                cc_generated_tokens_and_codes = torch.cat([cc_generated_tokens_and_codes, next_token[None]], dim=0)
                                cc_generated_tokens = [g[g < og_vocab_size] for g in cc_generated_tokens_and_codes.T]
                                # cc_lm_args = {k: torch.cat([v, next_token.unsqueeze(1) if k != 'attention_mask' else v.new_ones(v.shape[0], 1)], dim=1)[:,-args.max_seq_len:] for k, v in cc_lm_args.items()}
                                for k, v in cc_lm_args.items():
                                    if k == 'input_ids':
                                        new_val = next_token.unsqueeze(1)
                                    elif k == 'attention_mask':
                                        new_val = v.new_ones(v.shape[0], 1)
                                    elif k == 'code_mask':
                                        new_val = (next_token >= og_vocab_size).unsqueeze(1)
                                    cc_lm_args[k] = torch.cat([v, new_val], dim=1)[:,-args.max_seq_len:]
                                n_extra_tokens_tried += 1
                            cc_generated_tokens = torch.nested.nested_tensor([g[:num_tkns] for g in cc_generated_tokens]).to_padded_tensor(-1)

                            # nocc_generated_tokens = generate_extra_tokens(nocc_generated_tokens)
                        else:
                            # planner args:
                            num_codes = len(fill_codes)
                            if ns.insert_cc:
                                input_ids_without_codes = subbatch['input_ids_and_codes'][~batch['code_mask']]
                                attn_mask_without_codes = subbatch['attention_mask'][~batch['code_mask']]
                            else:
                                input_ids_without_codes = subbatch['input_ids']
                                attn_mask_without_codes = subbatch['attention_mask']
                            planner_args = {'input_ids': input_ids_without_codes.repeat(num_codes, 1),
                                            'attention_mask': attn_mask_without_codes.repeat(num_codes, 1)}
                            planner_args = trim_last_column(planner_args)

                            gs = Namespace(cc_lm_args=cc_lm_args, nocc_lm_args=nocc_lm_args, planner_args=planner_args, generated_tokens=[], nocc_generated_tokens=[], code_locs=[[(0, c)] for cs in fill_codes for c in cs])

                            for gen_idx in range(num_tkns):
                                code_mask = gs.cc_lm_args['attention_mask'].clone()
                                cc_next_token = sample_next_token(args, lm, gs.cc_lm_args)
                                nocc_next_token = sample_next_token(args, lm, gs.nocc_lm_args) if not args.only_match_eval else None

                                update_generation_state_with_token(args, gs, lm, cc_next_token, nocc_next_token)

                                idxs, new_codes = get_new_codes_for_multiple_code_types(gs, planner, ns.should_plan_detector, args,
                                                                                        ns.generation_code_types, device,
                                                                                        num_possible_codes=lm.hparams.codebook.shape[0])

                                # update_generation_state_with_codes_OLD(gs, args, ns.generation_code_types, gen_idx, lm, `needs_new_code`, idxs, new_codes)
                                update_generation_state_with_codes(gen_idx, args, device, gs, idxs, new_codes, code_mask)

                            cc_generated_tokens = torch.stack(gs.generated_tokens).T
                            nocc_generated_tokens = torch.stack(gs.nocc_generated_tokens).T if not args.only_match_eval else None
                            code_locs = gs.code_locs

                        # region Get the true next strings
                        true_next_strs = get_true_next_strings(args, sub_art_idxs, articles, ds, ns, num_tkns, snt_subidxs, subbatch)
                        # endregion

                        # Maybe log generated sentences
                        extra_kwargs = {'cc_generated_tokens_and_codes': cc_generated_tokens_and_codes, 'og_vocab_size': og_vocab_size} if args.plantoken else {}

                        maybe_log_sentences(args, cc_generated_tokens, code_locs,
                                            context_strings, ds, fill_codes, logger, nocc_contex_strings,
                                            nocc_generated_tokens, ns, num_tkns, snt_subidxs,
                                            subbatch_size, true_next_strs,**extra_kwargs)

                        # generated_strings = [ds.tokenizer.decode(cc_generated_tokens[k,:][cc_generated_tokens[k,:] != -1]) for k in range(len(ns.generation_code_types))] + \
                        #                     ([ds.tokenizer.decode(nocc_generated_tokens[0,:])] if not args.plantoken else [])
                        if -1 in cc_generated_tokens:
                            raise NotImplementedError("TODO when needed")
                        else:
                            generated_strings = ds.tokenizer.batch_decode(cc_generated_tokens if args.only_match_eval else torch.cat((cc_generated_tokens, nocc_generated_tokens)))


                        if args.rev_ppl_eval:
                            store_rev_ppl_in_state(args, context_strings, device, generated_strings, ns.generation_code_types,
                                                   ns.judge_model, ns.judge_tokenizer, state, nocc_contex_strings)

                        if args.structure_levenshtein_eval or args.sbert_eval: # or args.rouge_eval:
                            regex_pattern = r"<\d+>"
                            true_next_str_no_codes = [re.sub(regex_pattern, " ", s) for s in true_next_strs]
                            # infixes = ["", "noctx_" if (context_strings == EOT_STRING) else "fullctx_" if (
                            #             EOT_STRING not in context_strings) else "somectx_"]
                            infixes = ["noctx_" if (cs == EOT_STRING) else "fullctx_" if (EOT_STRING not in cs) else "somectx_" for cs in context_strings]

                        def store_geneval_in_state(metric, str_transform, extra_kwargs, score_function):
                            true_sequences = [str_transform(s, **extra_kwargs) for s in true_next_str_no_codes]
                            # for pred_strs, prefix in zip(generated_strings, ns.generation_code_types + (['nocc'] if not args.plantoken else [])):
                            for p_idx, prefix in enumerate(ns.generation_code_types + (['nocc'] if (not args.plantoken and not args.only_match_eval) else [])):
                                pred_strs = generated_strings[p_idx*subbatch_size:(p_idx+1)*subbatch_size]
                                pred_sequences = [str_transform(s, **extra_kwargs) for s in pred_strs]
                                scores = [score_function(t, p) for t, p in zip(true_sequences, pred_sequences)]
                                score = sum(scores)
                                state[f'{prefix}_{metric}'] += score
                                for ix, s in zip(infixes, scores):
                                    state[f'{prefix}_{ix}{metric}'] += s
                            state[f'{metric}_count'] += subbatch_size
                            for ix in infixes:
                                state[f'{ix}{metric}_count'] += 1

                        if args.structure_levenshtein_eval:
                            store_geneval_in_state('levenshtein', ds.get_sequence_of_actions, {'embedder': ns.snt_embedder}, levenshtein_distance)

                        if args.sbert_eval:
                            store_geneval_in_state('sbert_score', ns.snt_embedder.encode, {'convert_to_numpy': False}, ns.cosine_similarity)

                        # if args.rouge_eval:
                        #     for rouge_metric in ['rouge-1', 'rouge-2', 'rouge-l']:
                        #         store_geneval_in_state(rouge_metric, lambda x: x, {}, lambda x, y: ns.rouge.get_scores(y, x)[0][rouge_metric]['f'])
                    if args.randae_nll_for_all == 'add':
                        add_rand_noise_to_action_embedders(lm, ns, state)

                    state['sent_count'] += subbatch_size

                # assert all([state[f'{prefix}_code'] is not None for prefix in ns.prefixes])

                if not args.skip_nll_eval:
                    # NLL metrics
                    if not args.plantoken and not scores_iso_codes(args):
                        ## NLL without code conditioning
                        if ds.cc_type == 'insert':
                            # add batch dim and move to device
                            nocc_chunk = prep_batch_from_chunk(args, chunks_without_codes[chunk_idx], lm.device, cc_type='anything-but-insert-:P')
                            nocc_input_ids = nocc_chunk['input_ids']
                            nocc_attn_mask = nocc_chunk['attention_mask']
                            nocc_label_mask = nocc_chunk['label_mask']
                        else:
                            nocc_input_ids = batch['input_ids']
                            nocc_attn_mask = batch['attention_mask']
                            nocc_label_mask = batch['label_mask']
                        nocc_labels = torch.masked_fill(nocc_input_ids, ~nocc_label_mask, -100)
                        nocc_lm_outputs = lm(input_ids=nocc_input_ids, codes=None, attention_mask=nocc_attn_mask)
                        state[f'total_nocc_nll'] += get_unreduced_nll(args, nocc_labels, nocc_lm_outputs).sum().item()

                    ## (In parallel) NLL with code conditioning using oracle, greedy, rand, fixed, and mcts codes

                    if not ns.only_nocc:
                        labels = get_nll_eval_labels(args, batch)
                        store_cc_nll_in_state(args, batch, pc_batch, ds, ns.extended_prefixes, lm, state, labels, snt_idxs)

                        if args.log_nll_for_all_codes:
                            raise NotImplementedError("Need to update below to batched eval, deal with when needed")
                            update_allnll_pertoken_state(args, batch, chunk, ds, ns.insert_cc, labels, lm, ns.randae_lm, state)

                    state['nll_count'] += BS
                state['article_count'] = max(chunk_art_idxs)

                # Log running averages every args.eval_freq chunks to not have to wait till end of run to get estimate
                if ((state['chunk_count'] // args.eval_batch_size) % (args.eval_freq // args.eval_batch_size)) == 0:
                    log_metrics(logger, state, ns.extended_prefixes, args, stage)

                # pbar.update(1)  # Increment the progress bar for each chunk processed
                pbar.update(len(chunk_batch))  # Increment the progress bar for each chunk processed
        # Save the predictions list to a pickle file if the flag is set
        if args.eval_save_greedy_predictions:
            ckpt_name = args.ckpts_wid if args.ckpts_wid is not None else args.mz_ckpt_wid
            with open(ckpt_name + '_greedy_predictions.pkl', 'wb') as f:
                pickle.dump(ns.predictions_list, f)
        log_metrics(logger, state, ns.extended_prefixes, args, stage)


def maybe_filter_chunks(args, art_idxs, chunks_with_codes, chunks_with_planner_codes, ns):
    if args.skip_nll_eval:

        snt_bounds = padstack([c['snt_bounds'] for c in chunks_with_codes]).to(ns.device)
        mask = torch.stack([c['attention_mask'] for c in chunks_with_codes]).to(ns.device)
        nss = new_sent_starts(mask, snt_bounds, args.cc_type)
        snt_start_idxs = nss.nonzero().squeeze(1)
        art_idxs = art_idxs.to(ns.device)[nss]
        chunks_with_codes = [c for i, c in enumerate(chunks_with_codes) if i in snt_start_idxs]
        chunks_with_planner_codes = [c for i, c in enumerate(chunks_with_planner_codes) if i in snt_start_idxs] if ns.fully_greedy else None

    return art_idxs, chunks_with_codes, chunks_with_planner_codes


def get_nocc_lm_args(cc_lm_args, ns, subbatch_size):
    if ns.insert_cc:
        raise NotImplementedError("TODO insert-cc for batched geneval")
        nocc_batch = prep_batch_from_chunk(args, chunks_without_codes[chunk_idx], device,
                                           cc_type='anything-but-insert-:P')
        nocc_lm_args = {'input_ids': nocc_batch['input_ids'], 'attention_mask': nocc_batch['attention_mask']}
        nocc_lm_args = trim_last_column(nocc_lm_args)
    else:
        nocc_lm_args = {k: v[0:subbatch_size] for k, v in cc_lm_args.items() if 'code' not in k}
    return nocc_lm_args


def maybe_log_sentences(args, cc_generated_tokens, code_locs, context_strings, ds,
                        fill_codes, logger, nocc_contex_strings, nocc_generated_tokens, ns, num_tkns,
                        snt_idx, subbatch_size, true_next_strs, og_vocab_size=None, cc_generated_tokens_and_codes=None):
    # if state['geneval_count'] % snt_log_freq == 0:


    subbatch_idxs_to_log = [i for i in range(subbatch_size) if is_in_sorted_list(ns.snt_idx_to_log_at, int(snt_idx[i]))]
    if len(subbatch_idxs_to_log) == 0:
        return

    rel_idxs = [(i - snt_idx[0]).item() for i in snt_idx[subbatch_idxs_to_log]]
    cc_generated_tokens_tolog = cc_generated_tokens[rel_idxs]
    cc_generated_tokens_and_codes_tolog = cc_generated_tokens_and_codes[rel_idxs] if cc_generated_tokens_and_codes is not None else None
    fill_codes_tolog = [fc[rel_idxs] for fc in fill_codes]
    context_strings_tolog = [context_strings[i] for i in subbatch_idxs_to_log]
    nocc_contex_strings_tolog = [nocc_contex_strings[i] for i in subbatch_idxs_to_log]
    subbatch_size_tolog = len(subbatch_idxs_to_log)
    nocc_generated_tokens_tolog = nocc_generated_tokens[rel_idxs] if nocc_generated_tokens is not None else None
    true_next_strs_tolog = [true_next_strs[i] for i in subbatch_idxs_to_log]
    code_locs_tolog = [code_locs[i] for i in subbatch_idxs_to_log]

    if args.plantoken:
        code_locs_tolog = get_plantoken_code_locs(cc_generated_tokens_tolog, cc_generated_tokens_and_codes_tolog,
                                            fill_codes_tolog, og_vocab_size)
    strings_for_log = context_strings_tolog
    if args.cc_type == 'insert' and not args.plantoken:
        strings_for_log.append(nocc_contex_strings_tolog)
    generated_strings_for_log = ["" for _ in range(len(ns.generation_code_types)) for _ in range(subbatch_size_tolog)]
    for k in range(len(generated_strings_for_log)):
        for idx, (start_idx, code) in enumerate(code_locs_tolog[k]):
            end_idx = code_locs_tolog[k][idx + 1][0] if idx + 1 < len(code_locs_tolog[k]) else num_tkns
            tkns_maybe_pad = cc_generated_tokens_tolog[k, start_idx:end_idx]
            generated_strings_for_log[k] += f"<{code}>" + ds.tokenizer.decode(tkns_maybe_pad[tkns_maybe_pad != -1])
    if not args.plantoken and nocc_generated_tokens_tolog is not None:
        generated_strings_for_log.append(ds.tokenizer.decode(nocc_generated_tokens_tolog[0, :]))
    strings_for_log += true_next_strs_tolog + generated_strings_for_log
    for c in range(subbatch_size_tolog):
        ns.generated_snts_table.add_data(*strings_for_log[c::subbatch_size_tolog])
    logger.log_metrics({"generated_snts": copy(ns.generated_snts_table)})  # Copy: following https://github.com/wandb/wandb/issues/2981#issuecomment-1458447291


def get_true_next_strings(args, art_idxs, articles, ds, ns, num_tkns, snt_idxs, subbatch):
    strs = []
    for i, (art_idx, snt_idx) in enumerate(zip(art_idxs,snt_idxs)):
        article = articles[art_idx]
        within_art_snt_idx = snt_idx - article['snt_idxs'][0]
        true_next_chunk = WikiDataset.get_chunks_at_art_and_snt_idxs(
            article_idx=art_idx,
            within_art_snt_idx=within_art_snt_idx,
            get_eval_code=False,
            max_seq_len=num_tkns if not ns.insert_cc else 2 * num_tkns,
            # Because we want at least num_tkns non-code tokens, and in insert_cc case, true_next_chunk will give max_seq_len (noncode+code tokens)
            unpacked_art=ds.unpack_article(article, ds.tokenizer.bos_token_id, args.cc_type),
            bos_id=ds.tokenizer.bos_token_id,
            cc_type=args.cc_type
        )
        if not ns.insert_cc and args.generate_ntokens >= 0:
            sanity_check_chunk_match(args, article, {k:v[i] for k,v in subbatch.items()}, ns.insert_cc, num_tkns, true_next_chunk)
        true_next_str = get_true_next_str(true_next_chunk, ds.tokenizer, num_tkns, ns.insert_cc)
        strs.append(true_next_str)
    return strs


def get_insert_specific_args(args, article, ds):
    unpacked_art = ds.unpack_article(article, ds.tokenizer.bos_token_id, args.cc_type)
    # smallest non-(-1)-value in unpacked_art.art_snt_ids
    snt_offset = min([el for el in unpacked_art.art_snt_ids if el != -1])
    return {'snt_offset': snt_offset, 'unpacked_art': unpacked_art}


# def update_state_for_new_article(ns, state, args, article):
#     # for prefix in ns.extended_prefixes:
#     #     # It could be in state if we resumed from a previous eval run
#     #     if f'{prefix}_code' not in state:
#     #         state[f'{prefix}_code'] = None
#
#     update_state_codes_for_art(args, article, ns, state)
#
#     update_state_accuracies_for_art(article, ns, state)
#
#
# def update_state_codes_for_art(args, article, ns, state):
#     def tensorify(list):
#         return torch.tensor(list).to(ns.device)
#
#     if not args.fixed_code:
#         if args.joint_planner_lm:
#             raise NotImplementedError("TODO JPL eval")
#         # state['greedy_code'] = pc_batch['planner_codes'][:, -1]
#         # if ns.fully_greedy:
#         state['art_greedy_codes'] = tensorify(article['planner_codes'])
#     if args.eval_with_path_search:
#         assert not args.plantoken, "Not implemented for plantoken"
#         raise NotImplementedError("Not implemented for batch size > 1")
#         search_code = planner.get_beamsearch_code(**planner_args)
#         state['beamsearch_code'] = search_code
#     if not ns.eval_only_policy_head:
#         raise NotImplementedError("Not implemented for batch size > 1")
#         from torch2jax import j2t
#         state['mcts_code'] = j2t(policy_output.action).to(ns.device)
#     # state['rand_code'] = torch.randint(size=(BS,), low=0, high=lm.hparams.codebook.shape[0]).to(device)
#     state['art_rand_codes'] = torch.randint(size=(len(article['tokenized_sentences']),), low=0, high=args.cluster_count).to(ns.device)
#     if not ns.insert_cc:
#         # state['oracle_code'] = batch['codes'][:, -1]
#         state['art_oracle_codes'] = tensorify(article['codes'])
#     else:
#         raise NotImplementedError("Not implemented for batch size > 1")
#         assert (batch['code_mask'][:, -2] == True).all()
#         state['oracle_code'] = batch['input_ids_and_codes'][:, -2]
#     # state['fixed_code'] = state['fully_fixed_code'] = torch.tensor(FIXED_CODE, dtype=torch.int32).to(device).repeat(BS)
#     state['art_fixed_codes'] = tensorify([FIXED_CODE] * len(article['tokenized_sentences']))

def update_state_codes_all(args, articles, ns, state):
    c_or_s = 'scores' if scores_iso_codes(args) else 'codes'
    def tensorify(list):
        return torch.tensor(list).to(ns.device)

    def add_to_state(code_or_score_type, codes):
        if code_or_score_type not in state:
            state[code_or_score_type] = codes
        else:
            state[code_or_score_type] = torch.cat([state[code_or_score_type], codes], dim=0)


    for article in articles:
        if not (args.fixed_code or args.uniform_mix or args.cc_type == 'none'):
            add_to_state(f'greedy_{c_or_s}', tensorify(article[f'planner_{c_or_s}']))
        if args.eval_with_path_search:
            assert not args.plantoken, "Not implemented for plantoken"
            raise NotImplementedError("Not implemented for batch size > 1")
            # search_code = planner.get_beamsearch_code(**planner_args)
            # state['beamsearch_code'] = search_code
        if not ns.eval_only_policy_head:
            raise NotImplementedError("Not implemented for batch size > 1")
            # from torch2jax import j2t
            # state['mcts_code'] = j2t(policy_output.action).to(ns.device)
        # state['art_rand_codes'] = torch.randint(size=(len(article['tokenized_sentences']),), low=0, high=args.cluster_count).to(ns.device)
        if scores_iso_codes(args):
            pass # Could implement equivalent of random code aka random scores, but not immediately needed
        else:
            add_to_state('rand_codes', torch.randint(size=(len(article['tokenized_sentences']),), low=0, high=args.cluster_count).to(ns.device))
        if not ns.insert_cc:
            # state['art_oracle_codes'] = tensorify(article['codes'])
            add_to_state('oracle_codes', tensorify(article['codes']))
        else:
            raise NotImplementedError("Not implemented for batch size > 1")
            assert (batch['code_mask'][:, -2] == True).all()
            # state['oracle_code'] = batch['input_ids_and_codes'][:, -2]
        # state['art_fixed_codes'] = tensorify([FIXED_CODE] * len(article['tokenized_sentences']))
        if not scores_iso_codes(args):
            add_to_state('fixed_codes', tensorify([FIXED_CODE] * len(article['tokenized_sentences'])))
        else:
            add_to_state('uniform_scores', torch.zeros(len(article['tokenized_sentences']), args.cluster_count).to(ns.device))

#
# def update_state_accuracies_for_art(article, ns, state):
#     correct = state['art_oracle_codes']
#     for prefix in ns.prefixes:
#         if prefix != 'oracle':
#             # correct_predictions = is_new_code_prediction * (state[f'{prefix}_code'] == oracle).float().sum()
#             correct_predictions = (state[f'art_{prefix}_codes'] == correct).float().sum()
#             state[f'{prefix}_accuracy_correct'] += correct_predictions
#
#         if prefix == "beamsearch":
#             # beamsearch_diff_greedy = is_new_code_prediction * (state['beamsearch_code'] != state['greedy_code']).float()
#             raise NotImplementedError("Not implemented for batch size > 1")
#             state['beamsearch_diff_greedy_count'] += beamsearch_diff_greedy
#     state[f'accuracy_count'] += len(article['tokenized_sentences'])

def log_accuracies_all(articles, ns, state, logger, args):
    correct = state['oracle_codes']
    to_log = {}
    to_log[f'accuracy_count'] = len([s for a in articles for s in a['tokenized_sentences']])
    for prefix in ns.prefixes:
        if prefix != 'oracle':
            pred_codes = state[f'{prefix}_codes'] if not scores_iso_codes(args) else state[f'{prefix}_scores'].argmax(-1)
            correct_predictions = (pred_codes == correct).float().sum()
            to_log[f'{prefix}_accuracy_correct'] = correct_predictions

        if prefix == "beamsearch":
            raise NotImplementedError("Not implemented for batch size > 1")

        if prefix not in ['fully_fixed', 'fully_greedy']:
            if prefix != 'oracle':
                logger.log_metrics({f'eval/lm_{prefix}_accuracy': to_log[f'{prefix}_accuracy_correct'] / to_log['accuracy_count']})


def update_state_codes(policy_output, args, batch, pc_batch, device, lm, ns, planner, planner_args, state):
    BS = list(batch.values())[0].shape[0]
    if not args.fixed_code:
        state['greedy_code'] = pc_batch['planner_codes'][:, -1]
        if ns.fully_greedy:
            state['fully_greedy_code'] = state['greedy_code']
    if args.eval_with_path_search:
        assert not args.plantoken, "Not implemented for plantoken"
        search_code = planner.get_beamsearch_code(**planner_args)
        state['beamsearch_code'] = search_code
    if not ns.eval_only_policy_head:
        from torch2jax import j2t
        state['mcts_code'] = j2t(policy_output.action).to(device)
    state['rand_code'] = torch.randint(size=(BS,), low=0, high=lm.hparams.codebook.shape[0]).to(device)
    if not ns.insert_cc:
        state['oracle_code'] = batch['codes'][:, -1]
    else:
        assert (batch['code_mask'][:, -2] == True).all()
        state['oracle_code'] = batch['input_ids_and_codes'][:, -2]
    state['fixed_code'] = state['fully_fixed_code'] = torch.tensor(FIXED_CODE, dtype=torch.int32).to(device).repeat(BS)


def get_greedy_logits(args, batch, chunk_batch, device, ds, lm, ns, planner, planner_args, nss):
    if not nss.all():
        raise NotImplementedError("TODO these should only be calculated for the rows where nss is True")
    if args.plantoken:
        lm_args_for_code_pred = get_lm_args(args, batch, chunk_batch, ds, lm)
        assert (batch['code_mask'][:, -2] == True).all()
        all_logits = lm(**lm_args_for_code_pred).logits[:, -2]
        code_logits = all_logits[...,
                      -lm.hparams.codebook.shape[0]:]  # Last codebook.shape[0] logits are for code tokens
        greedy_logits = code_logits
        # For now: don't log rank metrics and accuracy metrics yet in plantoken setting
    else:
        if not args.fixed_code and not ns.only_nocc:
            greedy_logits = planner.get_greedy_logits(**planner_args, cb=get_cb(args, device, lm))

    # Sanity check
    if 'planner_codes' in batch:
        assert greedy_logits.argmax(-1) == batch['planner_codes'][:, -1]
    return greedy_logits


def add_rand_noise_to_action_embedders(lm, ns, state):
    ns.randae_lm = deepcopy(lm)
    std = sum([p.std() for n, p in ns.randae_lm.named_parameters() if 'codebook' in n]) / sum(
        [1 for n, p in ns.randae_lm.named_parameters() if 'codebook' in n])
    for n, p in ns.randae_lm.named_parameters():
        if 'codebook' in n:
            oracle_code = state[f'oracle_code']
            if type(oracle_code) != int:
                oracle_code = oracle_code[..., None].item()
            oracle_emb = p[oracle_code]
            p.data = torch.randn_like(p) * std + oracle_emb


def update_state_ranks(greedy_logits, mcts_logits, ns, state, nss):
    logits_list = ([greedy_logits] if greedy_logits is not None else []) + ([mcts_logits] if not ns.eval_only_policy_head else [])
    for prefix in ns.prefixes:
        for rank_type, logits in zip(ns.rank_types, logits_list):
            if not nss.all():
                raise NotImplementedError("TODO these should only be calculated for the rows where nss is True")
            code = state[f'{prefix}_code'][..., None]
            state[f'total_{prefix}_{rank_type}'] += (logits.argsort(descending=True) == code).nonzero(as_tuple=True)[
                -1].float().mean()
    state['rank_count'] += 1


def update_state_accuracies(ns, state, is_new_code_prediction):
    oracle = state['oracle_code']
    for prefix in ns.prefixes:
        if prefix != 'oracle':
            correct_predictions = is_new_code_prediction * (state[f'{prefix}_code'] == oracle).float().sum()
            state[f'{prefix}_accuracy_correct'] += correct_predictions

        if prefix == "beamsearch":
            beamsearch_diff_greedy = is_new_code_prediction * (state['beamsearch_code'] != state['greedy_code']).float()
            state['beamsearch_diff_greedy_count'] += beamsearch_diff_greedy


def update_prediction_list(art_count, batch, chunk_batch_start, chunk_batch_end, ds, greedy_logits, ns):
    for rel_idx, chunk_idx in enumerate(range(chunk_batch_start, chunk_batch_end)):
        assert not ns.insert_cc, "Not implemented yet for insert_cc"
        # Convert context ids to text
        context_text = ds.tokenizer.decode(batch['input_ids'][rel_idx].cpu().numpy())
        # Convert policy_logits to numpy array
        policy_logits_numpy = greedy_logits[rel_idx].detach().cpu().numpy()
        oc = batch['codes'][:, -1]  # assumes we're not doing insert
        ns.predictions_list.append((art_count, chunk_idx, context_text, policy_logits_numpy, oc))


def update_pdistr_vs_kmeansdistr_state(art_count, batch, device, ds, greedy_logits, ns, state):
    flat_idx = ds.nested2flat_idx(art_count, batch['snt_idxs'][0][-1].item(), ns.split)
    emb = ns.embs[flat_idx]
    kmeans_scores = ds.kmeans.transform(emb[None])
    greedy_probs, kmeans_probs = [F.softmax(logits, dim=-1) for logits in
                                  [greedy_logits, torch.tensor(kmeans_scores).to(device)]]
    state['pdistr_vs_kmeansdistr_dist'] += (greedy_probs - kmeans_probs).abs().mean()
    state['pdistr_vs_kmeansdistr_KL'] += F.kl_div(greedy_probs.log(), kmeans_probs, reduction='batchmean')


def prep_ds_for_eval(args, ds, logger, ns, planner):
    if ns.fully_greedy:
        if args.cc_type != 'adapter':
            raise NotImplementedError(
                "Fully greedy eval only implemented for adapter cc_type at the moment. Set --skip_fully_greedy_eval to skip this error.")
        ds.add_planner_codes_or_scores(planner, args, logger, [ns.split])
    if args.eval_pdistr_vs_kmeansdistr:
        ds.load_kmeans_model(vars(args))


def get_eval_articles(args, ds, logger, ns, planner):
    articles = ds.split2coded_articles[ns.split][:1000]

    if args.eval_save_greedy_predictions:
        articles = articles[:100]
    return articles


def prep_ns(args, device, ds, lm, planner):
    ns = Namespace()

    ns.insert_cc = args.cc_type == 'insert'
    ns.only_nocc = args.cc_type == 'none'
    ns.eval_only_policy_head = args.only_policy_head or args.ns.eval_only_policy_head
    ns.fully_greedy = not ns.only_nocc and not args.plantoken and not args.skip_fully_greedy_eval and not args.fixed_code and not args.uniform_mix  # if plantoken, we haven't calculated these yet probably

    # Set prefixes: these are the code settings that we evaluate
    if not ns.only_nocc:
        ns.extended_prefixes = [get_relevant_prefix(args)]
        if not args.only_match_eval:
            ns.extended_prefixes = ['oracle'] + (['greedy'] if not args.fixed_code else []) + ['rand', 'fixed']
            if args.eval_with_path_search:
                ns.extended_prefixes += ['beamsearch']
            ns.extended_prefixes += [e for e in ['fully_fixed', 'fully_greedy'] if e not in ns.extended_prefixes]
        ns.prefixes = basenames(ns.extended_prefixes)
    else:
        ns.prefixes, ns.extended_prefixes = [], []
    # if ns.fully_greedy:
    #     ns.extended_prefixes += ['fully_greedy']


    ns.rank_types = ['grank'] if not ns.only_nocc else []
    if is_geneval(args):
        prep_geneval_vars(args, device, ds, ns.extended_prefixes, ns)
    if args.eval_save_greedy_predictions:
        ns.predictions_list = []
    if not ns.eval_only_policy_head:
        prep_mcts_vars(args, ns, planner)
    ns.split = get_eval_split(args)
    if args.randae_nll_for_all == 'replace':
        ns.randae_lm = get_randae_lm(lm)
    else:
        ns.randae_lm = None
    if args.eval_pdistr_vs_kmeansdistr:
        ns.embs = ds.load_split_embeddings(ns.split, None, None)
    ns.device = device
    return ns


def prep_mcts_vars(args, ns, planner):
    model_g, model_f, model_h = planner.model_g, planner.model_f, planner.model_h
    import mctx
    from jax import numpy as jnp
    from torch2jax import t2j
    jax_model_f, state_dict_f = t2j(model_f), {k: t2j(v) for k, v in planner.model_f.named_parameters()}
    jax_model_g, state_dict_g = t2j(model_g), {k: t2j(v) for k, v in planner.model_g.named_parameters()}
    ns.prefixes += ['mcts']
    ns.rank_types += ['srank']

    def recurrent_fn(params, rng_key, action, embedding):
        next_embedding, reward = jax_model_g((embedding, action), state_dict_g)
        prior_logits, value = jax_model_f(next_embedding, state_dict_f)
        recurrent_fn_output = mctx.RecurrentFnOutput(
            reward=reward,
            discount=jnp.ones_like(reward) * args.discount,
            prior_logits=prior_logits,
            value=value)
        return recurrent_fn_output, next_embedding

    ns.recurrent_fn = recurrent_fn


def prep_geneval_vars(args, device, ds, extended_prefixes, ns):
    ns.should_plan_detector = UnigramEOSDetector(ds.tokenizer.name_or_path)
    ns.generation_code_types = get_generation_code_types(extended_prefixes)
    ns.generated_snts_table = wandb.Table(
        columns=["context"] + (['nocc context'] if (ns.insert_cc and not args.plantoken) else []) + [
            "true"] + ns.generation_code_types + (["no_cc"] if (not args.plantoken and not args.only_match_eval) else []))

    if args.rev_ppl_eval:
        from transformers import AutoModelForCausalLM
        from tokenizer_util import myAutoTokenizer
        ns.judge_tokenizer = myAutoTokenizer(args.judge_model)
        quantization_config = get_quantization_config(args, model_name=args.judge_model)
        ns.judge_model = AutoModelForCausalLM.from_pretrained(args.judge_model,
                                                              quantization_config=quantization_config)
        ns.judge_model.eval()
    if args.sbert_eval or args.structure_levenshtein_eval:
        from sentence_transformers import SentenceTransformer
        from torchmetrics import CosineSimilarity
        ns.snt_embedder = SentenceTransformer(args.embedder_name).to(device)
        ns.cosine_similarity = CosineSimilarity(reduction='mean')

    # if args.rouge_eval:
    if 'rouge' in args.startfullctx_metrics:
        from rouge import Rouge
        ns.rouge = Rouge()

    articles = ds.split2coded_articles[get_eval_split(args)][:1000]
    total_n_snts = sum([len(a['tokenized_sentences']) for a in articles])
    # snt_log_freq = max(total_n_snts // args.n_logged_snts, 1)
    np.random.seed(args.seed); ns.snt_idx_to_log_at = sorted(np.random.choice(range(total_n_snts), args.n_logged_snts, replace=False))
    if args.subsample_geneval != 1.0:
        np.random.seed(args.seed); ns.snt_idxs_to_generate_at = sorted(np.random.choice(range(total_n_snts), int(total_n_snts * args.subsample_geneval), replace=False))
        np.random.seed(args.seed); ns.snt_idx_to_log_at = sorted(np.random.choice(ns.snt_idxs_to_generate_at, min(args.n_logged_snts, len(ns.snt_idxs_to_generate_at)), replace=False))


def get_eval_split(args):
    split = 'val' if not args.testeval else 'test'
    return split


def is_geneval(args):
    return args.rev_ppl_eval or args.sbert_eval or args.structure_levenshtein_eval or args.startfullctx_geneval or args.noctx_geneval


def sanity_check_chunk_match(args, article, chunk, insert_cc, num_tkns, true_next_chunk):
    input_ids_key = 'input_ids_and_codes' if insert_cc else 'input_ids'
    mx_len = min(args.max_seq_len - 1, num_tkns)
    a = true_next_chunk[f'ctx_{input_ids_key}'][true_next_chunk['ctx_attention_mask'] != 0][-mx_len:]
    b = chunk[input_ids_key][chunk['attention_mask'] != 0][:-1][-mx_len:]
    assert torch.equal(a.to(b.device), b), f"true_next_chunk and chunk don't match: {a} vs {b} in article {article}"


def get_snt_idxs(articles, art_idxs, batch, snt_bounds, insert_cc):
    last_nonpad_idx = get_last_nonpad_idx_per_row(batch['attention_mask'])
    device = batch['snt_idxs'].device
    if not insert_cc:
        # contains_last_snt = batch['snt_idxs'][:,-1] == article['snt_idxs'][-1]
        # snt_idx = batch['snt_idxs'][-1] if contains_last_snt or not ((last_nonpad_idx + 1) in batch['snt_bounds']) else batch['snt_idxs'][-2]  # Second one in edge case of one-token sentence. See https://photos.app.goo.gl/44jVTH1A6JE3XyzC8

        contains_last_snt = batch['snt_idxs'][:, -1] == torch.tensor([articles[art_idx]['snt_idxs'][-1] for art_idx in art_idxs]).to(device)
        edge_case = ((last_nonpad_idx + 1)[:,None] == snt_bounds).any(dim=-1)
        snt_idxs = torch.where(contains_last_snt | edge_case, batch['snt_idxs'][:, -1], batch['snt_idxs'][:, -2])
    else:
        # snt_idx = batch['snt_idxs'][:,last_nonpad_idx - 1]
        snt_idxs = torch.gather(batch['snt_idxs'], -1, last_nonpad_idx[:, None]).squeeze(-1)
        assert (snt_idxs != -1).any()
    return snt_idxs


def update_allnll_pertoken_state(args, batch, chunk, ds, insert_cc, labels, lm, randae_lm, state):
    assert not insert_cc, "Not implemented yet for insert_cc"
    fill_codes = range(args.cluster_count)
    lm_full_args = get_lm_args(args, batch, chunk, ds, lm, fill_codes)
    rep_labels = labels.repeat(len(fill_codes), 1)
    MAX_BATCH_SIZE = 256 if torch.cuda.mem_get_info()[0] > 20 * 1e9 else 128
    all_nlls = []
    for cc_idx in range(0, args.cluster_count, MAX_BATCH_SIZE):
        def subidx(el):
            return el[cc_idx:cc_idx + MAX_BATCH_SIZE] if el is not None else None

        cc_lm_args = {k: subidx(v) for k, v in lm_full_args.items()}
        if args.randae_nll_for_all is not None:
            batch_lm_outputs = randae_lm(**cc_lm_args)
        else:
            batch_lm_outputs = lm(**cc_lm_args)

        batch_nlls = get_unreduced_nll(args, subidx(rep_labels), batch_lm_outputs)
        del batch_lm_outputs
        all_nlls.extend(batch_nlls.tolist())
        del batch_nlls
        # if MAX_BATCH_SIZE == 512:
        # torch.cuda.empty_cache() # Not sure why needed, but needed
    state['all_nlls_per_sentence'] += np.array(all_nlls)


def update_allnll_persentence_state(extended_prefixes, state):
    sum_all_nlls_last_snt = state['all_nlls_per_sentence']
    state['total_ordered_nlls'] += np.array(sorted(sum_all_nlls_last_snt))
    if not all(sum_all_nlls_last_snt == 0):
        for prefix in extended_prefixes:
            code = state[f'{prefix}_code']
            if type(code) != int:
                code = code[..., None].item()
            # Get the rank of the nll at position [code] in sum_all_nlls_last_snt
            rank = (np.array(sum_all_nlls_last_snt) < sum_all_nlls_last_snt[code]).sum()
            state[f'total_{prefix}_nllrank'] += rank
    state['all_nlls_per_sentence'] = np.zeros_like(state['all_nlls_per_sentence'])


def get_randae_lm(lm):
    '''
    Replaces all codebook weights in a model with random normal weights with the same std as the codebook weights
    '''
    randae_lm = deepcopy(lm)
    # get std() of all weights called codebook
    std = sum([p.std() for n, p in randae_lm.named_parameters() if 'codebook' in n]) / sum(
        [1 for n, p in randae_lm.named_parameters() if 'codebook' in n])
    # replace each codebook with random normal with std
    for n, p in randae_lm.named_parameters():
        if 'codebook' in n:
            p.data = torch.randn_like(p) * std
    return randae_lm


def get_cb(args, device, lm):
    if is_embedding_space_prediction(args.mz_policy_loss_fn):
        cb = lm.hparams.codebook
        cb_device = "cpu" if args.no_cluster else device
        cb = torch.tensor(cb, device=cb_device)
    else:
        cb = None
    return cb


def get_plantoken_code_locs(cc_generated_tokens, cc_generated_tokens_and_codes, fill_codes, og_vocab_size):
    code_locs_tAc = [torch.nonzero(g >= og_vocab_size, as_tuple=True)[0] for g in cc_generated_tokens_and_codes.T]
    code_locs = []
    for outer_idx in range(len(fill_codes)):
        locs_tAc = code_locs_tAc[outer_idx]
        locs = [(0, fill_codes[outer_idx])]
        last_loc_tAc = 0
        n_noncode_tokens = len(cc_generated_tokens[outer_idx][cc_generated_tokens[outer_idx] != -1])
        for loc_tAc in locs_tAc:
            ts_between = max(0, (
                        loc_tAc.item() - last_loc_tAc) - 1)  # E.g., if loc_tAc is 0 and 2, that means there is one non-code token between them
            last_loc = locs[-1][0]
            new_loc = last_loc + ts_between
            if new_loc >= n_noncode_tokens:  # This means these codes had location behind the last noncode-token, we don't care about those:
                break
            locs.append((last_loc + ts_between, cc_generated_tokens_and_codes[loc_tAc, outer_idx].item()))
            last_loc_tAc = loc_tAc.item()
        code_locs.append(locs)
    return code_locs


def store_cc_nll_in_state(args, batch, pc_batch, ds, extended_prefixes, lm, state, labels, snt_idxs):
    assert args.eval_stride == 1, "eval_stride > 1 not supported: need to figure out which predicted/oracle codes to give"
    c_or_s = 'scores' if scores_iso_codes(args) else 'codes'
    # fill_codes = [intify(state[f'{prefix}_code']) for prefix in extended_prefixes]
    # fill_codes = [state[f'{prefix}_code'] for prefix in extended_prefixes]
    fill_codes = [torch.index_select(state[f'{basename(prefix)}_{c_or_s}'], 0, snt_idxs) for prefix in extended_prefixes]
    cc_lm_args = get_lm_args(args, batch, pc_batch, ds, lm, fill_codes,
                             fully_fixed_idx=extended_prefixes.index('fully_fixed') if 'fully_fixed' in extended_prefixes else None,
                             fully_greedy_idx=extended_prefixes.index('fully_greedy') if 'fully_greedy' in extended_prefixes else None,
                             uniform_idx=extended_prefixes.index('uniform') if 'uniform' in extended_prefixes else None)
    try:
        lm_outputs = lm(**cc_lm_args)
    except Exception as e:
        print(f"Error in cc_lm_args: {cc_lm_args}")
        raise e
    rep_labels = labels.repeat(len(fill_codes), 1)
    nlls = get_unreduced_nll(args, rep_labels, lm_outputs)
    # for prefix, nll in zip(extended_prefixes, nlls):
    BS = batch['attention_mask'].shape[0]
    for p_idx, prefix in enumerate(extended_prefixes):
        nll = nlls[p_idx*BS:(p_idx+1)*BS].sum()
        state[f'total_{prefix}_nll'] += nll.item()


def get_nll_eval_labels(args, batch):
    if args.cc_type == 'insert':
        # if args.plantoken:
        #     labels = torch.masked_fill(batch['input_ids_and_codes'], ~batch['label_mask'], -100)
        # else:
        # Actually, in contrast to during training, we DO want to mask away code-locations during eval
        labels = torch.masked_fill(batch['input_ids_and_codes'],
                                   ~torch.logical_and(batch['label_mask'], ~batch['code_mask']), -100)
    else:
        labels = torch.masked_fill(batch['input_ids'], ~batch['label_mask'], -100)
    return labels


def store_nocc_nll_in_state(args, batch, chunks_without_codes, ds, chunk_idx, lm, state):
    if ds.cc_type == 'insert':
        # add batch dim and move to device
        nocc_chunk = prep_batch_from_chunk(args, chunks_without_codes[chunk_idx], lm.device, cc_type='anything-but-insert-:P')
        nocc_input_ids = nocc_chunk['input_ids']
        nocc_attn_mask = nocc_chunk['attention_mask']
        nocc_label_mask = nocc_chunk['label_mask']
    else:
        nocc_input_ids = batch['input_ids']
        nocc_attn_mask = batch['attention_mask']
        nocc_label_mask = batch['label_mask']
    nocc_labels = torch.masked_fill(nocc_input_ids, ~nocc_label_mask, -100)
    nocc_lm_outputs = lm(input_ids=nocc_input_ids, codes=None, attention_mask=nocc_attn_mask)
    state[f'total_nocc_nll'] += get_unreduced_nll(args, nocc_labels, nocc_lm_outputs).item()


def get_true_next_str(true_next_chunk, tokenizer, num_tkns, insert_cc):
    if insert_cc:
        true_next_tokens_and_codes = true_next_chunk['input_ids_and_codes']
        code_idxs = true_next_chunk['code_mask'].nonzero(as_tuple=True)[0]

        true_next_str = ""
        start = 0
        noncode_token_count = 0
        for i, el in enumerate(true_next_tokens_and_codes):
            if i in code_idxs:
                noncode_substr = tokenizer.decode(
                    true_next_tokens_and_codes[start:i][true_next_chunk['attention_mask'][start:i] != 0])
                code_substr = f"<{el.item()}>"
                true_next_str += noncode_substr + code_substr
                start = i + 1
            else:
                noncode_token_count += 1
                if noncode_token_count > num_tkns:
                    break

        final_noncode_substr = tokenizer.decode(
            true_next_tokens_and_codes[start:i][true_next_chunk['attention_mask'][start:i] != 0])
        true_next_str += final_noncode_substr
    else:
        true_next_tokens = true_next_chunk['input_ids']

        true_next_str = ""
        start = 0
        for code, sb in zip(true_next_chunk['codes'], true_next_chunk['snt_bounds']):
            true_next_str += f"<{code.item()}>"
            true_next_str += tokenizer.decode(
                true_next_tokens[start:sb.item()][true_next_chunk['attention_mask'][start:sb.item()] != 0])
            start = sb.item()
    return true_next_str


def trim_last_column(lm_args):
    '''
    The last token is already the first token of the upcoming sentence, used for NLL, but not needed for e.g., gen_eval
    '''
    if not lm_args['attention_mask'].all():
        result = trim_last_nonpad_element(lm_args)
        return result

    return {k: v[:, :-1] for k, v in lm_args.items()}


def trim_last_nonpad_element(lm_args):
    lnpi = get_last_nonpad_idx_per_row(lm_args['attention_mask'])
    dct = deepcopy(lm_args)
    for k in lm_args:
        if not any([c in k for c in ['codes','snt_idxs','scores']]):  # For codes, the "padding" is just a repeat of the last element
            k2pad = {
                'input_ids': 42,
                'attention_mask': 0,
                'label_mask': False
            }
            mask = torch.zeros_like(dct[k], dtype=torch.bool)
            # mask[torch.arange(dct[k].size(0)), lnpi] = True This sets all rows to true for each element in lnpi, but we want only corresponding row to true
            mask[torch.arange(dct[k].size(0)), lnpi] = True
            dct[k][mask] = k2pad[k]
    result = {k: v[:, :-1] for k, v in
              dct.items()}  # ALSO want to trim the last element (which should be replaced with padding)
    return result


def update_generation_state_with_codes_OLD(gs, args, generation_code_types, gen_idx, lm, needs_new_code, idxs, new_codes):
    # region Update inputs with new codes
    device = lm.device
    # gs.code_locs = [el + [(gen_idx + 1, nc.item())] if needed else el for el, needed, nc in
    #                 zip(gs.code_locs, needs_new_code, new_codes)]
    for i, idx in enumerate(idxs):
        gs.code_locs[idx].append((gen_idx, new_codes[i]))
    if args.cc_type == 'insert':
        raise NotImplementedError("TODO when needed. Make compatible with parallel")
        og_mask = gs.cc_lm_args['attention_mask'].clone()
        if needs_new_code.any():
            k2args = {
                'code_mask': {'pad_value': 0, 'new_value': torch.ones_like(new_codes, dtype=torch.bool)},
                'inputs_embeds': {'pad_value': 0, 'new_value': lm.base_model.codebook[new_codes]},
                'attention_mask': {'pad_value': 0, 'new_value': torch.ones_like(new_codes)}
            }
            for k, v in k2args.items():
                gs.cc_lm_args[k] = update_maybe_padded_tensor(gs.cc_lm_args[k], v['new_value'], og_mask, args, v['pad_value'],
                                                              needs_new_code)
    else:
        # last_codes = torch.tensor([gs.code_locs[k][-1][1] for k in range(len(generation_code_types))]).to(device)
        last_codes = torch.tensor([gs.code_locs[k][-1][1] for k in range(len(gs.code_locs))]).to(device)
        # codes_to_append = torch.where(needs_new_code, new_codes, last_codes)
        # gs.cc_lm_args['codes'] = update_tensor(gs.cc_lm_args['codes'], gs.cc_lm_args['attention_mask'], codes_to_append,
        #                                        args)
        codes_to_append = last_codes
        if len(idxs) > 0:
            codes_to_append[idxs] = new_codes
        gs.cc_lm_args['codes'] = update_tensor(gs.cc_lm_args['codes'], gs.cc_lm_args['attention_mask'], codes_to_append,
                                               args)
    # endregion


def get_new_codes_for_multiple_code_types(gs, planner, should_plan_detector, args, generation_code_types, device, num_possible_codes):
    needs_new_code = should_plan_detector.detect_should_plan(gs.generated_tokens)
    num_codes = len(generation_code_types)
    bs = gs.planner_args['attention_mask'].shape[0] // num_codes
    # new_codes = torch.full([num_codes*bs], -1).to(device)
    # for idx, code_type, needs_new_code_at_idx in zip(range(len(generation_code_types)), generation_code_types, needs_new_code):
    new_codes_or_scores_all = torch.tensor([]).to(device)
    new_idxs = torch.tensor([]).to(device)
    for code_type_idx, code_type in enumerate(generation_code_types):
        needs_new_code_at_codetypeidxs = needs_new_code[code_type_idx*bs:(code_type_idx+1)*bs]
        idxs, new_codes_or_scores = torch.tensor([]).to(device), torch.tensor([]).to(device)
        if code_type not in ['fixed', 'fully_fixed', 'uniform']:
            idxs = torch.where(needs_new_code_at_codetypeidxs)[0] + code_type_idx*bs
            if len(idxs) > 0:
                new_codes_or_scores = get_new_codes_at_idxs(args, idxs, code_type, num_possible_codes, planner, gs)
                # endregion
        new_idxs = torch.cat([new_idxs, idxs.to(device)])
        new_codes_or_scores_all = torch.cat([new_codes_or_scores_all, new_codes_or_scores.to(device)])

    return new_idxs.to(torch.int64), new_codes_or_scores_all


def get_new_codes_at_idxs(args, idxs, code_type, num_possible_codes, planner, gs):
    # region compute new code
    if code_type == 'rand':
        new_codes_or_scores = torch.randint(size=(len(idxs),), low=0, high=num_possible_codes) if not scores_iso_codes(args) else ... # DO when needed
    elif code_type in ['greedy', 'mcts', 'beamsearch', "fully_greedy"]:
        # planner_args_spec = {k: v[code_type_idx:code_type_idx + 1] for k, v in planner_args.items()}
        planner_args_spec = {k: v[idxs] for k, v in gs.planner_args.items()}
        if code_type == 'greedy' or 'fully_greedy':
            new_codes_or_scores = get_greedy_new_codes_or_scores(args, planner, planner_args_spec)
        elif code_type == 'mcts':
            from torch2jax import j2t
            mcts_logits, policy_output = planner.get_mcts_logits(**planner_args_spec)
            new_codes_or_scores = j2t(policy_output.action).to(planner_args_spec['attention_mask'].device)
        elif code_type == 'beamsearch':
            new_codes_or_scores = planner.get_beamsearch_code(**planner_args_spec)
    else:
        raise ValueError(f"Unknown code type: {code_type}")
    return new_codes_or_scores


def sample_next_token(args, lm, m_args):
    lm_outputs = lm(**m_args)
    last_nonpad_idx = get_last_nonpad_idx_per_row(m_args['attention_mask'])
    next_token_logits = lm_outputs.logits[range(len(last_nonpad_idx)), last_nonpad_idx]
    if args.temperature == 0:
        next_token = next_token_logits.argmax(-1)
    else:
        next_token = torch.multinomial(torch.softmax(next_token_logits / args.temperature, dim=-1), num_samples=1)[:, 0]
    return next_token


def update_generation_state_with_token(args, gs, lm, next_token=None, nocc_next_token=None):
    lst = []
    if next_token is not None:
        gs.generated_tokens.append(next_token)
        lst.append([gs.cc_lm_args, next_token])
    if 'planner_args' in gs:
        lst.append([gs.planner_args, next_token])
    if nocc_next_token is not None:
        lst.append([gs.nocc_lm_args, nocc_next_token])
        gs.nocc_generated_tokens.append(nocc_next_token)
    assert len(lst) > 0
    for m_args, new_token in lst:
        if 'input_ids' in m_args:
            keys_to_update = ['input_ids']
        else:
            keys_to_update = ['inputs_embeds', 'code_mask']
        keys_to_update += ['attention_mask']
        og_mask = m_args['attention_mask'].clone()
        for k in keys_to_update:
            key2to_append = {
                'input_ids': new_token,
                'attention_mask': torch.ones_like(new_token),
                'code_mask': torch.zeros_like(new_token, dtype=torch.bool),
                'inputs_embeds': lm.wte(new_token)
            }
            # m_args[k] = update_tensor(m_args[k], m_args['attention_mask'], key2to_append[k], args)
            m_args[k] = update_maybe_padded_tensor(m_args[k], key2to_append[k], og_mask, args)


def update_tensor(tensor, mask, new_value, args):
    if get_last_nonpad_idx_per_row(mask)[0] + 1 < mask.shape[1]:
        # raise NotImplementedError("Right-padding not implemented: assuming no padding")  # idea: extend the tensor by one, replace with new token at last_nonpad_idx, make copies: one that cuts the last element off (if last_nonpad_idx was < length), on that cuts FIRST element off. Then use torch.where based on whether last_nonpad_idx was < length or not
        raise NotImplementedError("Make sure use the right version of mask (pre or post-updated")
    else:
        result = torch.cat([tensor, new_value.unsqueeze(1)], dim=1)
        if result.shape[1] > args.max_seq_len:
            result = result[:, 1:]
        return result

def update_maybe_padded_tensor(tensor, new_value,  og_mask,  args, pad_value=None,new_value_mask=None):
    assert (pad_value is None) == (new_value_mask is None)
    if pad_value is None:
        pad_value = 0
        new_value_mask = torch.ones(len(new_value), dtype=torch.bool).to(new_value.device)
    assert (new_value_mask == 1).any()
    # Add dimensions to new_value_mask so it has the same number of dimensions as new_value
    ndims_extra = new_value.ndim - 1
    new_value_mask = new_value_mask.reshape( new_value_mask.shape + (1,)*ndims_extra)
    new_value_or_pad = torch.where(new_value_mask, new_value, pad_value + torch.zeros_like(new_value))
    if (og_mask[:,-1] == 1).all(): # Case: there was no right-padding in original tensor
        result = torch.cat([tensor, new_value_or_pad.unsqueeze(1)], dim=1)
    else: # Case: there is some right-padding. In that case, we want to append the new values after the last non-pad values in each row
        last_nonpad_idx = get_last_nonpad_idx_per_row(og_mask)
        rows = [torch.cat([og_t[:idx+1], new_t[None]]) if should else og_t[:idx+1] for og_t, new_t, idx, should in zip(tensor, new_value_or_pad, last_nonpad_idx, new_value_mask)]
        warnings.filterwarnings("ignore", message="The PyTorch API of nested tensors is in prototype stage and will change in the near future.")
        nested_tensor = torch.nested.nested_tensor(rows)
        result = nested_tensor.to_padded_tensor(pad_value)

    if result.shape[1] > args.max_seq_len:
        # result = result[:, 1:]
        # We want to trim the first element for nonpadded rows, but the last (padding) element for padded rows
        is_padded = og_mask[:,-1] == 0
        result = torch.where(is_padded[:,*(None,)*(len(result.shape) -1)], result[:,:-1], result[:,1:])
    return result.to(tensor.dtype)


def get_context_string(batch, tokenizer, cc_type):
    if cc_type != 'insert':
        if not batch['attention_mask'].all():
            result = trim_last_nonpad_element(batch)
            input_ids, attention_mask = result['input_ids'], result['attention_mask']
        else:
            input_ids, attention_mask = batch['input_ids'][:,:-1], batch['attention_mask'][:,:-1] # :-1 to remove the last token, which is the first token of the next sentence. That token is relevant for NLL but not for generation
        return [tokenizer.decode(input_ids[i][attention_mask[i] != 0]) for i in range(len(input_ids))]
    else:
        raise NotImplementedError("TODO implement non-one batch size for insert_cc")
        # certain tokens are codes rather than words. We don't want to use the tokenizer for those, but just decode them as <<<code>>>. Their position is given by the code_mask
        input_ids_and_codes, attention_mask, code_mask = batch['input_ids_and_codes'][:,:-1], batch['attention_mask'][:,:-1], batch['code_mask'][:,:-1]
        result = ""
        assert input_ids_and_codes.shape[0] == 1, "Only one example at a time supported"

        if code_mask[0, -1]: # If the last token is a code, we want to remove it, because that code won't be used as context for gen-eval (rather, predicted codes will be used)
            input_ids_and_codes = input_ids_and_codes[:, :-1]
            code_mask = code_mask[:, :-1]

        # Check that nothing matching <<<digits>>> is in the input
        input_ids_only = input_ids_and_codes[0, ~code_mask[0]]
        assert not re.search(r'<<<\d+>>>', tokenizer.decode(input_ids_only))

        for i in range(input_ids_and_codes.shape[1]):
            if code_mask[0,i]:
                result += f"<<<{input_ids_and_codes[0,i]}>>>"
            else:
                result += tokenizer.decode(input_ids_and_codes[0,i:i+1])
        return result



def prep_batch_from_chunk(args, chunk_batch, device, cc_type=None):
    if cc_type is None:
        cc_type = args.cc_type
    # batch = {k: v.unsqueeze(0).to(device) for k, v in chunk.items()}  # add batch dim and move to device
    chunk_batch = [(MyDataset.expand_chunk(chunk, 'deduce',cc_type, args) if cc_type in ['none','adapter'] else chunk) for chunk in chunk_batch]
    batch = {k: torch.stack([c[k] for c in chunk_batch]).to(device) for k in chunk_batch[0] if k != 'article_idx'}
    if args.eval_stride == 1:
        # remove padding
        last_nonpad_idx = get_last_nonpad_idx_per_row(batch['attention_mask'])
        max_last_nonpad_idx = last_nonpad_idx.max()
        # for k in paddable_keys_for_cc_type(cc_type):s
        for k in batch:
        #     batch[k] = batch[k][:, :last_nonpad_idx + 1]
            batch[k] = batch[k][:, :max_last_nonpad_idx + 1]
    return batch


def paddable_keys_for_cc_type(cc_type):
    result = ['attention_mask', 'label_mask']
    if cc_type == 'insert':
        result += ['code_mask', 'snt_idxs', 'input_ids_and_codes']
    else:
        result += ['input_ids']
    return result

def store_rev_ppl_in_state(args, context_string, device, generated_strings, generation_code_types, judge_model,
                           judge_tokenizer, state, nocc_contex_string):
    judge_tokenized = [judge_tokenizer.encode(s) for s in generated_strings]
    judge_context_tokenized = judge_tokenizer.encode(context_string)
    judge_tokenized = [judge_context_tokenized + jt for jt in judge_tokenized]
    # Pad to max length among tokenized
    max_len = max([len(jt) for jt in judge_tokenized])
    attention_mask = torch.tensor([[1] * len(jt) + [0] * (max_len - len(jt)) for jt in judge_tokenized]).to(device)
    input_ids = torch.tensor([jt + [42] * (max_len - len(jt)) for jt in judge_tokenized]).to(
        device)  # 42 ignored anyway
    label_mask = torch.tensor(
        [[0] * len(judge_context_tokenized) + [1] * (len(jt) - len(judge_context_tokenized)) + [0] * (max_len - len(jt))
         for jt in judge_tokenized]).to(device).to(torch.bool)
    labels = torch.masked_fill(input_ids, ~label_mask, -100)
    judge_outputs = judge_model(input_ids, attention_mask)
    nlls = get_unreduced_nll(args, labels, judge_outputs, only_last_stride=False, leftpad_possible=True)

    spec_prefix = 'noctx_' if context_string == EOT_STRING else 'fullctx_' if EOT_STRING not in context_string else 'somectx_'
    for p in ["", spec_prefix]:
        for k in range(len(generation_code_types)):
            state[f'{generation_code_types[k]}_{p}rev_nll'] += nlls[k].item()
        if not args.plantoken:
            state[f'nocc_{p}rev_nll'] += nlls[-1].item()
        state[f'{p}rev_nll_count'] += 1


def get_generation_code_types(extended_prefixes):
    return [e for e in extended_prefixes if e != 'oracle']


def get_planner_args(args, batch, art_count=None, device=None, ds=None, snt_offset=None, unpacked_art=None):
    if args.cc_type == 'insert':
        last_nonpad_idxs = get_last_nonpad_idx_per_row(batch['attention_mask'])
        global_snt_idx = batch['snt_idxs'][:last_nonpad_idxs+1][:, -2]
        assert global_snt_idx != -1
        local_snt_idx = global_snt_idx - snt_offset
        context_chunk_for_planner = ds.get_chunks_at_art_and_snt_idxs(art_count, local_snt_idx, get_eval_code=False,
                                                                      max_seq_len=ds.max_seq_len,
                                                                      unpacked_art=unpacked_art, bos_id=ds.tokenizer.bos_token_id,
                                                                      cc_type=args.cc_type)
        ctx4planner_input_ids = context_chunk_for_planner['ctx_input_ids_and_codes'][None].to(device)
        ctx4planner_attention_mask = torch.logical_and(context_chunk_for_planner['ctx_attention_mask'],
                                                       ~context_chunk_for_planner['ctx_code_mask'])[None].to(device)
        # Remove columns that are all padding
        whole_column_is_pad_mask = ctx4planner_attention_mask.any(dim=0) # 0 if whole column is pad, 1 otherwise
        ctx4planner_input_ids = ctx4planner_input_ids[:, whole_column_is_pad_mask]
        ctx4planner_attention_mask = ctx4planner_attention_mask[:, whole_column_is_pad_mask]
    else:
        ctx4planner_input_ids, ctx4planner_attention_mask = (batch['input_ids'][:, :-args.eval_stride],
                                                             batch['attention_mask'][:,:-args.eval_stride])
    planner_args = {'input_ids': ctx4planner_input_ids, 'attention_mask': ctx4planner_attention_mask}
    return planner_args

class UnigramEOSDetector:

    def __init__(self, tokenizer_name):
        with open(jn(DEFAULT_PICKLE_DIR,f'EOS_ids_{tokenizer_name.replace("/", "_")}.json')) as f:
            self.eos_ids = torch.tensor(json.load(f))
    def detect_should_plan(self, generated):
        last_generated = generated[-1]
        if self.eos_ids.device != last_generated.device:
            self.eos_ids = self.eos_ids.to(last_generated.device)
        # Detect if should invoke planner (currently: when sentence ends)
        return (last_generated[:].unsqueeze(-1) == self.eos_ids).any(-1)

def get_lm_args(args, batch, pc_batch, ds, lm, fill_codes=None, fully_fixed_idx=None, fully_greedy_idx=None, uniform_idx=None):
    # Fills in each of fill_codes in a variant of the original batch, in the place of the latest oracle code
    lm_args = {}

    if fill_codes is None:
        assert args.plantoken
        lm_args |= {
            'input_ids': batch['input_ids_and_codes'],
            # 'inputs_embeds': lm.get_embeds(batch['code_mask'], batch['input_ids_and_codes']),
                   'code_mask': batch['code_mask'],
                   'attention_mask': batch['attention_mask']}
    else:
        num_codes = len(fill_codes)
        if ds.cc_type == 'insert':
            raise NotImplementedError("TODO finish implementing that deals with a chunk batch iso a single chunk")
            code_mask = batch['code_mask']
            ids_and_codes = batch['input_ids_and_codes']

            rep_code_mask = code_mask.repeat(num_codes, 1)
            most_recent_code_mask = get_most_recent_code_mask(batch)
            inp_ora_spec_ids = \
                torch.cat([torch.masked_fill(ids_and_codes,
                                             most_recent_code_mask if (i != fully_fixed_idx) else code_mask,
                                             c) # no need to handle this for fully_greedy since not defined
                           for i, c in enumerate(fill_codes)], dim=0)
            if not args.plantoken:
                inp_ora_spec_embs = lm.get_embeds(code_mask, inp_ora_spec_ids)
                lm_args['inputs_embeds'] = inp_ora_spec_embs
            else:
                # codes can be passed to plantoken model either with within-codebook index (eg between 0 and 1023) or beyond-vocab-idx (eg between 50257 and 51280).
                # During lm.generate(), beyond-vocab-idx method is used. Here, we ensure that the starting input for generate is also in beyond-vocab-idx
                vocab_size = lm.lm_head.weight.shape[0]; codebook_size = lm.base_model.codebook.shape[0]
                assert (inp_ora_spec_ids[rep_code_mask] < codebook_size).all()
                inp_ora_spec_ids = torch.where(rep_code_mask, inp_ora_spec_ids + vocab_size, inp_ora_spec_ids)
                lm_args['input_ids'] = inp_ora_spec_ids
            lm_args['code_mask'] = rep_code_mask
        else:
            codes_for_lm_args = get_codes_for_lm_args(args, batch, fill_codes, fully_fixed_idx, fully_greedy_idx, num_codes, pc_batch, uniform_idx)
            rep_input_ids = batch['input_ids'].repeat(num_codes, 1)
            lm_args |= {'input_ids':                                                rep_input_ids,
                        'codes' if not scores_iso_codes(args) else 'planner_scores': codes_for_lm_args}
        rep_attn_masks = batch['attention_mask'].repeat(num_codes, 1)
        lm_args['attention_mask'] = rep_attn_masks
    return lm_args


def get_codes_for_lm_args(args, batch, fill_codes, fully_fixed_idx, fully_greedy_idx, num_codes, pc_batch, uniform_idx):
    codes_for_lm_args = [None for _ in range(num_codes)]
    # oracle_codes_and_specific_code = get_oracle_codes_and_specific_code(args, batch, pc_batch, ds, fill_codes,
    #                                                                     planner_codes_idx=[fully_greedy_idx] if fully_greedy_idx is not None else None)
    c_or_s = 'scores' if scores_iso_codes(args) else 'codes'
    if len(fill_codes) > 1 or (fully_greedy_idx is None and fully_fixed_idx is None and uniform_idx is None):
        oracle_codes_per_token = batch['codes']
        snt_idxs_per_token = batch['snt_idxs']
        snt_idx_of_next_token = snt_idxs_per_token[..., -2:-1]
        codes_for_lm_args = torch.stack([torch.where(snt_idxs_per_token == snt_idx_of_next_token,
                                                     torch.ones_like(oracle_codes_per_token) * c[:, None],
                                                     oracle_codes_per_token)
                                         for c in fill_codes], dim=0)

        if scores_iso_codes(args):
            raise NotImplementedError(
                "TODO finish implementing that deals with a chunk batch iso a single chunk")
            # change og codes into one-hot scores
            t = F.one_hot(result, num_classes=exp_chunk_planner['planner_scores'].shape[-1]).to(
                exp_chunk_planner['planner_scores'].dtype)
            t[t == 1] = 1e9;
            t[t == 0] = -1e9
    if fully_greedy_idx is not None:
        codes_for_lm_args[fully_greedy_idx] = pc_batch[f'planner_{c_or_s}']
    if fully_fixed_idx is not None:
        if not scores_iso_codes(args):
            codes_for_lm_args[fully_fixed_idx] = FIXED_CODE * torch.ones_like(batch['codes'])
        else:
            t = torch.zeros_like(batch['codes']).unsqueeze(-1).repeat(1, 1, args.num_clusters)
            t += -1e9
            t[FIXED_CODE] = 1e9
            codes_for_lm_args[fully_fixed_idx] = t
    if uniform_idx is not None:
        assert scores_iso_codes(args)
        codes_for_lm_args[uniform_idx] = torch.zeros_like(batch['codes']).unsqueeze(-1).repeat(1, 1, args.cluster_count)
    if type(codes_for_lm_args) == list:
        codes_for_lm_args = torch.stack(codes_for_lm_args, dim=0)
    last_nonpad_idx = get_last_nonpad_idx_per_row(batch['attention_mask'])
    codes_for_lm_args = codes_for_lm_args[:, :, :last_nonpad_idx.max() + 1].to(batch['input_ids'].device)
    # reshape oracle_codes_and_specific_code from [NCODES, BATCHSIZE, SEQLEN] to [NCODES*BATCHSIZE, SEQLEN]
    codes_for_lm_args = codes_for_lm_args.view(-1, *codes_for_lm_args.shape[2:])
    return codes_for_lm_args


def get_oracle_codes_and_specific_code(args, batch, pc_batch, ds, fill_codes, planner_codes_idx=None):
    result = torch.zeros(len(fill_codes)).to(batch['input_ids'].device)
    c_or_s = 'scores' if scores_iso_codes(args) else 'codes'
    if (planner_codes_idx is None and not args.uniform_mix) or len(fill_codes) > 1:
        oracle_codes_per_token = batch['codes']
        snt_idxs_per_token = batch['snt_idxs']
        snt_idx_of_next_token = snt_idxs_per_token[..., -2:-1]
        result = torch.stack([torch.where(snt_idxs_per_token == snt_idx_of_next_token,
                                                                  torch.ones_like(oracle_codes_per_token) *c[:,None],
                                                                  oracle_codes_per_token)
                                                      for c in fill_codes], dim=0)

        if scores_iso_codes(args):
            raise NotImplementedError("TODO finish implementing that deals with a chunk batch iso a single chunk")
            # change og codes into one-hot scores
            t = F.one_hot(result, num_classes=exp_chunk_planner['planner_scores'].shape[-1]).to(exp_chunk_planner['planner_scores'].dtype)
            t[t == 1] = 1e9; t[t == 0] = -1e9
    if planner_codes_idx is not None:
        # # exp_chunk_planner = ds.expand_chunk(chunk_batch, code_type=f'planner_{"scores" if args.joint_planner_lm else "codes"}')
        # if args.joint_planner_lm:
        #     raise NotImplementedError("TODO finish implementing that deals with a chunk batch iso a single chunk")
        #     # change og codes into one-hot scores
        #     t = F.one_hot(oracle_codes_and_specific_code, num_classes=exp_chunk_planner['planner_scores'].shape[-1]).to(exp_chunk_planner['planner_scores'].dtype)
        #     t[t == 1] = 1e9; t[t == 0] = -1e9
        #     t[planner_codes_idx[0]] = exp_chunk_planner['planner_scores']
        #     oracle_scores_and_specific_score = t
        #     return oracle_scores_and_specific_score
        # else:
        #     # oracle_codes_and_specific_code[planner_codes_idx[0]] = exp_chunk_planner['planner_codes']
        result[planner_codes_idx[0]] = pc_batch[f'planner_{c_or_s}']
    # if is list, cat into tensor
    if isinstance(result, list):
        result = torch.stack(result, dim=0)
    return result


def prep_models_for_eval(lm, planner, device='cuda'):
    noplanner = (planner is None)
    if not noplanner:
        model_g, model_f, model_h = planner.model_g, planner.model_f, planner.model_h
    else:
        model_g, model_f, model_h = None, None, None
    lm.eval()
    if planner is not None:
        planner.eval()
    lm.to(device)
    # move all params in model_g, model_f and model_h to device if the model is not None
    for m in [model_g, model_f, model_h]:
        if m is not None:
            m.to(device)


def init_state(args, extended_prefixes, logger, prefixes, stage):
    '''
    Initialize state for evaluation, either from scratch or from a resumed run
    '''
    if logger.experiment.resumed and f'eval/state{stage}' in logger.experiment.summary._as_dict():
        state = logger.experiment.summary[f'eval/state{stage}']._items
        if args.log_nll_for_all_codes:
            for key_to_numpyify in ['total_ordered_nlls', 'all_nlls_per_sentence']:
                assert key_to_numpyify + '_list' in state
                state[key_to_numpyify] = np.array(state[key_to_numpyify + '_list'])
        # if 'total_ordered_nlls' in state:
        #     assert 'total_ordered_nlls_list' in state
        #     state['total_ordered_nlls'] = np.array(state['total_ordered_nlls_list'])
        check_start = True
    else:
        state = {'total_nocc_nll': 0, 'nll_count': 0, 'chunk_count': 0, 'article_count': 0, 'sent_count': 0}
        if not args.cc_type == 'none':
            state |= {f'total_{prefix}_{metric}': 0 for prefix in prefixes for metric in ['srank', 'grank', 'nll']} | \
                    {'total_fully_fixed_nll': 0} | ({} if args.skip_fully_greedy_eval else {'total_fully_greedy_nll':0}) | \
                    {'rank_count': 0, 'accuracy_count': 0} | \
                    {f'{prefix}_accuracy_correct': 0 for prefix in prefixes} | \
                    {'beamsearch_diff_greedy_count': 0}
        if args.plantoken:
            # Filter out keys with 'srank', 'grank', 'rank_count', 'accuracy_correct', 'beamsearch_diff_greedy_count' in them: not implemented (yet) for plantoken
            state = {k: v for k, v in state.items() if not any([el in k for el in ['srank', 'grank', 'rank_count', 'accuracy_correct', 'beamsearch_diff_greedy_count']])}
        def add_generation_state(state, scoretype):
            for prefix in ["","noctx_", "somectx_", "fullctx_"]:
                state |= {f'{prefix}{scoretype}_count': 0, f'nocc_{prefix}{scoretype}': 0} | {f'{t}_{prefix}{scoretype}': 0 for t in get_generation_code_types(extended_prefixes)}
            return state

        if args.rev_ppl_eval:
            state = add_generation_state(state, "rev_nll")
        if args.sbert_eval:
            state = add_generation_state(state, "sbert_score")
        if args.structure_levenshtein_eval:
            state = add_generation_state(state, "levenshtein")
        if is_geneval(args):
            state['geneval_count'] = 0

        if args.log_nll_for_all_codes:
            state |= ({'total_ordered_nlls': np.zeros(args.cluster_count),
                       'all_nlls_per_sentence': np.zeros(args.cluster_count)} |
                      {f'total_{prefix}_nllrank': 0 for prefix in extended_prefixes})

        if args.eval_pdistr_vs_kmeansdistr:
            state |= {'pdistr_vs_kmeansdistr_dist': 0,
                      'pdistr_vs_kmeansdistr_KL': 0
                      }

        check_start = False
    start_chunk_count = 0 if check_start else None
    return check_start, state, start_chunk_count


def get_most_recent_code_mask(batch):
    # last element of each row that is not equal to -1
    snt_idx_of_next_token = torch.max(batch['snt_idxs'], dim=1)[0]  # Per row maximum element
    # If the max is -1, it means the row was all -1. We don't want to match with those, so set them to -100
    snt_idx_of_next_token[snt_idx_of_next_token == -1] = -100
    most_recent_code_mask = (batch['snt_idxs'] == snt_idx_of_next_token)
    return most_recent_code_mask




def get_unreduced_nll(args, labels, outputs, only_last_stride=True, leftpad_possible=False):
    lm_logits = outputs.logits
    # Shift so that tokens < n predict n
    shift_logits = lm_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    if only_last_stride:
        assert not leftpad_possible, "Not supported yet" # Would need to use last_nontoken_idxs = get_last_nonpad_idx_per_row((labels != -100).to(torch.int32), leftpad_possible)
        # assert (labels[:, -1 ] != -100).all(), "Right-padding not supported yet here, only no-padding"
        if (shift_labels != -100).all():
            shift_logits = shift_logits[:, -args.eval_stride:]
            shift_labels = shift_labels[:, -args.eval_stride:]
        else:
            lnpi = get_last_nonpad_idx_per_row((shift_labels != -100).to(torch.int32), leftpad_possible)
            shift_labels = torch.stack([shift_labels[i, lnpi[i] + 1 - args.eval_stride:lnpi[i] + 1] for i in range(len(lnpi))])
            shift_logits = torch.stack([shift_logits[i, lnpi[i] + 1 - args.eval_stride:lnpi[i] + 1] for i in range(len(lnpi))])
    nll_per_token = CrossEntropyLoss(reduction='none')(shift_logits.view(-1, shift_logits.shape[-1]),
                                                       shift_labels.view(-1)).view(shift_labels.shape)
    # Calculate NLL mask based on where labels = -100
    nll_mask = (shift_labels != -100).float()
    sum_nll_of_last_stride_tokens = (nll_per_token * nll_mask).sum(-1)
    nll_of_last_stride_tokens = sum_nll_of_last_stride_tokens / nll_mask.sum(-1)
    return nll_of_last_stride_tokens


def log_metrics(logger, state, extended_prefixes, args, stage):
    relevant_prefix = get_relevant_prefix(args)

    if not args.skip_nll_eval:
        for prefix in ['nocc'] + extended_prefixes:
            logger.log_metrics({f'eval/lm_{prefix}_perplexity': np.exp(state[f'total_{prefix}_nll'] / state['nll_count'])}, step=state['chunk_count'])

        logger.log_metrics({f'eval/relevant_perplexity': np.exp(state[f'total_{relevant_prefix}_nll'] / state['nll_count'])}, step=state['chunk_count'])

    logger.log_metrics({'eval/lm_article_count': state['article_count']}, step=state['chunk_count'])
    logger.log_metrics({'eval/sent_count': state['sent_count']}, step=state['chunk_count'])
    if not args.fixed_code and not args.cc_type == 'none':
        for rank_type in ['srank', 'grank']:
            for prefix in extended_prefixes:
                if 'rank_count' in state and state['rank_count'] > 0:
                    if prefix not in ['fully_fixed', 'fully_greedy']:
                        logger.log_metrics({f'eval/lm_{prefix}_{rank_type}': state[f'total_{prefix}_{rank_type}'] / state['rank_count']}, step=state['chunk_count'])

                        if prefix == "beamsearch":
                            logger.log_metrics({f'eval/beamsearch_doesnt_match_greedy': (state['beamsearch_diff_greedy_count'] / state['rank_count']) * 100}, step=state['chunk_count'])

    def log_gen(type):
        for ix in ["", "noctx_", "somectx_", "fullctx_"]:
            for prefix in get_generation_code_types(extended_prefixes) + ['nocc']:
                if state[f'{ix}{type}_count'] > 0:
                    logger.log_metrics({f'eval/lm_{prefix}_{ix}{type}': state[f'{prefix}_{ix}{type}'] / state[f'{ix}{type}_count']}, step=state['chunk_count'])
            if state[f'{ix}{type}_count'] > 0:
                logger.log_metrics({f'eval/relevant_{ix}{type}': state[f'{relevant_prefix}_{ix}{type}'] / state[f'{ix}{type}_count']}, step=state['chunk_count'])

    if args.rev_ppl_eval:
        log_gen("rev_nll")
    if args.sbert_eval:
        log_gen("sbert_score")
    if args.structure_levenshtein_eval:
        log_gen("levenshtein")
    # if args.rouge_eval:
    #     log_gen("rouge-1")
    #     log_gen("rouge-2")
    #     log_gen("rouge-l")

    if args.log_nll_for_all_codes:
        for prefix in extended_prefixes:
            logger.log_metrics({f'eval/lm_{prefix}_nllrank': state[f'total_{prefix}_nllrank'] / state['nll_count']}, step=state['chunk_count'])

        # log ordered_nlls as plot with rank on x-axis and nll on y-axis
        ordered_nlls = (state['total_ordered_nlls'] + np.array(sorted(state['all_nlls_per_sentence']))) / state['nll_count'] # Could be mid-sentence, so add the average per-sentence nll
        ordered_ppls = np.exp(ordered_nlls)
        data = [[x, y] for (x, y) in zip(list(range(args.cluster_count)), ordered_ppls)]
        logger.log_metrics({'eval/lm_ordered_ppls': wandb.Table(data=data, columns=["rank", "PPL"])}, step=state['chunk_count'])


    if args.eval_pdistr_vs_kmeansdistr:
        logger.log_metrics({'eval/pdistr_vs_kmeansdistr_dist': state['pdistr_vs_kmeansdistr_dist'] / state['rank_count']}, step=state['chunk_count'])
        logger.log_metrics({'eval/pdistr_vs_kmeansdistr_KL': state['pdistr_vs_kmeansdistr_KL'] / state['rank_count']}, step=state['chunk_count'])


    # Store metrics dict so we can resume if crashed mid-evaluation
    # if 'total_ordered_nlls' in state:
    #     state['total_ordered_nlls_list'] = state['total_ordered_nlls'].tolist() # because storing numpy arrays is not supported
    # More general: detect if value is numpy array and convert to list if so
    keys, values = list(state.keys()), list(state.values())
    for k, v in zip(keys, values):
        if isinstance(v, np.ndarray):
            state[k + '_list'] = v.tolist()

    logger.experiment.summary[f'eval/state{stage}'] = state


def get_relevant_prefix(args):
    relevant_prefix = 'fully_greedy'
    if args.fixed_code:
        relevant_prefix = 'fully_fixed'
    if args.cc_type == 'none':
        relevant_prefix = 'nocc'
    if args.uniform_mix:
        relevant_prefix = 'uniform'
    return relevant_prefix


def new_sent_starts(mask, snt_bounds, cc_type,snt_idxs=None):
    '''
    For each batch element, return whether it starts a new sentence
    '''
    # mask = batch['attention_mask']
    last_nonpad_idxs = get_last_nonpad_idx_per_row(mask)
    if cc_type != 'insert':
        # snt_bounds = batch['snt_bounds']

        # first sentence if all but first two tokens have attention_mask == 1

        # first_sent = not(mask[:, 2:].any()) and mask[:, :2].all()
        # Above only returns a single true/false, and only works for batch size one. We want a tensor of true/false for each batch element
        first_sent = ~mask[:, 2:].any(dim=-1) & mask[:, :2].all(dim=-1)


        # start of non-first sentence if one of the snt_bounds == (index of the first 0 in attention_mask) - 1
        # non_first_sent = False
        # for row in range(len(last_nonpad_idxs)):
        #     if (snt_bounds[row] == last_nonpad_idxs[row]).any():
        #         non_first_sent = True
        #         break
        # Above only returns a single true/false, and only works for batch size one. We want a tensor of true/false for each batch element
        # Snt_bounds is a 2D tensor
        non_first_sent = (snt_bounds == last_nonpad_idxs[:, None]).any(dim=-1)


        # result = first_sent or non_first_sent
        result = first_sent | non_first_sent
    else:
        # result = (batch['snt_idxs'][:last_nonpad_idxs+1][:,-2] != -1).any()
        # Above only returns a single true/false, and only works for batch size one. We want a tensor of true/false for each batch element
        result = (snt_idxs[:last_nonpad_idxs+1][:,-2] != -1).any(dim=-1)
        raise ValueError("TODO check whether this is correct :P")
    return result


def get_last_nonpad_idx_per_row(attention_mask, leftpad_possible=False):
    has_padding = (attention_mask == 0).any(dim=-1)
    if not leftpad_possible:
        last_nonpad_idxs = torch.where(has_padding,
                                   attention_mask.argmin(-1),
                                   attention_mask.shape[1]) - 1
    else:
        assert len(attention_mask.unique()) <= 2, "Left-padding not implemented for more than two unique values in attention_mask"
        last_nonpad_idxs = torch.tensor([attention_mask[i].nonzero().max() if has_padding[i] else -1 for i in range(len(attention_mask))]).to(attention_mask.device)
    return last_nonpad_idxs


def padstack(tensors):
    # Get the maximum length among all tensors
    max_len = max(len(tensor) for tensor in tensors)

    # Pad tensors with -1 to make them of equal length
    padded_tensors = [F.pad(tensor, pad=(0, max_len - len(tensor)), value=-1) for tensor in tensors]

    # Stack padded tensors
    stacked_tensors = torch.stack(padded_tensors)

    return stacked_tensors


def basename(prefix):
    return prefix[6:] if prefix.startswith('fully_') else prefix

def basenames(prefixes):
    # basename, make unique, keep order
    return list(OrderedDict.fromkeys([basename(prefix) for prefix in prefixes]))