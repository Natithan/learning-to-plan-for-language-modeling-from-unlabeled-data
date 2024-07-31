import os
import getpass
from os.path import join as jn
from jsonargparse import ArgumentParser

from constants import DEFAULT_PICKLE_DIR, DEFAULT_CKPT_DIR, SF_CACHE_DIR, SHORT2FULL_EMBEDDER_NAME, DEFAULT_EMBEDDER, \
    OTHER_CKPT_DIR
from glob import glob


def get_args():
    parser = ArgumentParser()

    # dataset args
    parser.add_argument("--kmeans_cluster_debug", action="store_true")
    parser.add_argument("--max_articles", type=int, default=285310,
                        help="Number of articles to sample from Wikipedia. If negative, use all articles.")
    parser.add_argument("--data_name", type=str, default="newenwiki")
    parser.add_argument("--cluster_count", type=int, default=1024)
    parser.add_argument("--out_dir", type=str, default=DEFAULT_PICKLE_DIR)
    parser.add_argument("--sbert_batch_size", type=int, default=256,
                        help="Batch size for creating oracle sentence-embedding codes with SBERT.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cc_param_seed", type=int, default=None,
                        help="Seed for initializing the code-conditioning parameters. If None, use the same seed as for the main seed.")
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--force_remake_chunks", action="store_true")
    parser.add_argument("--kmeans_path", type=str, default=None,
                        help="If given, load kmeans from this path. Else, get it from the traiing data.")
    parser.add_argument("--embedder_name", type=str, default=DEFAULT_EMBEDDER, choices=SHORT2FULL_EMBEDDER_NAME.keys(),
                        help="SentenceTransformer model for sentence-based representation function.")
    parser.add_argument("--rand_codebook", action="store_true",
                        help="Use random codebook instead of kmeans centroids. Still use kmeans indices however.")
    parser.add_argument("--stream", type=bool, default=None,
                        help="Stream data instead of loading it all into memory.")
    parser.add_argument("--buffer_size", type=int, default=1000,
                        help="Number of chunks to buffer in memory when streaming.")

    # language model args
    parser.add_argument("--base_model_name", type=str, default="gpt2")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training language model.")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--accumulate_lm_grad_batches", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--data_pkl_path", type=str)
    parser.add_argument("--no_cluster", action="store_true")
    parser.add_argument("--unclustered_pkl_path", type=str)
    parser.add_argument("--skip_lm_load", type=bool, default=True, help="Skip getting an lm.")
    parser.add_argument("--skip_plm_load", action="store_true", help="Skip getting planner-trained lm (presumably if getting jplm).")
    parser.add_argument("--skip_lm_train", action="store_true", help="Get an lm, but don't train it.")
    parser.add_argument("--skip_plm_train", action="store_true")
    parser.add_argument("--skip_lm_ckpt_from_wid", action="store_true",
                        help="Skip loading lm ckpt from folder indicated by ckpts_wid.")
    parser.add_argument("--skip_plm_ckpt_from_wid", action="store_true",
                        help="Skip loading plm ckpt from folder indicated by ckpts_wid.")
    parser.add_argument("--lm_epochs", type=int, default=1)
    parser.add_argument("--lm_ckpt_path", type=str, default=None)
    parser.add_argument("--plm_ckpt_path", type=str, default=None)
    parser.add_argument("--lm_ckpt_wid", type=str, default=None)
    parser.add_argument("--plm_ckpt_wid", type=str, default=None)
    parser.add_argument("--fixed_code", action="store_true",
                        help="Use a fixed code for all sentences. This is a sanity check to see if the improvement is only due to the added parameters rather than the code conditioning.")
    parser.add_argument("--init_gate", choices=['zero', 'one', 'xavier_normal'], default='zero',
                        help="Initialization for code-conditioning gate in LMFT.")
    parser.add_argument("--no_gate", type=bool, default=True, help="Don't use code-conditioning gate in LMFT.")
    parser.add_argument("-c", "--cc_type", choices=['adapter', 'insert', 'add_to_embedding','none'], default='adapter',
                        help="Type of code-conditioning to use in LMFT.")
    parser.add_argument("--log_nll_per_relative_tkn_idx", action="store_true",
                        help="Log nll per relative token index during LMFT.")
    parser.add_argument("--tune_lm_with_planner", type=bool, default=True, help="Tune LM with planner-predicted codes.")
    parser.add_argument("--unfreeze_codebook_after_step", type=int, default=0,
                        help="Unfreeze codebook after this step. If negative, never unfreeze.")  # Very slight improvement if unfreezing later, but not worth
    parser.add_argument("-f", "--og_params_finetune_type", type=str, default="freeze",
                        choices=["freeze", "lora", "full_finetune"],
                        help="Type of finetuning to use for original parameters in LMFT.")
    parser.add_argument("--insert_no_proj", type=bool, default=True,
                        help="Flag to make insert-style cc closer to the plantoken baseline: still use external planner, but don't have learnt projection of codebook embeddings. Instead, like plantoken, just append codebook embeddings to token embeddings, and finetune whole model (or part with lora)")
    parser.add_argument("--adapter_extra_context", type=str, default=None, choices=["concat", "gate"])
    parser.add_argument("--randinit_lm_code_embedder", action="store_true",
                        help="Randomly initialize the code embedder in the language model.")
    parser.add_argument("--uniform_mix", action="store_true",
                        help="Use uniform mixing of codebook embeddings in LMFT. Basically the mixed equivalent of fixed_code.")

    # jpl args
    parser.add_argument("--jplm_ckpt_wid", type=str, default=None)
    parser.add_argument("--jplm_ckpt_path", type=str, default=None)
    parser.add_argument("--skip_jplm_ckpt_from_wid", action="store_true",
                        help="Skip loading jplm ckpt from folder indicated by ckpts_wid.")
    parser.add_argument("--skip_jplm_train", action="store_true")
    parser.add_argument("--jpl_freeze_planner", action="store_true", help="Freeze the planner during jplm training.")
    parser.add_argument("--jpl_freeze_planner_sentence_transformer", action="store_true", help="Freeze the sentence transformer part of the planner during jplm training.")
    parser.add_argument("--jpl_unfreeze_planner_after_step_or_frac", type=str, default="0", help="Unfreeze planner after this step (if int given) or fraction of total steps (if float given). If negative, never unfreeze.")
    parser.add_argument("--jpl_freeze_lm", action="store_true",
                        help="Freeze the language model during jplm training.")
    parser.add_argument("--straight_through", action="store_true",
                        help="Use straight-through selector for interface planner and lm.")
    parser.add_argument("--jpl_temperature", type=float, default=1.0,
                        help="Temperature used in the softmax at the interface between planner and lm.")
    parser.add_argument("--jpl_gumbel", action="store_true",
                        help="Use Gumbel-Softmax for the interface between planner and lm.")
    parser.add_argument("--gumbel_scale", type=float, default=1.0, help="Multiplier for the gumbel samples added to logits. If 1.0 (default), equal to theoretical gumbel-softmax samples.")
    # planner args
    parser.add_argument("--mz_multi_vector_representation", type=bool, default=True)
    parser.add_argument("--mz_hidden_size", type=int, default=768)
    # parser.add_argument("--mz_h_tune_top_layers", type=int, default=0)
    parser.add_argument("--freeze_mz_h", action="store_true")
    parser.add_argument("--mz_h_sent_model", type=str, default=None,
                        help="SentenceTransformer model for sentence-based representation function. If not specified, defaults to same as --embedder_name")
    parser.add_argument("--mz_h_model_name", type=str, default=None,
                        help="Model to use for mzpt. If None, use args.base_model_name.")
    parser.add_argument("--mz_h_sent_model_cluster", action="store_true",
                        help="If set, project sentence embeddings to most nearby codebook entry.")
    parser.add_argument("--mz_h_extra_transformer_layers", type=int, default=0,
                        help="Number of extra transformer layers to add to the model.")
    parser.add_argument("--mz_f_hidden_size", type=int, default=768)
    parser.add_argument("--mz_f_num_layers", type=int, default=1)
    parser.add_argument("--mz_f_dropout_p", type=float, default=0.0)
    parser.add_argument("--mz_f_skip_and_norm", action="store_true")
    parser.add_argument("--mz_g_num_layers", type=int, default=1)
    parser.add_argument("--mz_g_dropout_p", type=float, default=0.0)
    parser.add_argument("--mz_g_skip_and_norm", action="store_true")
    parser.add_argument("--mz_policy_loss_fn", type=str, default="ce", choices=["ce", "mse"])
    parser.add_argument("--mz_regression_loss_transform", type=str, default=None, choices=["linear", "mlp"])
    parser.add_argument("--mz_policy_loss_fn_distance_function", type=str, default="cosine",
                        choices=["cosine", "euclidean", "contrastive"])
    parser.add_argument("--accumulate_mz_grad_batches", type=int, default=1)
    parser.add_argument("--mzpt_batch_size", type=int, default=32,
                        help="Batch size for training muzero planning model.")
    parser.add_argument("--mz_lr", type=float, default=1e-4)
    parser.add_argument("--mz_epochs", type=int, default=1)
    parser.add_argument("--skip_mz_load", action="store_true", help="Skip getting a muzero planning model.")
    parser.add_argument("--skip_mz_train", action="store_true", help="Get a muzero planning model, but don't train it.")
    parser.add_argument("--skip_mz_ckpt_from_wid", action="store_true",
                        help="Skip loading muzero ckpt from folder indicated by ckpts_wid.")
    parser.add_argument("--num_timesteps", type=int, default=5)
    parser.add_argument("--mz_max_timesteps", type=int, default=1)
    parser.add_argument("--mz_weight_decay", type=float, default=1e-4)
    parser.add_argument("--mz_policy_loss_lambda", type=float, default=1.0)
    parser.add_argument("--mz_value_loss_lambda", type=float, default=1.0)
    parser.add_argument("--mz_reward_loss_lambda", type=float, default=1.0)
    parser.add_argument("--mz_reward_loss_fn", type=str, default="mse", choices=["mse", "bce"],
                        help="Loss function to use for reward prediction in muzero.")
    parser.add_argument("--mz_value_loss_fn", type=str, default="mse", choices=["mse", "bce"],
                        help="Loss function to use for value prediction in muzero.")
    parser.add_argument("--mz_no_gradient_scaling", action="store_true")
    parser.add_argument("--mz_reward_baseline", type=str, default=None, choices=["nocc", "random"])
    parser.add_argument("--mz_ckpt_path", type=str, default=None)
    parser.add_argument("--mz_ckpt_wid", type=str, default=None)
    parser.add_argument("--discount", type=float, default=1)  # Don't think discounting makes sense for perplexity
    parser.add_argument("--nonoracle_fraction", type=float, default=0.0,
                        help="Fraction of random non-oracle codes to use for training.")
    # parser.add_argument("--init_action_embedder_with_codebook", action="store_true", help="Initialize action embedder in model_g with codebook.")
    parser.add_argument("--init_action_embedder_with_codebook", type=bool, default=True,
                        help="Initialize action embedder in model_g with codebook.")
    parser.add_argument("--action_embedder_frozen_epochs", type=int, default=0,
                        help="Number of epochs to freeze action embedder in model_g for.")
    # parser.add_argument("--init_policy_head_with_codebook", action="store_true", help="Initialize policy head in model_f with codebook.")
    parser.add_argument("--init_policy_head_with_codebook", type=bool, default=True,
                        help="Initialize policy head in model_f with codebook.")
    parser.add_argument("--only_policy_head", type=bool, default=True,
                        help="Only have model_f in planner, and only its policy head, during mzpt, and only eval greedy policy.")
    parser.add_argument("--eval_only_policy_head", action="store_true",
                        help="During evaluation, only use policy head (no jax required).")
    parser.add_argument("--no_reward_head", action="store_true", help="Don't have reward head in model_g during mzpt.")
    parser.add_argument("--no_sbound", action='store_true')
    # parser.add_argument("--pcodes_wid", default=None,
    #                     help="Wandb id of the planner used to produce planner-predicted codes (for pLMFT) or scores (for JPL). If none, defaults to wid from mzpt_ckpt_wid, and if that is none, to logger.experiment.id")
    parser.add_argument("--pcodes_not_from_ckpt", action="store_true",
                        help="If set, don't use the planner-predicted codes from a given planner ckpt for PLMFT/nonJPL-eval, but instead predict new ones.")
    parser.add_argument("--pscores_not_from_ckpt", action="store_true",
                        help="If set, don't use the planner-predicted scores from a given planner ckpt for JPL-eval, but instead predict new ones.")
    parser.add_argument("--lm_as_planner", action="store_true",
                        help="Use the language model as the planner, instead of a separate model.")
    parser.add_argument("--soft_planner_targets", action="store_true",
                        help="Use kmeans softmax probabilities iso kmeans label as target for planner.")

    # mcts args
    parser.add_argument("--mctx_seed", type=int, default=42)
    parser.add_argument("--num_simulations", type=int, default=2048)
    parser.add_argument("--max_depth", type=int, default=None)

    # misc
    parser.add_argument("--ckpts_wid", type=str, default=None,
                        help="If mz_ckpt_path and lm_ckpt_path are both None, but this is given, we fill them with the latest ckpts from the given wandb run.")
    parser.add_argument("--ckpts_dir", type=str, default=DEFAULT_CKPT_DIR)
    parser.add_argument("--skip_pretraining_validation", type=bool, default=True)
    parser.add_argument("--plantoken", action="store_true",
                        help="Planning tokens baseline: https://arxiv.org/abs/2310.05707")
    parser.add_argument("--skip_train", action="store_true",
                        help="Shorthand for setting skip_lm_train, skip_mz_train and skip_plm_train all to true")
    parser.add_argument("--switch_esval_val", action="store_true",
                        help="Switch the esval and val splits for validation. For debugging purposes")
    parser.add_argument("--validate_even_if_skip_train", action="store_true",
                        help="Validate even if skip_train is set.")
    parser.add_argument("--skip_train_load", action="store_true",
                        help="Skip loading training split chunks from pickle.")
    parser.add_argument("--joint_planner_lm", action="store_true",
                        help="Jointly train planner and language model.")
    parser.add_argument("--last_iso_best_ckpt", action="store_true",
                        help="Use the last checkpoint in stead of the best checkpoint when loading the model.")

    # eval args
    parser.add_argument("--eval_stride", type=int, default=1, help="How many tokens to skip between each eval chunk.")
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--eval_freq", type=int, default=5000, help="How many batches to wait before logging eval.")
    parser.add_argument("--skip_chunk_load", type=bool, help="Skip loading chunks from pickle. Do this by default if only evaluating given checkpoints, because loading chunks takes some time.", default=None)
    parser.add_argument("--skip_noneval_load", type=bool, default=None)
    parser.add_argument("--logger_wid", type=str, default=None,
                        help="If resume=False, this will not be set automatically."
                             "Else, it will equal the first that is not None from: "
                             " - the manually set logger_wid"
                             " - ckpts_wid"
                             " - jplm_ckpt_wid"
                             " - plm_ckpt_wid"
                             " - mz_ckpt_wid"
                             " - lm_ckpt_wid"
                             "Else it will error.")
    parser.add_argument("--skip_eval_resume", action="store_true", help="Skip loading eval state if resume is True.")
    parser.add_argument("--skip_eval", action="store_true", help="Skip eval.")
    parser.add_argument("--log_nll_for_all_codes", action="store_true", help="Log nll for all codes during eval.")
    parser.add_argument("--randae_nll_for_all", type=str, default=None, help="Only relevant if log_nll_for_all_codes is set.", choices=['replace','add'])
    parser.add_argument("--eval_save_greedy_predictions", action="store_true",
                        help="Save greedy predictions to a pickle file.")
    parser.add_argument("--eval_with_path_search", action="store_true",
                        help="Use path search to find the best code for each state during evaluation.")
    parser.add_argument("--eval_mz_search_K", type=int, default=3,
                        help="Number of top codes to consider during path search.")
    parser.add_argument("--eval_mz_search_selection", type=str, default="top_path",
                        choices=["top_path_threshold", "top_path", "most_visited", "most_likely",
                                 "most_likely_average"],
                        help="Method to select action during evaluation with MuZero search.")
    parser.add_argument("--eval_mz_search_score", type=str, default="reward", choices=["policy", "reward"],
                        help="Score to use for selecting action during evaluation with beamsearch.")
    parser.add_argument("--eval_mz_search_threshold", type=float, default=float('-inf'))
    parser.add_argument("--rev_ppl_eval", action="store_true", help="Evaluate with reverse perplexity.")
    parser.add_argument("--generate_ntokens", type=int, default=128,
                        help="Number of tokens to generate for edit distance, reverse ppl, ... . If negative, then as many tokens as remain in the true article with the current context.")
    parser.add_argument("--n_logged_snts", type=int, default=100, help="How many generated sentences to log.")
    parser.add_argument("--judge_model", type=str, default="llama", choices=["llama", "gpt2-xl"])
    parser.add_argument("--sbert_eval", action="store_true", help="Evaluate with SBERT.")
    parser.add_argument("--skip_nll_eval", action="store_true", help="Skip nll eval.")
    parser.add_argument("-l", "--structure_levenshtein_eval", action="store_true",
                        help="Evaluate the generated structure via levenshtein distance.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling from language model.")
    parser.add_argument("--skip_fully_greedy_eval", action="store_true",
                        help="Skip fully greedy evaluation during eval.")
    parser.add_argument("--parallel_ppl_eval", action="store_true",
                        help="Do parallellized fully greedy evaluation during eval.")
    parser.add_argument("--testeval", action="store_true", help="Evaluate on test set iso val set.")
    parser.add_argument("--nll_then_lev", action="store_true", help="Evaluate sequentially: nll for all articles, then levenshtein for all articles.")
    # parser.add_argument("--nll_then_lev", type=bool, default=True, help="Evaluate sequentially: nll for all articles, then levenshtein for all articles.")
    parser.add_argument("-m","--mini", action="store_true", help="Sets max_articles to 60 and nowandb to true (a combo often used during debugging)")
    parser.add_argument("--eval_save_per_token_nll", action="store_true", help="Save per-token nll during eval, intended for comparing improvement vs difficulty.")
    parser.add_argument("--eval_pdistr_vs_kmeansdistr", action="store_true")
    parser.add_argument("--parallel_eval_batch_size", type=int, default=64)
    parser.add_argument("--noctx_geneval", action="store_true")
    parser.add_argument("--noctx_metrics", type=str, nargs='+', default=['mauve','latent_ppl'], help="List of metrics to evaluate with noctx_geneval.")
    parser.add_argument("--latent_ppl_critic_path", type=str, default=None)
    parser.add_argument("--mauve_max_text_length", type=int, default=256)
    parser.add_argument("--subsample_geneval", type=float, default=1.0, help="Subsample the generated articles for evaluation.")
    parser.add_argument("--noctx_gen_wid", type=str, default=None, help="Wandb id of the run that generated no-context articles.")
    parser.add_argument("--sfctx_gen_wid", type=str, default=None, help="Wandb id of the run that generated articles that start with full context.")
    parser.add_argument("--gen_wid", type=str, default=None, help="If noctx and sfctx gen wids are the same, use this one.")
    parser.add_argument("--noctx_gen_json_path", type=str, default=None, help="Path to json file with generated articles for noctx eval.")
    parser.add_argument("--noctx_gen_num_samples", type=int, default=5000, help="Number of samples to generate for noctx eval.")
    parser.add_argument("--noctx_gen_num_tokens", type=int, default=1024, help="Number of samples to generate for noctx eval.")
    parser.add_argument("--noctx_gen_batch_size", type=int, default=128, help="Batch size for generating samples for noctx eval.")
    parser.add_argument("--startfullctx_geneval", action="store_true", help="Generate once per evaluation article, with context that is the earliest at the start when it is full.")
    parser.add_argument("--startfullctx_gen_num_tokens", type=int, default=2048, help="Number of samples to generate for noctx eval.")
    parser.add_argument("--startfullctx_metrics", type=str, nargs='+', default=['rouge-2','levenshtein'], help="List of metrics to evaluate with startfullctx_geneval.")
    parser.add_argument('--startfullctx_token_limits', type=int, nargs='+', default=[128,256,512,1024,2048], help='List of token limits.')
    parser.add_argument("--startfullctx_gen_json_path", type=str, default=None, help="Path to json file with generated articles for startfullctx eval.")
    parser.add_argument("--only_match_eval", type=bool, default=True, help="Only evaluate with codes that match the training setting.")

    # lightning args
    parser.add_argument("--val_check_interval", default=None)
    parser.add_argument("--check_val_every_n_epoch", type=int, default=1)
    parser.add_argument("--checkpoint_dir", type=str, default=DEFAULT_CKPT_DIR)
    parser.add_argument("--fast_dev_run", action="store_true",
                        help="If set, only to one batch of each train, val, test for testing your code.")
    parser.add_argument("--detect_anomaly", action="store_true", help="Detect anomaly in the model.")
    parser.add_argument("--save_top_k", type=int, default=1, help="Save top k models.")
    parser.add_argument("--log_every_n_steps", type=int, default=None)

    # transformers args
    parser.add_argument("--use_4bit", action="store_true")

    # wandb args
    parser.add_argument("--nowandb", action="store_true")
    parser.add_argument("--notes", type=str, default="")
    parser.add_argument("--prefix", type=str, default="")
    parser.add_argument("--postfix", type=str, default="")
    parser.add_argument("-n", "--name", type=str, default="")
    parser.add_argument("--tags", type=str, nargs='+', default=[])
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()
    args.sentence_transformer_cache = SF_CACHE_DIR
    if args.mini:
        args.max_articles = 60
        args.nowandb = True
        args.noctx_gen_num_samples = 120
        args.noctx_gen_num_tokens = 130
        args.startfullctx_gen_num_tokens = 130
    if args.nowandb:  # For when debugging
        os.environ['WANDB_MODE'] = 'disabled'
    if args.base_model_name == "llama":
        args.base_model_name = "meta-llama/Llama-2-7b-hf"
    if args.base_model_name == "olmo":
        args.base_model_name = "allenai/OLMo-1B"
    if args.judge_model == "llama":
        args.judge_model = "meta-llama/Llama-2-7b-hf"

    if args.ckpts_wid is not None:
        assert args.lm_ckpt_wid is None and args.mz_ckpt_wid is None, "ckpts_wid and lm_ckpt_wid/mz_ckpt_wid/plm_ckpt_wid are mutually exclusive"
        args.lm_ckpt_wid = args.mz_ckpt_wid = args.plm_ckpt_wid = args.jplm_ckpt_wid = args.ckpts_wid


    if args.cc_type == "none":
        print("cc_type is None, so setting skip_mz_load to True, skip_lm_load to False, tune_lm_with_planner to False, joint_planner_lm to False")
        args.skip_mz_load = True
        args.skip_lm_load = False
        args.tune_lm_with_planner = False
        args.joint_planner_lm = False

    if args.fixed_code:
        print("setting lm_load to true because fixed_code is True")
        args.skip_lm_load = False
    if args.fixed_code or args.uniform_mix:
        print("setting tune_lm_with_planner to false because fixed_code or uniform_mix is True")
        args.tune_lm_with_planner = False
        print("Setting mzpt_load to false because fixed_code or uniform_mix  is True")
        args.skip_mz_load = True
    if args.uniform_mix:
        print("Setting jpl to true because uniform_mix is True")
        args.joint_planner_lm = True

    if args.plantoken:
        print("Setting cc_type to insert because plantoken is True")
        args.cc_type = "insert"
        assert args.og_params_finetune_type != "freeze", "og_params_finetune_type should not be freeze for plantoken"
        print("Setting no-projection insert because plantoken is True")
        args.insert_no_proj = True
        print("Setting lm_as_planner to True because plantoken is True")
        args.lm_as_planner = True

    if args.base_model_name == "allenai/OLMo-1B":
        print("Setting insert_no_proj to False because base_model_name is OLMo, and dimensions don't match without projection")
        args.insert_no_proj = False

    if args.lm_as_planner:
        print("Skipping loading of external planner because lm_as_planner is true")
        args.skip_mz_load = True
        print("Setting skip lm to false because lm_as_planner is true")
        args.skip_lm_load = False
        print("Setting pcclm to false because lm_as_planner is true")
        args.tune_lm_with_planner = False

    # for prefix in (['lm'] if not args.skip_lm_load else []) + (['mz'] if not args.skip_mz_load else []) + (
    # ['plm'] if args.tune_lm_with_planner else []):
    prefixes = []
    if not args.skip_lm_load:
        prefixes.append('lm')
    if not args.skip_mz_load:
        prefixes.append('mz')
    if args.tune_lm_with_planner and not args.skip_plm_load:
        prefixes.append('plm')
    if args.joint_planner_lm:
        prefixes.append('jplm')

    for prefix in prefixes:
        path_attr = f"{prefix}_ckpt_path"
        skip_attr = f"skip_{prefix}_ckpt_from_wid"
        if getattr(args, path_attr) is None and not getattr(args, skip_attr):
            wid_attr = f"{prefix}_ckpt_wid"
            if getattr(args, wid_attr) is not None:
                def g(dir, infix):
                    return glob(jn(dir, getattr(args, wid_attr), f"{prefix}-{infix}epoch=*-step=*.ckpt"))
                # if args.lm_epochs > 1 or args.mz_epochs > 1:
                    # raise ValueError("WARNING: multiple epochs set, but resuming from last checkpoint. If multiple epochs, overfitting means it might be better to use BEST epoch."
                    #                  "Not doing at the moment cuz assuming we'll work in one-epoch setting")
                LAST_OR_BEST = "LAST" if args.last_iso_best_ckpt else "BEST"
                ckpt_paths = g(args.ckpts_dir, f"{LAST_OR_BEST}-")
                if len(ckpt_paths) == 0:
                    ckpt_paths = g(OTHER_CKPT_DIR, f"{LAST_OR_BEST}-")

                # backwards compatibility
                if len(ckpt_paths) == 0:
                    ckpt_paths = g(args.ckpts_dir, "")
                if len(ckpt_paths) == 0:
                    ckpt_paths = g(OTHER_CKPT_DIR, "")

                if len(ckpt_paths) == 0:
                    raise ValueError(f"No {prefix} checkpoints found with given ckpt_wid {getattr(args, wid_attr)}")
                setattr(args, path_attr, max(ckpt_paths, key=os.path.getctime))

    if args.mz_h_sent_model is None:
        args.mz_h_sent_model = args.embedder_name
    for key in ['mz_h_sent_model', 'embedder_name']:
        if args.__dict__[key] in SHORT2FULL_EMBEDDER_NAME:
            args.__dict__[key] = SHORT2FULL_EMBEDDER_NAME[args.__dict__[key]]
        if args.__dict__[key] not in SHORT2FULL_EMBEDDER_NAME.values():
            raise ValueError(f"Invalid {key} {args.__dict__[key]}")

    bigdata = (args.max_articles == -1) or (args.max_articles > 1000000)
    args.stream = bigdata if args.stream is None else args.stream

    if args.val_check_interval is None:
        if bigdata:
            args.val_check_interval = 100000
        elif any([e == 1 for e in [args.lm_epochs, args.mz_epochs]]) and (
                args.max_articles > 1000 or args.max_articles < 0):
            args.val_check_interval = 0.1
        else:
            args.val_check_interval = 1.0
    if 0 < args.max_articles < 150:
        args.eval_freq = 500

    if args.adapter_extra_context:
        print("Setting cc_type to adapter")
        args.cc_type = "adapter"

    if args.cc_type == 'adapter':
        print("Setting insert_no_proj to False because cc_type is adapter")
        args.insert_no_proj = False

    if args.insert_no_proj:
        assert args.cc_type in ["insert", "none"]
        if not args.skip_train:
            assert args.og_params_finetune_type != "freeze", "og_params_finetune_type should not be freeze for insert_no_proj"
        print("Unfreezing codebook after step 0 because insert_no_proj is True")
        args.unfreeze_codebook_after_step = 0

    if 0 < args.max_articles < 1000:
        args.skip_pretraining_validation = True
        print("Setting skip_pretraining_validation to True because max_articles < 1000")

    if args.skip_train:
        args.skip_lm_train = args.skip_mz_train = args.skip_plm_train = args.skip_jplm_train = True


    def check_train(args):
        return ((
                        args.skip_lm_load or
                        args.skip_lm_train)
                and
                args.skip_mz_train
                and (
                        not args.tune_lm_with_planner or
                        args.skip_plm_train)
                and (
                        not args.joint_planner_lm or
                        args.skip_jplm_train
                )
                )

    if args.skip_noneval_load is None:
        args.skip_noneval_load = False
        if check_train(args):
            print("Setting skip_noneval_load to True because skip_lm_train and skip_mz_train are both True")
            args.skip_noneval_load = True

    # if skip_chunk_load not set explicitly:
    if args.skip_chunk_load is None:
        # default is False
        args.skip_chunk_load = False
        # if skip_lm_train and skip_lm_load, then set skip_chunk_load to True
        if check_train(args):
            print("Setting skip_chunk_load to True because skip_lm_train and skip_mz_train are both True")
            args.skip_chunk_load = True
        elif args.stream:
            print("Setting skip_chunk_load to True because stream is True")
            args.skip_chunk_load = True
    assert args.skip_chunk_load or not args.stream, "streaming requires not loading chunks"

    on_vsc = getpass.getuser().startswith('vsc')
    if on_vsc and args.logger_wid is None:
        # set equal to slurm job id
        args.logger_wid = os.environ.get('SLURM_JOB_ID')

    # A bit lazy, but with Olmo, validating pre-training gives an error like this: https://stackoverflow.com/questions/75517324/runtimeerror-inference-tensors-cannot-be-saved-for-backward-to-work-around-you
    # simply setting skip_pretraining_validation to True seems to avoid this, so lets do it by default if olmo
    if args.base_model_name == "allenai/OLMo-1B":
        args.skip_pretraining_validation = True
        print("Setting skip_pretraining_validation to True because base_model_name is OLMo")
        args.noctx_gen_batch_size = 64

    if args.jpl_freeze_planner:
        print("Setting jpl_unfreeze_planner_after_step_or_frac to negative because jpl_freeze_planner is True")
        args.jpl_unfreeze_planner_after_step_or_frac = "-1"

    if args.cc_param_seed is None:
        args.cc_param_seed = args.seed

    if args.gen_wid is not None:
        if args.noctx_gen_wid is None:
            args.noctx_gen_wid = args.gen_wid
        if args.sfctx_gen_wid is None:
            args.sfctx_gen_wid = args.gen_wid
    return args


def scores_iso_codes(args):
    return args.joint_planner_lm and not args.straight_through
