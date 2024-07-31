import gzip
import random
from dataclasses import dataclass
import datasets
from time import time
import json
import os
import pickle
from os.path import join as jn
from typing import Dict

import numpy as np
import torch
# print("Importing sentence_transformers ..."); s = time();
from sentence_transformers import SentenceTransformer;
# print(f"Imported sentence_transformers in {time() - s} seconds")
from sklearn.cluster import KMeans
from torch.utils.data import random_split, DataLoader
from functools import partial
from tqdm import tqdm as std_tqdm

from tokenizer_util import myAutoTokenizer
tqdm = partial(std_tqdm, dynamic_ncols=True)
import warnings
from util import Namespace, pdump, pload
from multiprocessing import Pool
from constants import EN_WIKI_ARTICLE_COUNT, EN_WIKI_FILE, DEFAULT_PICKLE_DIR, DEFAULT_LLM, FULL2SHORT_EMBEDDER_NAME, \
    NEW_EN_WIKI_ARTICLE_COUNT, DOLMA_ARTICLE_COUNT
from glob import glob
from lightning import pytorch as pl
import torch.nn.functional as F
import re
from args import scores_iso_codes

warnings.filterwarnings("ignore", category=UserWarning, message=".*pydantic*.")

def get_chunk_keys(cc_type=None):
    if cc_type == 'insert':
        CHUNK_KEYS = sorted(['input_ids_and_codes', 'attention_mask', 'label_mask', 'snt_idxs', 'article_idx', 'code_mask'])
    else:
        CHUNK_KEYS = sorted(['input_ids', 'attention_mask', 'label_mask', 'codes', 'snt_idxs', 'snt_bounds', 'article_idx'])
    return CHUNK_KEYS

class MyDataset(torch.utils.data.Dataset):
    splits = ['train', 'esval', 'val', 'test']
    nocc_splits = ['esval-nocc', 'val-nocc'] # These will contain chunks without codes. They are needed if cc_type == insert, because we cannot then use the same chunks as the normal esval/val

    prefix: str
    total_article_count: int
    TEST_SIZE = 5000
    VAL_SIZE = 1000
    ESVAL_SIZE = 1000
    TEST_SEED = 123

    def __init__(self, tokenizer=None, max_articles=-1, max_seq_len=4096, seed=42, root_pickle_dir=DEFAULT_PICKLE_DIR, coded_pkl_path=None, base_model_name=None, fixed_code=False, skip_chunk_load=False, skip_noneval_load=None, embedder_name=None, no_cluster=False,cc_type='adapter', skip_train_load=False, soft_planner_targets=False, **kwargs):
        if tokenizer is None:
            model_name = base_model_name if base_model_name is not None else DEFAULT_LLM
            tokenizer = myAutoTokenizer(model_name)
            tokenizer.bos_token_id = tokenizer.eos_token_id
        self.nlp = None
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.phase = 'lm'
        self.use_planner_codes = False
        self.seed = seed
        self.num_train_articles = self.get_num_train_articles(max_articles)
        self.maybe_set_article_split_counts()
        self.fixed_code = fixed_code
        self.root_pickle_dir = root_pickle_dir
        self.embedder_name = embedder_name
        self.no_cluster = no_cluster
        self.cc_type = cc_type
        self.no_sbound = kwargs['no_sbound']
        self.kmeans = None
        self.skip_train_load = skip_train_load
        self.soft_planner_targets = soft_planner_targets


        self.split2idxs_seed = self.load_split2idxs_seed()
        self.load_coded_articles(kwargs, skip_noneval_load=skip_noneval_load, skip_train_load=skip_train_load)
        self.kwargs = kwargs
        if not skip_chunk_load:
            self.load_chunks(kwargs)

        if self.soft_planner_targets:
            self.load_kmeans_model(kwargs)

    def flat2nested_idx(self, flat_idx, split):
        return {f: (a, s)
                for f, (a, s) in enumerate(
                (aa,ss) for (aa, article) in enumerate(
                    self.split2coded_articles[split]) for ss in range(len(article['sentences'])))
                }[flat_idx]

    def get_kmeans_probs(self, snt_idxs, split):
        snt_idxs = np.array(list(snt_idxs))
        embs = self.split2sent_embs[split][snt_idxs]
        kmeans_dists = self.kmeans.transform(embs)
        # Following https://datascience.stackexchange.com/a/14441
        kmeans_unnormalized_probs = np.exp(-kmeans_dists)
        kmeans_probs = kmeans_unnormalized_probs / kmeans_unnormalized_probs.sum(-1)[:,None]
        return kmeans_probs

    def nested2flat_idx(self, article_idx, sentence_idx, split):
        return sum(len(article['sentences']) for article in self.split2coded_articles[split][:article_idx]) + sentence_idx

    def load_split2idxs_seed(self):
        test_idxs, remaining_idxs = self.get_test_and_remaining_idxs()

        val_filepath = jn(self.pickle_dir('val', self.VAL_SIZE, self.seed), "idxs.pkl")
        os.makedirs(os.path.dirname(val_filepath), exist_ok=True)
        esval_filepath = jn(self.pickle_dir('esval', self.ESVAL_SIZE, self.seed), "idxs_and_complement.pkl")
        os.makedirs(os.path.dirname(esval_filepath), exist_ok=True)
        if any(not os.path.exists(path) for path in [val_filepath, esval_filepath]):
            random.seed(self.seed);shuffled_remaining_idxs = random.sample(remaining_idxs, len(remaining_idxs))
            val_idxs = shuffled_remaining_idxs[:self.VAL_SIZE]
            esval_idxs = shuffled_remaining_idxs[self.VAL_SIZE:self.VAL_SIZE + self.ESVAL_SIZE]
            remaining_idxs = shuffled_remaining_idxs[self.VAL_SIZE + self.ESVAL_SIZE:]
            with open(val_filepath, 'wb') as f:
                pickle.dump(val_idxs, f)
            with open(esval_filepath, 'wb') as f:
                pickle.dump((esval_idxs,remaining_idxs), f)
        with open(val_filepath, 'rb') as f:
            val_idxs = pickle.load(f)
        with open(esval_filepath, 'rb') as f:
            esval_idxs, remaining_idxs = pickle.load(f)

        train_filepath = jn(self.pickle_dir('train', self.num_train_articles, self.seed), "idxs.pkl")
        os.makedirs(os.path.dirname(train_filepath), exist_ok=True)
        if not os.path.exists(train_filepath):
            train_idxs = remaining_idxs[:self.num_train_articles]
            with open(train_filepath, 'wb') as f:
                pickle.dump(train_idxs, f)
        with open(train_filepath, 'rb') as f:
            train_idxs = pickle.load(f)
            assert len(train_idxs) == self.num_train_articles

        # return {'train': train_idxs, 'esval': esval_idxs, 'val': val_idxs, 'test': test_idxs}
        return {'train': {'idxs': train_idxs, 'seed': self.seed},
                'esval': {'idxs': esval_idxs, 'seed': self.seed},
                'val': {'idxs': val_idxs, 'seed': self.seed},
                'test': {'idxs': test_idxs, 'seed': self.TEST_SEED}}


    def get_test_and_remaining_idxs(self):# One seed for the paper for test split
        test_filepath = jn(self.pickle_dir('test', self.TEST_SIZE, self.TEST_SEED), "idxs_and_complement.pkl")
        os.makedirs(os.path.dirname(test_filepath), exist_ok=True)
        if not os.path.exists(test_filepath):
            random.seed(self.TEST_SEED)
            shuffled_all_idxs = random.sample(list(range(self.total_article_count)), self.total_article_count)
            test_idxs = shuffled_all_idxs[:self.TEST_SIZE]
            remaining_idxs = shuffled_all_idxs[self.TEST_SIZE:]
            with open(test_filepath, 'wb') as f:
                pickle.dump((test_idxs, remaining_idxs), f)
        with open(test_filepath, 'rb') as f:
            test_idxs, remaining_idxs = pickle.load(f)
        return test_idxs, remaining_idxs

    @classmethod
    def get_num_train_articles(cls, max_articles):
        available_train_articles = cls.total_article_count - (cls.TEST_SIZE + cls.VAL_SIZE + cls.ESVAL_SIZE)
        num_articles = available_train_articles if max_articles < 0 else min(max_articles, available_train_articles)
        return num_articles


    def maybe_set_article_split_counts(self):
        pass


    def pickle_dir(self,split, num_articles=None, seed=None):
        if num_articles is None:
            num_articles = len(self.split2idxs_seed[split]['idxs'])
        if seed is None:
            seed = self.split2idxs_seed[split]['seed']
        result = jn(self.root_pickle_dir, f"{self.format_name(split, num_articles, seed)}")
        return result



    def get_sent_embedder(self):
        if self.embedder_name in FULL2SHORT_EMBEDDER_NAME:
            return SentenceTransformer(self.embedder_name)
        else:
            raise NotImplementedError(f"embedder_name {self.embedder_name} not implemented")

    def get_sequence_of_actions(self, text_str, embedder):
        self.load_spacy()
        self.load_kmeans_model(self.kwargs)
        text_sentences = self.get_sentencized_text(text_str)
        text_embeds = embedder.encode(text_sentences, convert_to_numpy=True)
        sequence_of_actions = self.kmeans.predict(text_embeds)
        return sequence_of_actions


    def load_split_embeddings(self, split, sentencized_articles=None, sbert_batch_size=None, **_):
        en = FULL2SHORT_EMBEDDER_NAME[self.embedder_name]
        embs_path = jn(self.pickle_dir(split), f'{en}_sbert_embeddings.pkl')
        if os.path.exists(embs_path):
            print(f"Loading embeddings from {embs_path}")
            with open(embs_path, 'rb') as f:
                embs = pickle.load(f)
        else:
            assert (sentencized_articles is not None) and (sbert_batch_size is not None), "sentencized_articles and sbert_batch_size must be provided if embeddings are not already saved"
            embs = self.get_sbert_embeddings(sbert_batch_size, sentencized_articles)
            with open(embs_path, 'wb') as f:
                pickle.dump(embs, f)
        return embs

    def get_sbert_embeddings(self, sbert_batch_size, sentencized_articles):
        start = time()
        print(f'Generating embeddings for {len(sentencized_articles)} articles')
        sbert = self.get_sent_embedder()
        pool = sbert.start_multi_process_pool()
        embs = my_encode_multi_process(sentencized_articles, pool=pool, batch_size=sbert_batch_size)
        print(f"Finished generating embeddings in {time() - start} seconds")
        sbert.stop_multi_process_pool(pool)
        return embs

    def load_kmeans_model(self, kwargs):
        args = Namespace(**kwargs)
        en = FULL2SHORT_EMBEDDER_NAME[self.embedder_name]
        if self.kmeans is None:
            kmeans_path = jn(self.pickle_dir("train"), f"kmeans_{en}c{args.cluster_count}.pkl") if args.kmeans_path is None else args.kmeans_path
            if os.path.exists(kmeans_path):
                print(f"Loading kmeans from {kmeans_path}")
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="Trying to unpickle estimator KMeans from version")
                    with open(kmeans_path, 'rb') as f:
                        kmeans = pickle.load(f)
                self.kmeans = kmeans
            else:
                raise ValueError(f"Kmeans path {kmeans_path} does not exist")

    def get_kmeans(self, sentencized_articles, kwargs):
        sse = []
        kmeans_list = []
        args = Namespace(**kwargs)
        en = FULL2SHORT_EMBEDDER_NAME[self.embedder_name]
        kmeans_path = jn(self.pickle_dir("train"), f"kmeans_{en}c{args.cluster_count}.pkl") if args.kmeans_path is None else args.kmeans_path
        if os.path.exists(kmeans_path):
            print(f"Loading kmeans from {kmeans_path}")
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Trying to unpickle estimator KMeans from version")
                with open(kmeans_path, 'rb') as f:
                    kmeans = pickle.load(f)
        else:
            assert sentencized_articles is not None
            train_embeddings = self.load_split_embeddings('train', sentencized_articles,
                                                          **kwargs)  # always use train split as basis for kmeans
            print(f"Fitting kmeans with {args.cluster_count} clusters for {len(train_embeddings)} embeddings");
            s = time()
            kmeans = KMeans(n_clusters=args.cluster_count, init='k-means++', n_init='auto', verbose=1, random_state=self.seed).fit(train_embeddings)
            print(f"Done fitting kmeans in {time() - s} seconds")
            os.makedirs(os.path.dirname(kmeans_path), exist_ok=True)
            with open(kmeans_path, 'wb') as f:
                print(f"Saving kmeans to {kmeans_path}")
                pickle.dump(kmeans, f)
        sse.append(kmeans.inertia_)
        kmeans_list.append(kmeans)
        if args.kmeans_cluster_debug:
            plot_kmeans(args, sse)
        kmeans = kmeans_list[np.argmin(sse)]
        if args.kmeans_cluster_debug:
            qual_cluster_sentences_check(kmeans, self.articles)
        return kmeans

    @staticmethod
    def get_chunks_path(code_pickle_dir, tokenizer_name, max_seq_len, cc_type='adapter'):#,phase='lm',num_timesteps=None):
        # postfix = "" if num_timesteps is None else f"_t{num_timesteps}"
        # return jn(code_pickle_dir, f"{tokenizer_name}-tokenized_chunks_l{max_seq_len}_{phase}{postfix}.pkl")
        prefix = 'CODE_INSERTED_' if (cc_type == 'insert') else ''
        return jn(code_pickle_dir, f"{prefix}{tokenizer_name}-tokenized_chunks_l{max_seq_len}.pkl")

    def set_phase(self, phase):
        self.phase = phase

    # This is kinda ugly, but couldn't immediately find a nice way to adapt the existing tokenizer's padding/truncating/chunking capacities to include the codes
    def load_chunks(self, kwargs):
        args = Namespace(**kwargs)
        self.split2chunks = {}
        for split in self.splits + (self.nocc_splits if self.cc_type == 'insert' else []):
            if self.skip_train_load and split == 'train':
                continue
            chunks_for_split = self.get_chunks_for_split(args, split)
            self.split2chunks[split] = chunks_for_split

    def get_chunks_for_split(self, args, split):
        core_split_name = split.split('-')[0]
        cc_type = None if 'nocc' in split else self.cc_type
        CHUNK_KEYS = get_chunk_keys(cc_type)
        dir = self.code_pickle_dir(core_split_name, args.cluster_count, args.kmeans_path, len(self.split2coded_articles[
                                                                                                  core_split_name]))  # if mini-run, num_articles can differ from what is inferred automatically based on self.split2idxs_seed
        path = WikiDataset.get_chunks_path(dir, type(self.tokenizer).__name__, self.max_seq_len, cc_type)
        as_tensor_path = path.replace('.pkl', '_as_tensor.pt')
        if os.path.exists(as_tensor_path) and not args.force_remake_chunks:
            print(f"Loading all_chunks as tensor from {as_tensor_path}");
            s = time()
            with open(as_tensor_path, 'rb') as f:
                all_chunks_as_tensor = torch.load(f)
            print(f"Loaded in {time() - s} seconds")
            assert len(all_chunks_as_tensor) % len(CHUNK_KEYS) == 0
            # all_chunks = [{k: maybe_unexpand(k, all_chunks_as_tensor[i+j]) for j,k in enumerate(keys)} for i in tqdm(range(0,len(all_chunks_as_tensor),len(keys)), desc="Unexpanding all_chunks_as_tensor")]
        else:
            if os.path.exists(path) and not args.force_remake_chunks:
                # Backwards compatibility
                print(f"{as_tensor_path} does not exist but {path} does, so loading chunks from {path}");
                s = time()
                print(f"Loading chunks from {path}");
                s = time()
                with open(path, 'rb') as f:
                    all_chunks = pickle.load(f)
                print(f"Done loading chunks from {path} in {time() - s} seconds")
            else:
                print(
                    f"{as_tensor_path} does not exist or force_remake_chunks is True, so making chunks and saving to {as_tensor_path}")
                # with Pool() as p:
                #     all_chunks = p.map(partial(WikiDataset.make_article_chunks,tokenizer=self.tokenizer,max_seq_len=self.max_seq_len), tqdm(self.articles))
                # Parallellizing like above gives problem with too many open files that I couldn't figure out immediately, so just doing it with one process for now
                all_chunks = [chunk for i, article in
                              tqdm(enumerate(self.split2coded_articles[core_split_name]), desc="Making chunks") for
                              chunk in
                              self.make_article_chunks(article, self.max_seq_len, self.tokenizer, article_idx=i,
                                                       cc_type=cc_type)]  # , phase=phase, num_timesteps=num_timesteps)]
            IDS_KEY = 'input_ids' if cc_type != 'insert' else 'input_ids_and_codes'
            all_chunks_as_tensor = torch.stack(
                [maybe_expand(k, v, len(chunk[IDS_KEY]), cc_type) for chunk in tqdm(all_chunks, desc="dict->tensor") for
                 k, v in {k: chunk[k] for k in sorted(chunk)}.items()])
            os.makedirs(os.path.dirname(as_tensor_path), exist_ok=True)
            with open(as_tensor_path, 'wb') as f:
                print(f"Saving all_chunks as tensor to {as_tensor_path}")
                torch.save(all_chunks_as_tensor, f)
        all_chunks = all_chunks_as_tensor.view(-1, len(CHUNK_KEYS), len(all_chunks_as_tensor[0]))
        return all_chunks

    def get_splits(self):
        # subsets_dict = {split: Subset(self, idxs) for split, idxs in self.split2chunk_idxs.items()}
        subsets_dict = {split: WikisplitDataset(self, split) for split in self.splits}
        if self.cc_type == 'insert' and self.phase == 'lm':
            subsets_dict |= {split: WikisplitDataset(self, split, nocc=True) for split in self.nocc_splits}
        return subsets_dict

    def set_splits(self):
        # This class keeps track of the data at two levels: article level and within-article-chunk level
        # all_idxs = list(range(self.num_articles))
        # random.Random(self.seed).shuffle(all_idxs)
        # ratios = [.8, .1, .1]
        # split1 = int(self.num_articles * ratios[0])
        # split2 = split1 + int(self.num_articles * ratios[1])
        # self.split2article_idxs = {'train': all_idxs[:split1], 'esval': all_idxs[split1:split2],
        #                            'val': all_idxs[split2:]} # 'esval' stands for early-stopping val

        nontrain_idxs = self.val_idxs + self.esval_idxs + self.test_idxs
        train_idxs = []
        i = 0
        while (len(train_idxs) < self.num_train_articles) and i < self.total_article_count:
            if i not in nontrain_idxs:
                train_idxs.append(i)
            i += 1
        random.Random(self.seed).shuffle(train_idxs)

        # for mini-runs (eg with only 50 articles, for debugging), we don't want to use the full 1k esval resp. val articles
        val_idxs = self.val_idxs[:self.num_train_articles]
        esval_idxs = self.esval_idxs[:self.num_train_articles]
        self.split2article_idxs = {'train': train_idxs, 'esval': esval_idxs, 'val': val_idxs}



    @staticmethod
    def make_article_chunks(article, max_seq_len, tokenizer=None, stride=None, article_idx=None,get_eval_code=False, pairs_at_snt_bounds=False,cc_type='adapter', eval=False, code_type='codes', bos_id=None):
        '''
        NOTE: this description is somewhat out of date/incomplete, but maybe still partly useful :P
        Chunks do not cross article boundaries. Each chunk is a sequence of max_seq_len tokens, not necessarily ending at a sentence boundary.
        Stride is the number of tokens to move forward after each chunk. If stride is None, stride=max_seq_len, so chunks are non-overlapping.
        Example: if max_seq_len=chunk_size is 5 (Note that the example uses words but the code will use token ids)
        INPUT:
        Article sentences:
        " He went there. He loved it. The leaves were falling again. Autumn nights. The next day was chilly."
        Codes:
        42               95           8                                20             33
        snt_idxs:
        0                1           2                                 3              4
        OUTPUT for stride = 4 (maximum possible stride)
        Chunks:
        [
            {'input_ids': [ [BOS] He  went  there. He ],     'codes': [42 95],   'snt_idxs': [0 1],   'snt_bounds': [4 7],   'attention_mask': [1 1 1 1 1]},
            {'input_ids': [ He loved it. The leaves] ,       'codes': [95 8],    'snt_idxs': [1 2],   'snt_bounds': [3 8],   'attention_mask': [1 1 1 1 1]},
            {'input_ids': [ leaves were falling again. Autumn ], 'codes': [8 20],    'snt_idxs': [2 3],   'snt_bounds': [4 6],   'attention_mask': [1 1 1 1 1]},
            {'input_ids': [Autumn nights. The next day ],    'codes': [20 33],   'snt_idxs': [3 4],   'snt_bounds': [2 7],   'attention_mask': [1 1 1 1 1]},
            {'input_ids': [day was chilly. [PAD] [PAD]],     'codes': [33],      'snt_idxs': [4],     'snt_bounds': [4],     'attention_mask': [1 1 1 0 0]}
            ]
         ]
         OUTPUT for stride = 2
         Chunks:
            [
                 {'input_ids': [ [BOS] He  went  there. He ],         'codes': [42 95],   'snt_idxs': [0 1],   'snt_bounds': [4 7],   'attention_mask': [1 1 1 1 1]},
                 {'input_ids': [went  there. He  loved it.],          'codes': [42 95 8], 'snt_idxs': [0 1 2], 'snt_bounds': [2 5 10],'attention_mask': [1 1 1 1 1]},
                    {'input_ids': [He loved it. The leaves] ,         'codes': [95 8],    'snt_idxs': [1 2],   'snt_bounds': [3 8],   'attention_mask': [1 1 1 1 1]},
                    {'input_ids': [it. The leaves were falling],      'codes': [95 8],    'snt_idxs': [1 2],   'snt_bounds': [1 6],   'attention_mask': [1 1 1 1 1]},
                    {'input_ids': [leaves were falling again. Autumn],'codes': [8 20],    'snt_idxs': [2 3],   'snt_bounds': [4 6],   'attention_mask': [1 1 1 1 1]},
                    {'input_ids': [falling again. Autumn nights. The],'codes': [8 20 33], 'snt_idxs': [2 3 4], 'snt_bounds': [2 4 9], 'attention_mask': [1 1 1 1 1]},
                    {'input_ids': [Autumn nights. The next day],      'codes': [20 33],   'snt_idxs': [3 4],   'snt_bounds': [2 7],   'attention_mask': [1 1 1 1 1]},
                    {'input_ids': [The next day was chilly.],        'codes': [33],      'snt_idxs': [4],     'snt_bounds': [5],     'attention_mask': [1 1 1 1 1]},
                ]

        '''
        if bos_id is None:
            assert tokenizer is not None
            bos_id = tokenizer.bos_token_id
        codes_between_tokens = cc_type == 'insert'
        if get_eval_code:
            assert stride == 1, "get_eval_code only works for stride 1"
        if eval:
            assert stride == 1, "eval only works for stride 1"
        if not pairs_at_snt_bounds:
            if stride is None:
                stride = max_seq_len - 1
            else:
                assert 1 <= stride < max_seq_len # because we take input and labels from the same chunk, if stride == max_seq_len, one in (max_seq_len) tokens will never be used as a label

        unpacked_art = WikiDataset.unpack_article(article, bos_id, cc_type, code_type=code_type)


        result = []
        if not pairs_at_snt_bounds:
            art_len = len(unpacked_art.art_ids) if not codes_between_tokens else len(unpacked_art.art_ids_and_codes)
            first_chunk_end = stride + 1  # +1 to start beyond BOS token if stride = 1
            maybe_overshoot_end = first_chunk_end
            while maybe_overshoot_end < art_len + stride:
                end = min(maybe_overshoot_end, art_len)
                if eval and codes_between_tokens:
                    # num_codes_among_new_tokens = sum(unpacked_art.art_code_mask[
                    #                                  end - stride:end])  # we want to have [stride] new tokens that we actually use the NLL from, during eval only the last [stride] tokens are evaluated
                    # maybe_overshoot_end += num_codes_among_new_tokens
                    # if num_codes_among_new_tokens > 0:
                    #     continue
                    assert stride == 1, "Not necessarily correct of strides other than 1 in eval"
                    if unpacked_art.art_code_mask[end-1]: # Means that the last token is a code token
                        maybe_overshoot_end += 1
                        continue
                maybe_undershoot_start = maybe_overshoot_end - max_seq_len
                target_start = max(0, maybe_undershoot_start)
                padded_target_start = target_start
                padded_target_end = target_start + max_seq_len
                chunk = WikiDataset.get_chunk(unpacked_art, target_start, end, padded_target_start, padded_target_end,
                                              max_seq_len, article_idx, get_eval_code, bos_id, cc_type, code_type=code_type)
                result.append(chunk)
                maybe_overshoot_end += stride

        else:
            num_sents = len(article['sentences'])
            if cc_type != 'insert':
                assert num_sents == len(unpacked_art.art_snt_bounds)
            # for bound in [1] + unpacked_art.art_snt_bounds[:-1]: # 1 is the first sentence boundary, which is not in art_snt_bounds
            for chunk_idx in range(num_sents):
                chunk = WikiDataset.get_chunks_at_art_and_snt_idxs(article_idx, chunk_idx, get_eval_code, max_seq_len,
                                                                   unpacked_art,bos_id, cc_type, code_type=code_type)

                result.append(chunk)

        return result

    @staticmethod
    def get_chunks_at_art_and_snt_idxs(article_idx, within_art_snt_idx, get_eval_code, max_seq_len, unpacked_art, bos_id, cc_type='adapter', code_type='codes'):
        art_token_ids_and_maybe_code_ids = unpacked_art.art_ids if cc_type != 'insert' else unpacked_art.art_ids_and_codes
        if cc_type != 'insert':
            bound = ([1] + unpacked_art.art_snt_bounds[:-1])[within_art_snt_idx]
            # art_only_tk_ids = art_token_ids_and_maybe_code_ids
        else:
            # use unpacked_art.art_snt_ids, which looks like [-1, 0, -1, -1, ... -1, 1, -1 etc]. AKA at every location where the token is a code-token rather than a normal token (which corresponds to sentence boundaries), there is a non-minus_one number.
            # The numbers are global and so don't correspond to within-article sentence idxs
            bounds_tksANDcodes = []
            # bounds_tks = []
            art_only_tk_ids = []
            for b_tAc, (sent_idx, tk_id) in enumerate(zip(unpacked_art.art_snt_ids, art_token_ids_and_maybe_code_ids)):
                if sent_idx == -1:
                    art_only_tk_ids.append(tk_id)
                else:
                    bounds_tksANDcodes.append(b_tAc)
                    # bounds_tks.append(len(art_only_tk_ids))
            bound = bounds_tksANDcodes[within_art_snt_idx]
            # bound_onlyTks = bounds_tks[within_art_snt_idx]
        # chunk starting at snt_bound
        padded_target_end = bound + max_seq_len
        target_end = min(padded_target_end, len(art_token_ids_and_maybe_code_ids))
        chunk = WikiDataset.get_chunk(unpacked_art, start=bound, end=target_end, padded_start=bound, padded_end=padded_target_end,
                                      max_seq_len=max_seq_len, article_idx=article_idx, get_eval_code=get_eval_code, bos_id=bos_id, cc_type=cc_type, code_type=code_type)
        ctx_end = bound # if cc_type != 'insert' else bound_onlyTks # NATHAN: I guess before I was loading a context chunk without inserted codes. But now it is WITH, so I want the bound that includes the codes
        ctx_start = max(0, ctx_end - max_seq_len)
        # padded_ctx_start = ctx_start - (max_seq_len - min(max_seq_len, ctx_end))
        padded_ctx_start = ctx_start # no left padding!
        paddex_ctx_end = ctx_start + max_seq_len
        ctx_chunk = WikiDataset.get_chunk(unpacked_art, ctx_start, ctx_end, padded_ctx_start, paddex_ctx_end, max_seq_len, article_idx, get_eval_code, bos_id, cc_type, code_type=code_type)
        for k,v in ctx_chunk.items():
            chunk[f'ctx_{k}'] = v
        return chunk

    @staticmethod
    def get_chunk(unpacked_art, start, end, padded_start, padded_end, max_seq_len, article_idx,
                  get_eval_code, bos_id, cc_type='adapter', code_type='codes'):
        only_nocc = cc_type == 'none'
        get_art_codes_per_token = (not only_nocc) and (code_type is not None)

        art_ids = unpacked_art.art_ids if cc_type != 'insert' else unpacked_art.art_ids_and_codes
        prepadding = [-1] * (start - padded_start)
        postpadding = [-1] * (padded_end - end)
        ids = prepadding + art_ids[start:end] + postpadding  # temporarily use -1 to help attention_mask
        assert len(ids) == max_seq_len
        IDS_KEY = 'input_ids' if cc_type != 'insert' else 'input_ids_and_codes'
        chunk = {IDS_KEY: torch.tensor(ids)}
        chunk['attention_mask'] = torch.where(chunk[IDS_KEY] == -1, 0, 1)
        # label maks if attention mask or if token id equals BOS token (we DO want to attend to that, but DON'T want it to count towards the loss)
        chunk['label_mask'] = torch.where((chunk[IDS_KEY] == -1) | (chunk[IDS_KEY] == bos_id), False, True)
        # now replace -1 with whatever (eg 42) (due to attention_mask it is ignored anyway)
        chunk[IDS_KEY] = torch.where(chunk[IDS_KEY] == -1, 42, chunk[IDS_KEY])
        if article_idx is not None:
            chunk['article_idx'] = article_idx

        if cc_type != 'insert':
            art_codes_per_token, art_snt_bounds, art_snt_idxs_per_token = unpacked_art.art_codes_per_token, unpacked_art.art_snt_bounds, unpacked_art.art_snt_idxs_per_token
            snt_idxs = list(sorted(set(art_snt_idxs_per_token[start:end])))
            # Get codes for each sentence in chunk by using snt_idxs
            codes = [art_codes_per_token[art_snt_idxs_per_token.index(snt_idx)] for snt_idx in snt_idxs] if get_art_codes_per_token else None
            # Get sentence end boundaries of sentences that occur in this chunk
            post_chunk_boundaries = [b for b in art_snt_bounds if b >= end]
            art_snt_bounds_in_chunk = [b for b in art_snt_bounds if start < b < end] + (
                post_chunk_boundaries[:1] if len(post_chunk_boundaries) > 0 else [])
            snt_bounds = [b - padded_start for b in art_snt_bounds_in_chunk] # relative to start of chunk
            # We store sentence boundaries and non-per-token codes/snt_idxs to save storage space. We expand them again in getitem
            # assert len(snt_bounds) == len(snt_idxs) == len(codes), f"len(snt_bounds)={len(snt_bounds)} != len(snt_idxs)={len(snt_idxs)} != len(codes)={len(codes)}"
            chunk |= {k: torch.tensor(lst) for (k, lst) in
                     {code_type: codes, 'snt_idxs': snt_idxs, 'snt_bounds': snt_bounds}.items() if lst is not None}
        else:
            code_mask = [0 for _ in prepadding] + unpacked_art.art_code_mask[start:end] + [0 for _ in postpadding]
            snt_idxs  = prepadding + unpacked_art.art_snt_ids[start:end] + postpadding
            chunk |= {'code_mask': torch.tensor(code_mask, dtype=bool), 'snt_idxs': torch.tensor(snt_idxs)}
        # region maybe_store_eval_code
        # When using this for PPL evaluation, we are only interested in retrieving the oracle code of the final target word whenever the final target word starts a new sentence.
        # Example for chunk length 4, in the (unlikely) edge case that there is a one-token-length sentence
        # Full article: He went there. Well. That was his choice
        # codes for each sentence: c1 c2 c3
        # input_ids: [He went there. Well.]
        # source_words: [He went there]
        # target_words: [went there. So]
        # codes: [c1 c1 c2 c3] -> Note that at each position is the code of the NEXT token]
        # Because indeed the final target word starts a new sentence, we want the code. It is the element at the second-to-last position of the codes list.
        if get_eval_code:
            is_first_snt = all(e == 0 for e in chunk['attention_mask'][:-2]) and all(e == 1 for e in chunk['attention_mask'][-2:])
            is_nonfirst_snt = any(b == (max_seq_len - 1) for b in snt_bounds)
            is_new_snt = is_first_snt or is_nonfirst_snt
            if is_new_snt:
                chunk_codes_per_token = prepadding + art_codes_per_token[start:end] + postpadding
                eval_code = chunk_codes_per_token[-2]
                chunk['eval_code'] = eval_code
            else:
                chunk['eval_code'] = -100
        # endregion
        return chunk

    @staticmethod
    def unpack_article(article, bos_id, cc_type='adapter', code_type='codes'):

        only_nocc = cc_type == 'none'

        art_ids = [bos_id]
        get_art_codes_per_token = (not only_nocc) and (code_type is not None)
        if get_art_codes_per_token:
            art_codes_per_token = [article[code_type][0]]
        def append_snt_ids(art_ids_and_maybe_codes, snt_token_ids):
            art_ids_and_maybe_codes += (snt_token_ids.tolist() if not type(snt_token_ids) == list else snt_token_ids)  # backwards compatibility, new should already be list
        if cc_type == 'insert':
            if 'score' in code_type:
                raise NotImplementedError
            code_mask = [0] # for BOS token
            art_ids_and_codes = art_ids
            snt_ids = [-1]
            for i, snt in enumerate(article['sentences']):
                art_ids_and_codes += [article[code_type][i]]
                snt_ids += [article['snt_idxs'][i]]
                code_mask += [1]

                snt_token_ids = article['tokenized_sentences'][i]
                snt_len = len(snt_token_ids)
                append_snt_ids(art_ids_and_codes, snt_token_ids)
                snt_ids += [-1]*snt_len
                code_mask += [0]*snt_len
            result = InsertStyleUnpackedArticle(art_ids_and_codes=art_ids_and_codes, art_snt_ids=snt_ids, art_code_mask=code_mask)
        else:
            art_snt_idxs_per_token = [article['snt_idxs'][0]]
            art_snt_bounds = []
            for i, snt in enumerate(article['sentences']):
                # snt_token_ids = tokenizer.encode(snt, return_tensors="pt")[0]
                snt_token_ids = article['tokenized_sentences'][i]
                snt_len = len(snt_token_ids)
                append_snt_ids(art_ids, snt_token_ids)
                if get_art_codes_per_token:
                    art_codes_per_token += [article[code_type][i]] * (snt_len - 1) + (
                        [article[code_type][i + 1]] if i < len(article[code_type]) - 1 else [article[code_type][i]])
                art_snt_idxs_per_token += [article['snt_idxs'][i]] * (snt_len - 1) + (
                    [article['snt_idxs'][i + 1]] if i < len(article['snt_idxs']) - 1 else [article['snt_idxs'][i]])
                art_snt_bounds += [snt_len + 1] if len(art_snt_bounds) == 0 else [
                    art_snt_bounds[-1] + snt_len]  # +1 because of BOS token
            # Example of above:
            # " He went there. He loved it. The leaves were falling again. Autumn nights. The next day was chilly."
            # art_ids:                [BOS He  went  there. He  loved it. The leaves were falling again. Autumn nights. The next day was chilly.] (but with the ID of each token)
            # art_codes_per_token:    [42  42  42    95     95  95    8   8   8      8    8       20     20     33      33  33   33  33  33     ] # Note that the last token has the code of the next sentence
            # art_snt_idxs_per_token: [0   0   0     1      1   1     2   2   2      2    2       3      3      4       4   4    4   4   4      ]
            # art_snt_bounds:         [                    4             7                              12             14                     19]
            # return art_codes_per_token, art_ids, art_snt_bounds, art_snt_idxs_per_token
            result = UnpackedArticle(art_codes_per_token=art_codes_per_token if get_art_codes_per_token else None,
                                   art_ids=art_ids,
                                   art_snt_bounds=art_snt_bounds,
                                   art_snt_idxs_per_token=art_snt_idxs_per_token)
        return result




    def load_tokenized_and_sentencized_articles(self, split):
        tokenizer_name = type(self.tokenizer).__name__
        tokenized_and_sentencized_articles_path = jn(self.pickle_dir(split),f"{tokenizer_name}-tokenized_and_sentencized.pkl")
        if os.path.exists(tokenized_and_sentencized_articles_path):
            print(f"Loading tokenized and sentencized articles from {tokenized_and_sentencized_articles_path}"); s = time()
            with open(tokenized_and_sentencized_articles_path, 'rb') as f:
                tokenized_and_sentencized_articles = pickle.load(f)
            print(f"Loaded in {time() - s} seconds")
        else:
            sentencized_articles = self.load_sentencized_articles(split)
            print(f"Tokenizing and sentencizing {len(sentencized_articles)} articles")
            # with Pool() as p:
            #     tokenized_and_sentencized_articles = p.map(self.get_tokenized_and_sentencized_article, sentencized_articles)
            for article in tqdm(sentencized_articles):
                article['tokenized_sentences'] = [self.tokenizer.encode(sent) for sent in article['sentences']]
            tokenized_and_sentencized_articles = sentencized_articles
            with open(tokenized_and_sentencized_articles_path, 'wb') as f:
                print(f"Saving {len(tokenized_and_sentencized_articles)} tokenized and sentencized articles to {tokenized_and_sentencized_articles_path}")
                pickle.dump(tokenized_and_sentencized_articles, f)
        return tokenized_and_sentencized_articles

    def load_spacy(self):
        if self.nlp is None:
            import spacy # lazy load cuz takes a few seconds which is annoying during quick debugging
            self.nlp = spacy.load("en_core_web_sm", disable=['tagger', 'parser', 'ner', 'lemmatizer'])
            self.nlp.add_pipe('sentencizer')
            self.nlp.max_length = 1e7

    def load_sentencized_articles(self, split):
        sentencized_articles_path = jn(self.pickle_dir(split),'sentencized.pkl')
        if os.path.exists(sentencized_articles_path):
            print(f"Loading sentencized articles from {sentencized_articles_path}"); t = time()
            with open(sentencized_articles_path, 'rb') as f:
                sentencized_articles = pickle.load(f)
            print(f"Loaded in {time() - t} seconds")
        else:
            articles = self.load_unsentencized_articles(split)
            # sentencize articles in parallel
            print(f"Sentencizing {len(articles)} articles")
            self.load_spacy()
            with Pool() as p:
                sentencized_articles = p.map(self.get_sentencized_article, articles)
            with open(sentencized_articles_path, 'wb') as f:
                print(f"Saving {len(sentencized_articles)} sentencized articles to {sentencized_articles_path}")
                pickle.dump(sentencized_articles, f)
        return sentencized_articles

    def code_pickle_dir(self, split, cluster_count, kmeans_path=None, split_num_articles=None):
        en = FULL2SHORT_EMBEDDER_NAME[self.embedder_name]
        if kmeans_path is None:
            cc,na,s = cluster_count, len(self.split2idxs_seed['train']['idxs']), self.split2idxs_seed['train']['seed']
        else:
            # # kmeans_path can look like [self.root_pickle_dir]/train_enwiki_a28531_s42/kmeans_c1024.pkl
            # na, s, cc = re.findall(r'a(\d+)_s(\d+)/kmeans_c(\d+).pkl', kmeans_path)[0]
            # kmeans_path can look like [self.root_pickle_dir]/train_enwiki_a28531_s42/kmeans_[embedder]c1024.pkl with the embedders any of the keys in constants.SHORT2FULL_EMBEDDER_NAME
            na, s, e, cc = re.findall(r'a(\d+)_s(\d+)/kmeans_(\w+)c(\d+).pkl', kmeans_path)[0]
            assert e == en

        subdir = f'{en}c{cc}_a{na}_s{s}'
        return jn(self.pickle_dir(split, num_articles=split_num_articles), subdir)


    def load_coded_articles(self, kwargs, skip_noneval_load, skip_train_load, rand_codebook=False):
        args = Namespace(**kwargs)
        self.split2coded_articles = {}
        load_embs = self.soft_planner_targets
        if load_embs:
            self.split2sent_embs = {}
        kmeans = None
        for split in ['train'] + [s for s in self.splits if s != 'train']: # Ensure train split is first, as it might form the basis for kmeans of other splits
            num_arts = min(len(self.split2idxs_seed[split]['idxs']), self.num_train_articles) # Also limit nontrain splits if mini-run
            dir = self.code_pickle_dir(split, args.cluster_count, args.kmeans_path, num_arts)
            os.makedirs(dir, exist_ok=True)
            pickle_path = jn(dir,"book_and_coded_sents.pkl") # For backwards compatibility
            art_json_path = jn(dir,f"coded_{type(self.tokenizer).__name__}-tokenized_arts.json")
            book_np_path = jn(dir,"codebook.npy")
            if not all(os.path.exists(p) for p in [art_json_path, book_np_path]):
                if os.path.exists(pickle_path):
                    print(f"{pickle_path} exists, but not {art_json_path} or {book_np_path}, so loading from pickle path and saving to json and npy")
                    print(f"Loading from {pickle_path}"); s = time()
                    with open(pickle_path, 'rb') as f:
                        pkl = pickle.load(f)
                    print(f"Done loading in {time() - s} seconds")

                    # articles as json instead of pkl
                    articles, codebook = pkl['articles'], pkl['codebook']
                    assert 'tokenized_sentences' in articles[0], f"{pickle_path} is old format without tokenized_sentences. Delete and remake."
                else:
                    print(f"{pickle_path} nor {art_json_path} nor {book_np_path} exists, so making and saving art_json_path and book_np_path")
                    # sentencized_articles = self.load_sentencized_articles(split)
                    tokenized_and_sentencized_articles = self.load_tokenized_and_sentencized_articles(split)
                    if kmeans is None:
                        kmeans = self.get_kmeans(tokenized_and_sentencized_articles if split == 'train' else None, kwargs)
                    kmeans_predictions = self.get_kmeans_predictions(args, kmeans, split,
                                                                     tokenized_and_sentencized_articles, **kwargs)
                    snt_idx = 0
                    for article in tqdm(tokenized_and_sentencized_articles):
                        article['codes'] = []
                        article['snt_idxs'] = []
                        for _ in article['sentences']:
                            article['codes'].append(kmeans_predictions[snt_idx])
                            article['snt_idxs'].append(snt_idx)
                            snt_idx += 1
                    coded_articles = tokenized_and_sentencized_articles
                    if args.kmeans_cluster_debug:
                        check_alignment(coded_articles, kmeans, self.embedder_name)
                    articles, codebook = coded_articles[:self.num_train_articles], kmeans.cluster_centers_  # :self.num_train_articles for if mini run: don't want all 1k or 5k val / test arts

                # ensure json serializable
                for article in tqdm(articles, desc="Making json serializable"):
                    article['tokenized_sentences'] = [sent.tolist() if type(sent) == torch.Tensor else sent for sent in article['tokenized_sentences'] ]
                    article['codes'] = [int(c) for c in article['codes']]
                with open(art_json_path, 'w') as f:
                    print(f"Saving to {art_json_path}"); t = time()
                    json.dump(articles, f)
                print(f"Done saving in {time() - t} seconds")

                # codebook as np array
                if os.path.exists(book_np_path):
                    existing_codebook = np.load(book_np_path)
                    assert np.array_equal(existing_codebook, codebook), f"Codebook from pickle and codebook from kmeans are not equal. {book_np_path} already exists, but is not equal to the new codebook. Check why not."
                else:
                    np.save(book_np_path, codebook)

            # self.init_from_coded_pkl(pickle_path)
            skip_split_load = False
            if skip_noneval_load:
                skip_split_load = split != ('val' if not args.testeval else 'test')
            else:
                if skip_train_load:
                    skip_split_load = split == 'train'
            if not skip_split_load:

                print(f"Loading json from {art_json_path}"); s = time()
                with open(art_json_path, 'r') as f:
                    articles = json.load(f)
                print(f"Done loading in {time() - s} seconds")

                codebook = np.load(book_np_path)

                if split == 'train' or skip_noneval_load or (skip_train_load and split == 'esval'):
                    self.codebook = codebook
                else:
                    assert np.array_equal(self.codebook, codebook)

                # For backwards compatibility: old code didn't store tokenized sentences in pickle
                if 'tokenized_sentences' not in articles[0]:
                    for article in tqdm(articles, desc="Retroactively tokenizing sentences"):
                        article['tokenized_sentences'] = [self.tokenizer.encode(sent, return_tensors="pt")[0] for sent in article['sentences']]
                    # Parallellizing gives RuntimeError: received 0 items of ancdata. Seems fast enough non-parallel too tbh
                    # with Pool() as p:
                    #     updated_articles = p.map(partial(get_tokenized_and_sentencized_article, self.tokenizer), articles)
                    # articles = updated_articles
                    # with open(pickle_path, 'wb') as f:
                    #     print(f"Saving updated pickle to {pickle_path}")
                    #     pickle.dump({'codebook': codebook, 'articles': articles}, f)
                    with open(art_json_path, 'w') as f:
                        print(f"Saving updated json to {art_json_path}")
                        json.dump(articles, f)
                    with open(book_np_path, 'wb') as f:
                        print(f"Saving updated codebook to {book_np_path}")
                        np.save(f, codebook)
                if split == 'test':
                    articles = articles[:1000] # All 5k makes eval be unnecessarily slow
                self.split2coded_articles[split] = articles

                if load_embs and split not in self.split2sent_embs:
                    self.split2sent_embs[split] = self.load_split_embeddings(split, articles, **kwargs)
            else:
                print(f"Skipping loading of {split} articles to save some minutes of time")
                self.split2coded_articles[split] = None
        if rand_codebook:
            self.codebook = np.random.rand(*self.codebook.shape).astype(np.float32)

    def get_kmeans_predictions(self, args, kmeans, split, tokenized_and_sentencized_articles, sbert_batch_size, **kwargs):
        THRESH = 600000
        if not ((split == 'train') and args.kmeans_path is None):
            if len(tokenized_and_sentencized_articles) < THRESH:
                split_embeddings = self.load_split_embeddings(split, tokenized_and_sentencized_articles,
                                                              sbert_batch_size, **kwargs)
                if self.no_cluster:
                    self.split2sent_embs[split] = split_embeddings
                kmeans_predictions = kmeans.predict(split_embeddings)
            else:
                # If we have a lot of articles, we can't load all embeddings at once. Instead, we load them in chunks and predict on them
                print(f"Too many articles to load all embeddings at once. Loading in chunks and predicting on them")
                chunk_size = 100000

                kmeans_preds_ckpt_path = jn(self.code_pickle_dir(split, args.cluster_count, args.kmeans_path, len(tokenized_and_sentencized_articles)), "kmeans_preds.pkl")
                if os.path.exists(kmeans_preds_ckpt_path):
                    start_idx, kmeans_predictions = pload(kmeans_preds_ckpt_path)
                else:
                    kmeans_predictions = []
                    start_idx = 0

                for i in tqdm(range(start_idx, len(tokenized_and_sentencized_articles), chunk_size), desc="Predicting kmeans"):
                    split_embeddings = self.get_sbert_embeddings(sbert_batch_size, tokenized_and_sentencized_articles[i:i + chunk_size])
                    kmeans_predictions += kmeans.predict(split_embeddings).tolist()
                    pdump((i+chunk_size, kmeans_predictions), kmeans_preds_ckpt_path)
                pdump((len(tokenized_and_sentencized_articles), kmeans_predictions), kmeans_preds_ckpt_path)
        else:
            kmeans_predictions = kmeans.labels_  # Doing this with the idea that it is faster. However, predicting with kmeans is pretty fast anyway tbh, so maybe this is just unnecessarily complex/brittle
        assert len(kmeans_predictions) == len([1 for art in tokenized_and_sentencized_articles for _ in art['sentences']])
        return kmeans_predictions

    def add_planner_codes_or_scores(self, planner, args, logger, splits):
        c_or_s = 'scores' if scores_iso_codes(args) else 'codes'
        for split in splits:
            articles = self.split2coded_articles[split]
            if args.fixed_code:
                pcodess = [[0]*len(a['codes']) for a in articles]
            else:
                pcodes_path = self.get_pcodes_or_scores_path(args, logger, split)
                if os.path.exists(pcodes_path):
                    print(f"Loading planner predicted {c_or_s} from {pcodes_path}")
                    with open(pcodes_path, 'rb') as f:
                        pcodess = pickle.load(f)
                else:
                    # region Calculate planner codes
                    pcodess = planner.compute_codes_or_scores_for_articles(articles, self.tokenizer.bos_token_id, scores_iso_codes=scores_iso_codes(args))
                    # endregion

                    # region Store planner codes for future loading
                    os.makedirs(os.path.dirname(pcodes_path), exist_ok=True)
                    print(f"Saving planner predicted codes to {pcodes_path}")
                    with open(pcodes_path, 'wb') as f:
                        pickle.dump(pcodess, f)
                    # endregion

            # region Add loaded planner codes to articles
            assert len(pcodess) == len(articles)
            for pcodes, article in tqdm(zip(pcodess, articles), desc=f"Adding planner {c_or_s} to {split} articles",
                                        total=len(articles)):
                assert len(pcodes) == len(article['codes'])
                article[f'planner_{c_or_s}'] = pcodes
            # endregion

    def get_pcodes_or_scores_path(self, args, logger, split):
        c_or_s = 'scores' if scores_iso_codes(args) else 'codes'
        pcodes_filename = f"planner_predicted_{c_or_s}.pkl"
        general_dir = self.pickle_dir(split, num_articles=len(self.split2coded_articles[split]))

        def get_spec_dir(ckpt_path, skip_train):
            if ckpt_path is not None and skip_train:
                # use most specific dir (which should equal a wid) + filename of the path for spec_dir
                return jn(os.path.basename(os.path.dirname(ckpt_path)), os.path.splitext(os.path.basename(ckpt_path))[0])
            else:
                exp_id = logger.experiment.id
                LAST_OR_BEST = 'LAST' if args.last_iso_best_ckpt else 'BEST'
                pattern = jn(args.checkpoint_dir,exp_id,f'*{LAST_OR_BEST}*.ckpt')
                matches = glob(jn(args.checkpoint_dir,exp_id,f'*{LAST_OR_BEST}*.ckpt'))
                assert len(matches) > 0, f"No ckpt matches pattern {pattern}"
                filename = os.path.basename(os.path.splitext(matches[0])[0])
                return jn(exp_id, filename)

        if c_or_s == 'codes' and not args.straight_through:
            spec_dir = get_spec_dir(args.mz_ckpt_path, args.skip_mz_train)
        else:
            spec_dir = get_spec_dir(args.jplm_ckpt_path, args.skip_jplm_train)


        pcodes_path = jn(general_dir, spec_dir, pcodes_filename)
        return pcodes_path

    def get_sentencized_article(self, article):
        sent_split_text = [sent.text for sent in self.nlp(article["text"]).sents]
        return {'sentences':sent_split_text} | {k: article[k] for k in article if k != 'text'}

    def get_sentencized_text(self, text_str):
        return [sent.text for sent in self.nlp(text_str).sents]

    def prep_chunk_for_batch(self, chunk,split=None):
        result = self.expand_chunk(chunk) if self.cc_type != 'insert' else chunk
        if self.fixed_code:
            prefixes = [''] + ([] if 'ctx_input_ids' not in result else ['ctx_'])
            for p in prefixes:
                if self.cc_type != 'insert':
                    result[f'{p}codes'] = torch.zeros_like(result[f'{p}codes'])
                else:
                    result[f'{p}input_ids_and_codes'] = torch.where(chunk['code_mask'],torch.zeros_like(chunk[f'{p}input_ids_and_codes']), chunk[f'{p}input_ids_and_codes'])
        if self.soft_planner_targets:
            result['split'] = split
        return result

    @staticmethod
    def expand_chunk(chunk, code_type='codes',cc_type=None, args=None):
        if code_type == 'deduce':
            if cc_type == 'none':
                code_type = 'none'
            else:
                # See which one of codes, planner_codes, planner_scores is present, and use that. Throw error if more than one is present
                code_types = ['codes', 'planner_codes', 'planner_scores']
                present_code_types = [ct for ct in code_types if ct in chunk]
                if len(present_code_types) == 1:
                    code_type = present_code_types[0]
                else:
                    if args is not None and (args.fixed_code or args.uniform_mix):
                        # In this case, it could be no codes are present (because the method might be called in a case where we just replace the codes/scores with fixed/uniform anyway
                        code_type = 'none'
                    else:
                        raise ValueError(f"Different than one of {code_types} present in chunk: {present_code_types}")
        # expand codes and snt_idxs based on snt_bounds
        prefixes = [''] + ([] if 'ctx_input_ids' not in chunk else ['ctx_'])
        full_result = chunk.copy()
        for p in prefixes:
            has_codes = f'{p}{code_type}' in chunk
            codes, snt_idxs = [], []
            og_bounds = chunk[f'{p}snt_bounds']
            shifted_bounds = [b - 1 for b in og_bounds if (b-1) != 0] # codes and snt_idxs are shifted by 1 (code/snt_idx for last token of a sentence is that of the next sentence)
            acceptable_num_codes = [len(og_bounds)]
            if og_bounds[-1] == len(chunk[f'{p}input_ids']) or ((og_bounds[-1] < len(chunk[f'{p}input_ids'])) and (chunk[f'{p}attention_mask'][og_bounds[-1]] == 0)):
                acceptable_num_codes += [len(og_bounds) + 1] # if at end of sentence but not at end of article. (Case with and without right-padding)
            if og_bounds[0] == 1:
                acceptable_num_codes += [len(og_bounds) - 1] # if sentence bound after very first word. The first word has the code of the next sentence, resulting in one fewer code
            k = f'{p}{code_type}' if has_codes in chunk else f'{p}snt_idxs'
            assert len(chunk[k]) in acceptable_num_codes, f"len(chunk[{k}'])={len(chunk[f'{k}'])} not in {acceptable_num_codes}"
            # if og_bounds[-1] == len(chunk[f'{p}input_ids']):
            #     # Either chunk end aligned with sentence end, in which case there will be one code 'too many' (namely that of the following sentence). TODO There is no good reason for this, but would need to remake chunks to fix it
            #     # Or chunk end aligns with article end, in which case there should be as many codes as bounds
            #     assert ((len(og_bounds) + 1 == len(chunk[f'{p}codes'])) or (len(og_bounds) == len(chunk[f'{p}codes']))), \
            #         f"len(og_bounds)={len(og_bounds)} + 1 != len(chunk[f'{p}codes'])={len(chunk[f'{p}codes'])}"
            # else:
            #     assert len(og_bounds) == len(chunk[f'{p}codes']), f"len(shifted_bounds={[b.item() for b in shifted_bounds]}) != len(chunk[f'{p}codes']={chunk[f'{p}codes'].tolist()})"
            last_b = 0
            for i, b in enumerate(shifted_bounds):
                b = min(b, len(chunk[f'{p}input_ids']))
                snt_len = b - last_b
                if has_codes:
                    codes += [chunk[f'{p}{code_type}'][i]] * snt_len
                snt_idxs += [chunk[f'{p}snt_idxs'][i]] * snt_len
                last_b = b
            if last_b < len(chunk[f'{p}input_ids']):
                # This means the end of the article was reached within this chunk. input_ids is already padded to max_seq_len, so just pad codes and snt_idxs
                # We pad here with the last code/snt_idx. Really it shouldn't matter what we pad it with, but I have a feeling that some code somewhere might depend on this particular padding :P
                if len(codes if has_codes else snt_idxs) == 0:
                    assert og_bounds == torch.tensor([1])
                    assert len(chunk[f'{p}{code_type}']) == 1
                    if has_codes:
                        codes = [chunk[f'{p}{code_type}'][0]] * len(chunk[f'{p}input_ids'])
                    snt_idxs = [chunk[f'{p}snt_idxs'][0]] * len(chunk[f'{p}input_ids'])
                else:
                    pad_len = len(chunk[f'{p}input_ids']) - last_b
                    if has_codes:
                        codes += [codes[-1]] * pad_len
                    snt_idxs += [snt_idxs[-1]] * pad_len
            # result = {'input_ids': chunk[f'{p}input_ids'], 'attention_mask': chunk[f'{p}attention_mask'],
            #           'article_idx': chunk[f'{p}article_idx'],
            #           'codes': torch.tensor(codes), 'snt_idxs': torch.tensor(snt_idxs)}
            if has_codes:
                full_result |= {f'{p}{code_type}': torch.stack(codes) if 'score' in code_type else torch.tensor(codes)}
            full_result |= {f'{p}snt_idxs': torch.tensor(snt_idxs)}
            full_result.pop(f'{p}snt_bounds')
        return full_result


    def load_unsentencized_articles(self, split):
        pickle_path = jn(self.pickle_dir(split),'unsentencized.pkl')
        if os.path.exists(pickle_path):
            with open(pickle_path, 'rb') as f:
                articles = pickle.load(f)
        else:
            articles = self.get_articles(split)
            with open(pickle_path, 'wb') as f:
                print(f"Saving {len(articles)} articles to {pickle_path}")
                pickle.dump(articles, f)
        return articles

    def get_articles(self, split):
        raise NotImplementedError


    def format_name(self, split, num_articles, seed):
        return self.format_name_static(self.prefix, split, num_articles, seed)

    @staticmethod
    def format_name_static(prefix, split, num_articles, seed):
        return f"{split}_{prefix}_a{num_articles}_s{seed}"

class DolmaDataset(MyDataset):

    total_article_count = DOLMA_ARTICLE_COUNT
    prefix = "dolma"

    def get_articles(self, split):
        # if not hasattr(self, 'split2raw_articles'):
        #     dataset = datasets.load_dataset("wikipedia", "20220301.en")
        #     all_articles = dataset['train']
        #     self.split2raw_articles = {spl: [all_articles[i] for i in tqdm(self.split2idxs_seed[spl]['idxs'], desc=f'Selecting {spl} split')] for spl in ['train', 'esval', 'val', 'test']}
        # articles = []
        # for article in self.split2raw_articles[split]:
        #     articles.append({'title': article['title'], 'text': article['text']})
        # return articles
        if not hasattr(self, 'split2raw_articles'):
            dataset = datasets.load_dataset("allenai/dolma", "v1_6-sample")
            all_articles = dataset['train']
            self.split2raw_articles = {spl: [all_articles[i] for i in tqdm(self.split2idxs_seed[spl]['idxs'], desc=f'Selecting {spl} split')] for spl in ['train', 'esval', 'val', 'test']}
        articles = []
        for article in self.split2raw_articles[split]:
            articles.append({'text': article['text'],
                             'source': article['source']}) # no titles in dolma
        return articles

class WikiDataset(MyDataset):

    ...


def maybe_expand(k,v, size, cc_type):
    if k in ['input_ids', 'attention_mask', 'label_mask', 'input_ids_and_codes', 'code_mask']:
        return v
    elif cc_type != 'insert' and k in ['codes', 'snt_idxs', 'snt_bounds']:
        return torch.cat((v, -1*torch.ones(size - len(v), dtype=v.dtype, device=v.device)))
    elif cc_type == 'insert' and (k == 'snt_idxs'):
        return v
    elif k == 'article_idx':
        return torch.ones(size) * v
    else:
        raise ValueError(f"key {k} not recognized")


def maybe_unexpand(k, v, cc_type):
    if k in ['input_ids', 'attention_mask', 'label_mask', 'input_ids_and_codes', 'code_mask']:
        return v.to(dtype=(torch.long if k not in ['label_mask', 'code_mask'] else torch.bool))
    elif cc_type != 'insert' and k in ['codes', 'snt_idxs', 'snt_bounds']:
        return v[v != -1].to(dtype=(torch.long if k != 'codes' else torch.int32))
    elif cc_type == 'insert' and (k == 'snt_idxs'):
        return v
    elif k == 'article_idx':
        return int(v[0].item())
    else:
        raise ValueError(f"key {k} not recognized")

def chunk2dict(chunk, cc_type):
    return {k: maybe_unexpand(k, chunk[j], cc_type) for j, k in enumerate(get_chunk_keys(cc_type))}


class WikisplitDataset(WikiDataset):

    def __init__(self, dataset, split, nocc=False):
        self.dataset = dataset
        self.split = split
        self.nocc = nocc
        self.flat2nested_idx = {f: (a, s) for f, (a, s) in
                                enumerate((aa,ss) for (aa, article) in enumerate(self.dataset.split2coded_articles[self.core_split]) for ss in range(len(article['sentences'])))
                                }

    @property
    def chunks(self):
        return self.dataset.split2chunks[self.split]

    @property
    def core_split(self):
        # [split] -> [split], [split]-nocc -> [split]
        return self.split.split('-nocc')[0]

    @property
    def chunk_keys(self):
        return get_chunk_keys(self.cc_type)

    @property
    def cc_type(self):
        c = self.dataset.cc_type
        if c == 'insert' and 'nocc' in self.split:
            return 'adapter'
        else:
            return c

    def __len__(self):
        if (self.dataset.phase == 'lm') or self.dataset.no_sbound:
            return len(self.chunks)
        else:
            return len(self.flat2nested_idx)

    def __getitem__(self, idx):
        if self.dataset.phase == 'lm':
            chunk = chunk2dict(self.chunks[idx], self.cc_type)
            chunk = self.maybe_update_chunk_for_lm(chunk, idx)
            return chunk
        elif self.dataset.phase == 'mz':
            article_idx, within_art_snt_idx = self.flat2nested_idx[idx]
            article = self.dataset.split2coded_articles[self.split][article_idx]
            unpacked_art = WikiDataset.unpack_article(article, self.dataset.tokenizer.bos_token_id, self.dataset.cc_type)
            chunk = WikiDataset.get_chunks_at_art_and_snt_idxs(article_idx, within_art_snt_idx, get_eval_code=False, max_seq_len=self.dataset.max_seq_len, unpacked_art=unpacked_art, bos_id=self.dataset.tokenizer.bos_token_id, cc_type=self.dataset.cc_type)
            return self.dataset.prep_chunk_for_batch(chunk, self.split)

    def maybe_update_chunk_for_lm(self, chunk, idx=None, prev_chunk=None):
        if not self.nocc:
            chunk = self.dataset.prep_chunk_for_batch(chunk, split=self.split)
            if self.dataset.kwargs['joint_planner_lm']:
                keys = ['input_ids', 'attention_mask', 'snt_idxs']
                assert not (idx is None and prev_chunk is None)
                if (prev_chunk is not None) or (idx > 0 and
                        (prev_chunk := chunk2dict(self.chunks[idx - 1], self.cc_type))['article_idx'] == chunk[
                            'article_idx']  # this checks if this chunk didn't start a new article
                ):
                    # For sentences that start within the current chunk, we know we can always get the up-to-N tokens preceding that sentences from a combination of the current chunk and the previous chunk
                    prev_chunk = {k: v for k, v in self.dataset.prep_chunk_for_batch(prev_chunk).items() if k in keys}

                    # For tokens belonging to a sentence that started before the current chunk (which could have started even multiple chunks ago for very long sentences), we find the tokens preceding that sentence using WikiDataset.get_chunks_at_art_and_snt_idxs

                    article_idx, within_art_snt_idx = self.flat2nested_idx[chunk['snt_idxs'][0].item()]
                    article = self.dataset.split2coded_articles[self.split][article_idx]
                    unpacked_art = WikiDataset.unpack_article(article, self.dataset.tokenizer.bos_token_id, self.dataset.cc_type)
                    chunks_before_and_after_snt_start = WikiDataset.get_chunks_at_art_and_snt_idxs(article_idx,
                                                                                                   within_art_snt_idx,
                                                                                                   get_eval_code=False,
                                                                                                   max_seq_len=self.dataset.max_seq_len,
                                                                                                   unpacked_art=unpacked_art,
                                                                                                   bos_id=self.dataset.tokenizer.bos_token_id,
                                                                                                   cc_type=self.dataset.cc_type)
                    first_sent_ctx_chunk = {new_k: v for old_k, v in
                                            self.dataset.prep_chunk_for_batch(chunks_before_and_after_snt_start).items()
                                            if (old_k.startswith('ctx_') and ((new_k := old_k[len('ctx_'):]) in keys))}

                else:
                    prev_chunk, first_sent_ctx_chunk = [
                        {k: torch.zeros_like(v) - (1 if k == 'snt_idxs' else 0) for k, v in chunk.items() if k in keys}
                        for _ in range(2)]
                    first_sent_ctx_chunk['input_ids'][-1] = self.dataset.tokenizer.bos_token_id
                    first_sent_ctx_chunk['attention_mask'][-1] = 1
                for k in prev_chunk:
                    chunk[f'prev_{k}'] = prev_chunk[k]
                    chunk[f'first_sent_ctx_{k}'] = first_sent_ctx_chunk[k]
            if self.dataset.no_cluster:
                if self.dataset.fixed_code:
                    raise ValueError("choose either no_cluster or fixed_code, not both")
                chunk['sent_embs'] = self.dataset.split2sent_embs[self.split][chunk['snt_idxs']]
            if self.dataset.use_planner_codes:
                article = self.dataset.split2coded_articles[self.core_split][chunk['article_idx']]
                sntidx2plannercode = dict(zip(article['snt_idxs'], article['planner_codes']))
                if self.dataset.cc_type == 'insert' and not self.nocc:
                    planner_codes = torch.tensor(
                        [sntidx2plannercode[snt_idx.item()] if snt_idx.item() in sntidx2plannercode else -1 for snt_idx
                         in chunk['snt_idxs']]).to(torch.int32)
                    chunk['input_ids_and_planner_codes'] = torch.where(chunk['code_mask'].to(torch.bool), planner_codes,
                                                                       chunk['input_ids_and_codes'])
                else:
                    chunk['planner_codes'] = torch.tensor(
                        [sntidx2plannercode[snt_idx.item()] for snt_idx in chunk['snt_idxs']]).to(torch.int32)
        else:
            # delete keys: codes, snt_idxs, snt_bounds
            chunk = {k: v for (k, v) in chunk.items() if k not in ['codes', 'snt_idxs', 'snt_bounds']}
        return chunk

    def __getattr__(self, item):
        if item not in self.__dict__:
            return getattr(self.dataset, item)
        else:
            return self.__dict__[item]


class IterableWrapper(torch.utils.data.IterableDataset):

    def __init__(self, mapstyle_dataset, stride=None,split=None):
        self._ds = mapstyle_dataset
        self.stride = stride
        self.split = split


    @property
    def ds(self):
        return self._ds # if not (type(self._ds) == WikisplitDataset) else self._ds.dataset

    def __getattr__(self, item):
        if item not in self.__dict__:
            return getattr(self._ds, item)
        else:
            return self.__dict__[item]

    def __setattr__(self, item, value):
        if item not in  ('ds', '_ds') and hasattr(self._ds, item) and item not in self.__dict__:
            setattr(self._ds, item, value)
        else:
            self.__dict__[item] = value

    def set_phase(self, phase):
        self._ds.set_phase(phase)

    # @property
    # def phase(self):
    #     return self.ds.phase
    #
    # @property
    # def seed(self):
    #     return self.ds.seed
    #
    # @property
    # def max_seq_len(self):
    #     return self.ds.max_seq_len
    #
    # @property
    # def tokenizer(self):
    #     return self.ds.tokenizer

    def get_splits(self):
        assert self.split is None, "Trying to call get_splits on what is already a split part"
        split2subset = self.ds.get_splits()
        result = {split: IterableWrapper(ds, self.stride, split) for split, ds in split2subset.items()}
        if 'train' in result:
            result['train'] = ShuffleDataset(result['train'], self.kwargs['buffer_size'])
        return result

    def __iter__(self):
        if self.phase == 'lm':
            pairs_at_snt_bounds = False
        elif self.phase == 'mz':
            pairs_at_snt_bounds = True
        else:
            raise ValueError(f"phase {self.dataset.phase} not recognized")
        articles = self.ds.split2coded_articles[self.split]
        for i, article in enumerate(articles):
            prev_chunk = None # used in case of args.joint_planner_lm
            for chunk in WikiDataset.make_article_chunks(article, self.max_seq_len, self.tokenizer, stride=self.stride, article_idx=i, pairs_at_snt_bounds=pairs_at_snt_bounds):
                match self.phase:
                    case 'lm':
                        yield self.ds.maybe_update_chunk_for_lm(chunk, prev_chunk=prev_chunk)
                        prev_chunk = chunk
                    case 'mz':
                        yield self.ds.prep_chunk_for_batch(chunk)

    def __len__(self):
        if self.phase == 'mz':
            if self.split is not None:
                # articles = self.ds.get_articles_for_split(self.split)
                articles = self.ds.split2coded_articles[self.split]
            else:
                articles = sum([self.ds.split2coded_articles[split] for split in self.ds.splits], [])
            return sum([len(article['sentences']) for article in articles])
        else:
            raise NotImplementedError("len not implemented for phase {self.phase}")



class FullWikiDataset(WikiDataset):

    total_article_count = EN_WIKI_ARTICLE_COUNT
    prefix = "enwiki" # Not sure if this is good python practice :P I want a class property but this seems overkill: https://stackoverflow.com/questions/5189699/how-to-make-a-class-property

    def get_articles(self, split):
        if not hasattr(self, 'split2raw_articles'):
            with gzip.open(EN_WIKI_FILE, 'r') as f:
                all_raw_articles = list(tqdm(f, total=self.total_article_count, desc="Loading articles"))
            self.split2raw_articles = {spl: [all_raw_articles[i] for i in tqdm(self.split2idxs_seed[spl]['idxs'], desc=f'Selecting {spl} split')] for spl in ['train', 'esval', 'val', 'test']}
        articles = []
        for article in self.split2raw_articles[split]:
            article = json.loads(article)
            # text = "".join(article['section_texts'])
            text = "\n".join(["\n".join([f"== {sec_title} ==", text]) for (sec_title, text) in zip(article['section_titles'], article['section_texts'])])
            articles.append({'title': article['title'], 'text': text})
        return articles

class NewFullWikiDataset(FullWikiDataset):

    total_article_count = NEW_EN_WIKI_ARTICLE_COUNT
    prefix = "newenwiki"

    def get_articles(self, split):
        if not hasattr(self, 'split2raw_articles'):
            dataset = datasets.load_dataset("wikipedia", "20220301.en")
            all_articles = dataset['train']
            self.split2raw_articles = {spl: [all_articles[i] for i in tqdm(self.split2idxs_seed[spl]['idxs'], desc=f'Selecting {spl} split')] for spl in ['train', 'esval', 'val', 'test']}
        articles = []
        for article in self.split2raw_articles[split]:
            articles.append({'title': article['title'], 'text': article['text']})
        return articles


class WikiText103Dataset(WikiDataset):

    counts = {"train": 28471, "validation": 60, "test": 60}
    total_article_count = counts['train'] + counts['validation'] + counts['test']
    prefix = "wikitext-103"
    TEST_SIZE = counts['test']

    def get_test_and_remaining_idxs(self):
        return list(range(self.counts['train'] + self.counts['validation'], self.total_article_count)), list(range(self.counts['train'] + self.counts['validation'])) # aka 28531:28591, 0:28531

    # def maybe_set_article_split_counts(self):
    #     N = self.num_articles
    #     assert N <= self.total_article_count
    #     val_count = min(self.splits_article_counts['validation'], N // 3)
    #     num_articles_left = N - val_count
    #     test_count = min(self.splits_article_counts['test'], N // 3)
    #     num_articles_left -= test_count
    #     train_count = num_articles_left
    #
    #     self.article_split_counts = {'train': train_count, 'test': test_count, 'validation': val_count}


    def get_articles(self, split):
        raise NotImplementedError("TODO implement separate splits in Wikitext-103. Not done so far assuming we'll use enwiki.")
        if not hasattr(self, 'split2raw_articles'):
            ...

        # article_dict = {}
        articles = []
        for split in ['train', 'validation']: # Keep test split separate for reporting final results
            # articles = []
            current_article = []
            for i, row in enumerate(dataset[split]):
                # if title, we know we're starting a new article. Titles are of the form ' = <title> = \n', and are preceded and followed by empty rows
                # if re.match(r' = [^=]+ = \n', row['text']):
                #     if (dataset[split][i - 1]['text'] == '') and (dataset[split][i + 1]['text'] == ''):
                #         if len(article) > 0 and article != ['']:
                new_article_started = (len(dataset[split]) > i+2) and (row['text'] == '') and re.match(r' = [^=]+ = \n', dataset[split][i+1]['text']) and (dataset[split][i+2]['text'] == '')
                if new_article_started and len(current_article) > 0:
                    self.append_article(articles, current_article)
                    current_article = []
                    if len(articles) >= self.num_train_articles: #self.article_split_counts[split]:
                        break
                current_article.append(row['text'])
            if len(current_article) > 0:
                self.append_article(articles, current_article)
            # article_dict[split] = articles
        # return article_dict['train'] + article_dict['validation']
        return articles

    def append_article(self, articles, current_article):
        assert (len(current_article) > 3)  # title + empty rows around it
        title = current_article[1]
        extracted_title = re.match(r' = (.+) = \n', title).group(1)
        articles.append({"title": extracted_title,
                         "text": "".join(current_article[3:])})


class WikiDataModule(pl.LightningDataModule):

    def __init__(self, batch_size, mzpt_batch_size, data_pkl_path, max_seq_len, base_model_name=DEFAULT_LLM, dataset=None, phase='lm', num_workers=0, **kwargs):
        super().__init__()
        self.save_hyperparameters(ignore=[
            'dataset','lm_ckpt_path', 'lm_ckpt_wid','mz_ckpt_path','mz_ckpt_wid', 'skip_mz_ckpt_from_wid','skip_lm_ckpt_from_wid','ckpts_wid', 'nowandb', 'plm_ckpt_path', 'plm_ckpt_wid', 'skip_plm_train', 'eval_freq', 'val_check_interval', 'jplm_ckpt_wid', 'jplm_ckpt_path', 'skip_eval','mini', 'jpl_freeze_lm', 'jpl_freeze_planner', 'skip_mz_load','skip_mz_train','nll_then_lev','sentence_transformer_cache', 'out_dir', 'ckpts_dir', 'joint_planner_lm','checkpoint_dir', 'straight_through','lm_epochs'
        ])
        self.batch_size = batch_size if phase == 'lm' else mzpt_batch_size
        self.max_seq_len = max_seq_len
        self.base_model_name = base_model_name

        if dataset is None:
            assert data_pkl_path is not None
            dataset = WikiDataset(tokenizer=myAutoTokenizer(self.base_model_name),coded_pkl_path=self.data_pkl_path, max_seq_len=self.max_seq_len)
        if not callable(getattr(dataset, "__iter__", None)):
            assert dataset.split2chunks is not None, "Chunks not loaded, and not using IterableDataset. Call dataset.load_chunks() first"
        dataset.set_phase(phase)
        self.split2subset = dataset.get_splits()


    def split_dataloader(self, split):
        dataset = self.split2subset[split]
        iterable = type(dataset) in [IterableWrapper, ShuffleDataset]
        if (self.hparams.num_workers > 0) and iterable:
            raise NotImplementedError("IterableWrapper does not support num_workers > 0")

        shuffle = False if iterable else (split == 'train')
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.hparams.num_workers) # for now, dataloading is not the bottleneck so workers=0 is just as fast

    def train_dataloader(self): # Called by lightning internals
        return self.split_dataloader('train')

    def val_dataloader(self): # Called by lightning internals
        return self.split_dataloader('esval' if not ('switch_esval_val' in self.hparams and self.hparams.switch_esval_val) else 'val')

    def test_dataloader(self): # Called by lightning internals
        return self.split_dataloader('val' if not ('switch_esval_val' in self.hparams and self.hparams.switch_esval_val) else 'esval')


def plot_kmeans(args, sse):
    from matplotlib import pyplot as plt
    print(list(zip([args.cluster_count], sse)))
    plt.plot([args.cluster_count], sse)
    plt.savefig(f'kmeans_sse_{"_".join(map(str, [args.cluster_count]))}.png') # used to allow multiple cluster counts. Now this code is not really useful anymore
    plt.show()


def qual_cluster_sentences_check(kmeans, title2sentences_loaded):
    # for N clusters, sample S sentences from each cluster
    N = min(8, kmeans.n_clusters)
    # sample N cluster center idxs
    cluster_center_idxs = np.random.choice(kmeans.n_clusters, size=N, replace=False)
    all_sentences = [s for v in title2sentences_loaded.values() for s in v['sentences']]
    S = 5
    for c in cluster_center_idxs:
        # sample S sentence idxs from kmeans.labels_ == c
        sent_idxs = np.random.choice(np.where(kmeans.labels_ == c)[0], size=S, replace=False)
        sents = [all_sentences[i] for i in sent_idxs]
        # print on separate lines. note that backslashes are not allowed in f-strings
        print(f"Cluster {c}:\n" + "\n".join(sents))


def check_alignment(articles, kmeans, embedder_name):
    # Test using kmeans predict if alignment was ok
    article_ids = np.random.choice(len(articles), size=10, replace=False)
    sent_ids = [np.random.choice(len(articles[i]['sentences']), size=1)[0] for i in article_ids]
    sent_code_pairs = [(articles[i]['sentences'][j], articles[i]['codes'][j]) for i, j in
                       zip(article_ids, sent_ids)]
    sbert = SentenceTransformer(embedder_name)
    for sent, code in sent_code_pairs:
        emb = sbert.encode(sent)
        print(f"Predicted cluster: {kmeans.predict(emb[None])[0]}, actual cluster: {code}")


def my_encode_multi_process(sentencized_articles, pool: Dict[str, object], batch_size: int = 32, chunk_size: int = None):
    """
    This method allows to run encode() on multiple GPUs. The sentences are chunked into smaller packages
    and sent to individual processes, which encode these on the different GPUs. This method is only suitable
    for encoding large sets of sentences

    :param sentencized_articles:
    :param pool: A pool of workers started with SentenceTransformer.start_multi_process_pool
    :param batch_size: Encode sentences with batch size
    :param chunk_size: Sentences are chunked and sent to the individual processes. If none, it determine a sensible size.
    :return: Numpy matrix with all embeddings
    """
    import math
    if chunk_size is None:
        chunk_size = min(math.ceil(len(sentencized_articles) / len(pool["processes"]) / 10), 5000)

    input_queue = pool['input']
    last_chunk_id = 0
    chunk = []

    print("Chunking sentences")
    for sentence in (s for article in sentencized_articles for s in article['sentences']):
        chunk.append(sentence)
        if len(chunk) >= chunk_size:
            input_queue.put([last_chunk_id, batch_size, chunk])
            last_chunk_id += 1
            chunk = []
    print("Finished chunking sentences")

    if len(chunk) > 0:
        input_queue.put([last_chunk_id, batch_size, chunk])
        last_chunk_id += 1

    output_queue = pool['output']
    results_list = sorted([output_queue.get() for _ in tqdm(range(last_chunk_id))
                           ], key=lambda x: x[0])
    embeddings = np.concatenate([result[1] for result in results_list])
    return embeddings

@dataclass
class UnpackedArticle:
    art_codes_per_token: list
    art_ids: list
    art_snt_bounds: list
    art_snt_idxs_per_token: list

@dataclass
class InsertStyleUnpackedArticle:
    art_ids_and_codes: list
    art_code_mask: list
    art_snt_ids: list


class ShuffleDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset, buffer_size):
        super().__init__()
        self.dataset = dataset
        self.buffer_size = buffer_size

    def __iter__(self):
        shufbuf = []
        try:
            dataset_iter = iter(self.dataset)
            for i in range(self.buffer_size):
                shufbuf.append(next(dataset_iter))
        except:
            self.buffer_size = len(shufbuf)

        try:
            while True:
                try:
                    item = next(dataset_iter)
                    evict_idx = random.randint(0, self.buffer_size - 1)
                    yield shufbuf[evict_idx]
                    shufbuf[evict_idx] = item
                except StopIteration:
                    break
            random.shuffle(shufbuf) # Because pop doesn't pop a random element, but always the last element
            while len(shufbuf) > 0:
                yield shufbuf.pop()
        except GeneratorExit:
            pass

    def __getattr__(self, item):
        if item not in self.__dict__:
            return getattr(self.ds, item)
        else:
            return self.__dict__[item]

    def __setattr__(self, item, value):
        if item != 'dataset' and hasattr(self.dataset, item) and item not in self.__dict__:
            setattr(self.dataset, item, value)
        else:
            self.__dict__[item] = value

def get_kmeans_path_from_args(args):
    if args.kmeans_path is not None:
        return args.kmeans_path
    en = FULL2SHORT_EMBEDDER_NAME[args.embedder_name]
    dir = jn(DEFAULT_PICKLE_DIR, f"{MyDataset.format_name_static(args.data_name, 'train', args.max_articles, args.seed)}")
    return jn(dir, f"kmeans_{en}c{args.cluster_count}.pkl")