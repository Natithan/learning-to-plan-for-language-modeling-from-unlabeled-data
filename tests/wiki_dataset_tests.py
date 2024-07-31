from dataset import *
from tokenizer_util import myAutoTokenizer

article = {'sentences': ['The leaves were falling again.','He went there.',],
            'codes': [42, 95],
            'snt_idxs': [0, 1],
           }
tokenizer = myAutoTokenizer('gpt2')
bos_id = tokenizer.bos_token_id
pad_id = 42
snt_token_ids = [[t for t in tokenizer.encode(snt)] for snt in article['sentences'] ]
assert snt_token_ids == [ [464, 5667, 547, 7463, 757, 13], [1544, 1816, 612, 13]]
def test_make_article_chunks_max_stride():
    max_seq_len = 5
    stride = 4
    chunks = WikiDataset.make_article_chunks(article, max_seq_len, tokenizer, stride=stride)
    target = [
        {'input_ids': [bos_id, 464, 5667, 547, 7463], 'codes': [42], 'snt_idxs': [0], 'snt_bounds': [7], 'attention_mask': [1, 1, 1, 1, 1]},
        {'input_ids': [7463, 757, 13, 1544, 1816], 'codes': [42, 95], 'snt_idxs': [0, 1], 'snt_bounds': [3, 7], 'attention_mask': [1, 1, 1, 1, 1]},
        {'input_ids': [1816, 612, 13, pad_id, pad_id], 'codes': [95], 'snt_idxs': [1], 'snt_bounds': [3], 'attention_mask': [1, 1, 1, 0, 0]}
    ]
    check_equal(chunks, target)

    stride = 1
    chunks = WikiDataset.make_article_chunks(article, max_seq_len, tokenizer, stride=stride)
    target = [
        {'input_ids': [pad_id, pad_id, pad_id, bos_id, 464],    'codes': [42],     'snt_idxs': [0],    'snt_bounds': [10],    'attention_mask': [0, 0, 0, 1, 1]},
        {'input_ids': [pad_id, pad_id, bos_id, 464, 5667],      'codes': [42],     'snt_idxs': [0],    'snt_bounds': [9],     'attention_mask': [0, 0, 1, 1, 1]},
        {'input_ids': [pad_id, bos_id, 464, 5667, 547],         'codes': [42],     'snt_idxs': [0],    'snt_bounds': [8],     'attention_mask': [0, 1, 1, 1, 1]},
        {'input_ids': [bos_id, 464, 5667, 547, 7463],           'codes': [42],     'snt_idxs': [0],    'snt_bounds': [7],     'attention_mask': [1, 1, 1, 1, 1]},
        {'input_ids': [464, 5667, 547, 7463, 757],              'codes': [42],     'snt_idxs': [0],    'snt_bounds': [6],     'attention_mask': [1, 1, 1, 1, 1]},
        {'input_ids': [5667, 547, 7463, 757, 13],               'codes': [42, 95], 'snt_idxs': [0, 1], 'snt_bounds': [5],     'attention_mask': [1, 1, 1, 1, 1]},
        {'input_ids': [547, 7463, 757, 13, 1544],               'codes': [42, 95], 'snt_idxs': [0, 1], 'snt_bounds': [4, 8],  'attention_mask': [1, 1, 1, 1, 1]},
        {'input_ids': [7463, 757, 13, 1544, 1816],              'codes': [42, 95], 'snt_idxs': [0, 1], 'snt_bounds': [3, 7],  'attention_mask': [1, 1, 1, 1, 1]},
        {'input_ids': [757, 13, 1544, 1816, 612],               'codes': [42, 95], 'snt_idxs': [0, 1], 'snt_bounds': [2, 6],  'attention_mask': [1, 1, 1, 1, 1]},
        {'input_ids': [13, 1544, 1816, 612, 13],                'codes': [95],     'snt_idxs': [1],    'snt_bounds': [1,5],   'attention_mask': [1, 1, 1, 1, 1]}
    ]
    check_equal(chunks, target)

def test_expand_chunk():
    chunk_ending_mid_sent = {k:torch.tensor(v) for k,v in {
       'input_ids': [0, 1, 2, 3, 4],
       'attention_mask': [1, 1, 1, 1, 1],
       'article_idx': 0,
       'codes': [0,1],
       'snt_idxs': [0,1],
       'snt_bounds': [3,8],
    }.items()}
    expanded_chunk_ending_mid_sent = WikiDataset.expand_chunk(chunk_ending_mid_sent)
    target = {k:torch.tensor(v) for k,v in {
       'input_ids': [0, 1, 2, 3, 4],
       'attention_mask': [1, 1, 1, 1, 1],
       'article_idx': 0,
       'codes': [0,0,1,1,1],
       'snt_idxs': [0,0,1,1,1],
    }.items()}
    check_equal(expanded_chunk_ending_mid_sent, target)
    chunk_ending_with_sent = {k:torch.tensor(v) for k,v in {
         'input_ids': [0, 1, 2, 3, 4],
         'attention_mask': [1, 1, 1, 1, 1],
         'article_idx': 0,
         'codes': [0,1],
         'snt_idxs': [0,1],
         'snt_bounds': [5],
    }.items()}
    expanded_chunk_ending_with_sent = WikiDataset.expand_chunk(chunk_ending_with_sent)
    target = {
         'input_ids': [0, 1, 2, 3, 4],
         'attention_mask': [1, 1, 1, 1, 1],
         'article_idx': 0,
         'codes': [0,0,0,0,0], # really the last code is 1, but we don't use it anyway, so the method just sets it to the next-to-last code
         'snt_idxs': [0,0,0,0,0],
    }
    check_equal(expanded_chunk_ending_with_sent, target)




def check_equal(chunk_or_chunks, target_or_targets):
    # if list, it is multiple chunks. Otherwise convert to length-1 list
    if not isinstance(chunk_or_chunks, list):
        chunks, targets = [chunk_or_chunks], [target_or_targets]
    else:
        chunks, targets = chunk_or_chunks, target_or_targets
    for chunk, target in zip(chunks, targets):
        for k in target.keys():
            el_as_list = chunk[k].tolist() if isinstance(chunk[k], torch.Tensor) else chunk[k]
            ta_as_list = target[k].tolist() if isinstance(target[k], torch.Tensor) else target[k]
            assert el_as_list == ta_as_list, f"chunk[{k}] = {el_as_list} != {ta_as_list}"


if __name__ == '__main__':
    test_make_article_chunks_max_stride()
    test_expand_chunk()

