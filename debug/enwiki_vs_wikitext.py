from util import Namespace
from create_oracle_codes import get_ds_with_oracle_codes
from tokenizer_util import myAutoTokenizer

base_model_name = 'gpt2'

t = myAutoTokenizer(base_model_name)
common_args = Namespace(
    max_seq_len=128,
    cluster_count=1024,
kmeans_cluster_debug=False,
    force_remake_chunks=False,
    base_model_name=base_model_name,
    # max_articles=28531,
    max_articles=50,
)
enwiki_args = Namespace(
    data_name='enwiki',
)
wikitext_args = Namespace(
    data_name='wikitext-103',
)
enwiki_ds = get_ds_with_oracle_codes(common_args | enwiki_args)
wikitext_ds = get_ds_with_oracle_codes(common_args | wikitext_args)

enwiki_batch = enwiki_ds[0]
wikitext_batch = wikitext_ds[0]

enwiki_text = t.decode(enwiki_batch['input_ids'])
wikitext_text = t.decode(wikitext_batch['input_ids'])

# Print both
print(f"enwiki: {enwiki_text}")
print(f"wikitext: {wikitext_text}")
