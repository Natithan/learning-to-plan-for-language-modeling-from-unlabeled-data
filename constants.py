from os.path import join as jn

DATA_ROOT_DIR = "SPECIFY_ME"


CODE_ROOT_DIR = "SPECIFY_ME"

CODE_DIR = jn(CODE_ROOT_DIR, "muLM") # change if needed
SF_CACHE_DIR = jn(CODE_ROOT_DIR, 'hf-cache') # change the cache dir for sentence transformer library if needed
WIKI_DIR = jn(DATA_ROOT_DIR, 'wikipedia')
EN_WIKI_FILE = jn(WIKI_DIR, 'enwiki-latest.json.gz') 
EN_WIKI_ARTICLE_COUNT = 5866389 # Takes a few minutes to compute. To compute: sum(1 for _ in open(EN_WIKI_FILE, 'r'))
NEW_EN_WIKI_ARTICLE_COUNT = 6458670 # load_dataset("wikipedia", "20220301.en").shape['train'][0]
DOLMA_ARTICLE_COUNT = 13095416 # load_dataset("allenai/dolma", "v1_6-sample").shape['train'][0]
DEFAULT_PICKLE_DIR =jn(CODE_DIR, 'pickles') # change where to store the preprocessed data if needed
DEFAULT_LLM = "meta-llama/Llama-2-7b-hf"
DEFAULT_CKPT_DIR = jn(CODE_DIR, 'checkpoints')
OTHER_CKPT_DIR = "SPECIFY_ME" # if default ckpt dir doesn't contain the checkpoint

SHORT2FULL_EMBEDDER_NAME = { # See https://huggingface.co/spaces/mteb/leaderboard
    'mpnet': 'sentence-transformers/all-mpnet-base-v2',
    'bge_base': 'BAAI/bge-base-en-v1.5'
}
FULL2SHORT_EMBEDDER_NAME = {v: k for k, v in SHORT2FULL_EMBEDDER_NAME.items()}
DEFAULT_EMBEDDER = 'sentence-transformers/all-mpnet-base-v2'
FIXED_CODE = 0
EOT_STRING = "<|endoftext|>"