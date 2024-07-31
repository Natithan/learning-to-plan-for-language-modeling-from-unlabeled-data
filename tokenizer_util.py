from transformers import AutoTokenizer


def myAutoTokenizer(*args, **kwargs):
    tokenizer = AutoTokenizer.from_pretrained(*args, **kwargs)
    if tokenizer.bos_token_id is None:
        assert tokenizer.eos_token_id is not None
        tokenizer.bos_token_id = tokenizer.eos_token_id
    return tokenizer