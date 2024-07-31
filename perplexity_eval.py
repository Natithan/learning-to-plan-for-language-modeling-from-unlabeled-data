import pickle
import torch
from tokenizer_util import myAutoTokenizer

from modelling_llama_code_conditioned import CodeConditionedLlama


def main():
    model_name = "meta-llama/Llama-2-7b-hf"
    kmeans_path = 'kmeans_5866--1024.pkl'
    embeds_path = "wiki_sbert_embeddings_5866.pkl"
    kmeans = pickle.load(open(kmeans_path, 'rb'))
    embeds = pickle.load(open(embeds_path, 'rb'))
    title = list(embeds.keys())[0]
    article = embeds[title]
    sentences, embeddings, codes = article['sentences'], article['embeddings'], article['codes']
    model = CodeConditionedLlama.from_pretrained(model_name, kmeans.cluster_centers_)
    tokenizer = myAutoTokenizer(model_name)
    max_length = 1024
    for i, (sentence, embedding, code) in enumerate(zip(sentences, embeddings, codes)):
        text_ids = tokenizer.encode(sentence, return_tensors="pt")
        if len(text_ids[0]) > max_length:
            continue
        outputs = model(text_ids, labels=text_ids, code_idx=torch.tensor([code]), return_dict=True)
        loss = outputs.loss
        perplexity = torch.exp(loss)
        print(perplexity.item())


if __name__ == '__main__':
    main()
