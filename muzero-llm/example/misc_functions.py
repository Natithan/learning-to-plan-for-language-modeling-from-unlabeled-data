import torch
from sentence_transformers import SentenceTransformer

def embed_strings(strings, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(strings)
    return torch.tensor(embeddings)
