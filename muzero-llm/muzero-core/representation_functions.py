from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch import nn
from sentence_transformers import SentenceTransformer
import math
from constants import DEFAULT_EMBEDDER, SHORT2FULL_EMBEDDER_NAME
from tokenizer_util import myAutoTokenizer
from util import tn
import time

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:,:x.size(1), :].to(x.device)
        return self.dropout(x)


class SentenceBasedModelH(nn.Module):
    def __init__(self, model_name=DEFAULT_EMBEDDER, output_size=None, freeze=False, cache_dir=None, tokenizer_name='gpt2', codebook=None):
        super(SentenceBasedModelH, self).__init__()
        self.model = SentenceTransformer(model_name, cache_folder=cache_dir)
        self.tokenizer = myAutoTokenizer(tokenizer_name, trust_remote_code=False if (tokenizer_name == 'gpt2') else True)
        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        hidden_size = self.model.get_sentence_embedding_dimension()

        # Load English tokenizer, tagger, parser, NER and word vectors
        import spacy # Import here for lazy loading purposes
        self.nlp = spacy.load("en_core_web_sm", disable=["ner", "tagger", "lemmatizer"])

        # Initialize positional encoding
        self.pos_encoder = PositionalEncoding(hidden_size, dropout=0.0)
        
        # freeze the sentence transformer model
        # self.tune = tune
        self.freeze = freeze
        if self.freeze:
            for p in self.model.parameters():
                p.requires_grad = False

        if output_size is not None:
            self.projection = nn.Linear(hidden_size, output_size)
        else:
            self.projection = None

        self.m1_time = []
        self.m2_time = []

        self.codebook = codebook

    def forward(self, input_ids, attention_mask, skip_pooling=False):

        text = decode_with_tokenizer(self.tokenizer, input_ids, attention_mask)

        text, _ = check_empty(text)
        # Process whole documents in a vectorized fashion using Spacy's pipe method
        docs = list(self.nlp.pipe(text))

        # Split the text into separate sentences
        sentences = [[sent.text.strip() for sent in doc.sents] for doc in docs]

        # Embed each sentence into a single vector
        # Flatten the list of sentences and keep track of sentence lengths
        num_of_sentences = [len(s) for s in sentences]
        flat_sentences = [sent for sents in sentences for sent in sents]

        # Embed all sentences at once
        flat_embeddings = self.model.encode(flat_sentences)
        flat_embeddings_tensor = torch.tensor(flat_embeddings, device=self.model.device)

        # map the embeddings to the codebook by looking up the closest codebook entry
        if self.codebook is not None:
            flat_embeddings_tensor = torch.cdist(flat_embeddings_tensor, self.codebook)
            closest_codebook_indices = torch.argmin(flat_embeddings_tensor, dim=1)
            flat_embeddings_tensor = self.codebook[closest_codebook_indices]

        # Calculate the size of the padded tensor
        max_length = max(num_of_sentences)
        embedding_size = flat_embeddings.shape[1]
        num_docs = len(num_of_sentences)

        # Create the padded tensor
        padded_embeddings = torch.zeros((num_docs, max_length, embedding_size), device=self.model.device)

        # Create a mask to identify where embeddings should be placed
        mask = torch.arange(max_length).expand(len(num_of_sentences), max_length) < torch.tensor(num_of_sentences).unsqueeze(1)

        # Use advanced indexing to place embeddings
        padded_embeddings[mask] = flat_embeddings_tensor
        sentence_embeddings = padded_embeddings

        # Add position embeddings to the sentence representations
        sentence_embeddings = self.pos_encoder(sentence_embeddings)

        if self.projection is not None:
            sentence_embeddings = self.projection(sentence_embeddings)

        # Also return the number of sentence embeddings (i.e. length)
        return sentence_embeddings, torch.tensor(num_of_sentences).long().to(sentence_embeddings.device)
    
def decode_with_tokenizer(tokenizer, input_ids, attention_mask):
    # Decode the input into text
    copy_of_input_ids = input_ids.clone()
    # copy_of_input_ids[attention_mask == 0] = tokenizer.pad_token_id
    copy_of_input_ids = torch.where(attention_mask == 0, tokenizer.pad_token_id, copy_of_input_ids) # Doesn't error if attention_mask has only 1s.
    text = tokenizer.batch_decode(copy_of_input_ids, skip_special_tokens=True)
    return text

def check_empty(text):
    # TODO: figure out if empty strings actually cause an issue down the line. I got some NaNs at some point, but don't remember if it was caused by this.
    # some texts are empty, make them something so they get a represenation?
    new_text = []
    is_empty=False
    for t in text:
        if t == "":
            new_text.append("empty")
            is_empty=True
        else:
            new_text.append(t)
    return new_text,is_empty

class ModelH(nn.Module):
    def __init__(self, model_name='gpt2', use_self_attention=False, output_size = None, freeze=False, tokenizer_name='gpt2', cache_dir=None, default_tokenizer='gpt2', extra_transformer_layers = 0):
        super(ModelH, self).__init__()
        if "sentence-transformers" in model_name or model_name in SHORT2FULL_EMBEDDER_NAME:
            if model_name in SHORT2FULL_EMBEDDER_NAME:
                model_name = SHORT2FULL_EMBEDDER_NAME[model_name]
            self.model = SentenceTransformer(model_name, cache_folder=cache_dir)
            self.sentence_transformer = True
            hidden_size = self.model.get_sentence_embedding_dimension()
        else:
            self.sentence_transformer = False
            self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
            hidden_size = self.model.config.hidden_size if any(s in model_name for s in ["falcon","llama", "microsoft/phi-2"]) else self.model.config.n_embd

        if 'gpt2' not in model_name:
            self.tokenizer = myAutoTokenizer(model_name, trust_remote_code=True)
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        else:
            self.tokenizer = myAutoTokenizer(tokenizer_name, trust_remote_code=False)
        self.use_self_attention = use_self_attention

        if not 'gpt2' in model_name:
            self.need_retokenization = True
            self.default_tokenizer = myAutoTokenizer(default_tokenizer, trust_remote_code=True)
            self.default_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        else:
            self.need_retokenization = False
        
        # self.tune = tune
        # if self.tune > 0:
        #     assert 'gpt2' in model_name, "Tuning the top layers currently assumes GPT-2 as representation function"
        #
        #     # only tune the first 'tune' blocks
        #     total_blocks = len(self.model.transformer.h)
        #     for i, m in enumerate(self.model.transformer.h):
        #         if i < total_blocks - self.tune:
        #             for name, param in m.named_parameters():
        #                 param.requires_grad = False
        #                 #print(f"Freezing parameter: {name} in {m}")
        # else:
        self.freeze = True #freeze
        # if freeze:
        for p in self.model.parameters(): # Note that currently these parameters are used with torch.no_grad() anyway, so this is not strictly necessary
            p.requires_grad = False

        if self.use_self_attention:
            # Add multi-head self attention mechanism
            self.self_attention = nn.MultiheadAttention(hidden_size, num_heads=8)
            self.query_vector = nn.Parameter(torch.randn(hidden_size))

        if output_size is not None:
            self.projection = nn.Linear(hidden_size, output_size)
        else:
            self.projection = None

        # if we have extra transformer layers, we need to add them to the model
        if extra_transformer_layers > 0:
            self.extra_transformer_layers = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=hidden_size, nhead=8), num_layers=extra_transformer_layers)
        else:
            self.extra_transformer_layers = None

    def forward_sentence_transformer(self, input_ids, attention_mask):
        text = decode_with_tokenizer(self.default_tokenizer, input_ids, attention_mask)
        output_vector = self.model.encode(text, convert_to_tensor=True)
        return output_vector
        
    def forward_language_model(self, input_ids, attention_mask, skip_pooling):

        if self.need_retokenization:
            text = decode_with_tokenizer(self.default_tokenizer, input_ids, attention_mask)
            text, _ = check_empty(text)
            tokenized = self.tokenizer.batch_encode_plus(text, return_tensors='pt', padding=True, truncation=True)
            input_ids, attention_mask = tokenized.input_ids.to(input_ids.device), tokenized.attention_mask.to(input_ids.device)
        # Pass the tokenized texts and attention masks through the model
        # if self.tune == 0:
        if self.freeze:
            with torch.no_grad():
                outputs = self.model(input_ids, output_hidden_states=True)
        else:
            outputs = self.model(input_ids, output_hidden_states=True)

        # Extract the hidden state at the last layer from outputs
        outputs = outputs.hidden_states[-1]

        # add few extra layers of transformers
        if self.extra_transformer_layers is not None:
            outputs = self.extra_transformer_layers(outputs.transpose(0, 1), src_key_padding_mask=attention_mask.logical_not()).transpose(0, 1)

        # Apply the attention mask to the output
        masked_outputs = outputs * attention_mask.unsqueeze(-1)
        if not skip_pooling:
            output_vector = self.pool(attention_mask, masked_outputs)
        else:
            output_vector = masked_outputs

        return output_vector

    def forward(self, input_ids, attention_mask, skip_pooling=False):

        if self.sentence_transformer:
            output_vector = self.forward_sentence_transformer(input_ids, attention_mask)
        else:
            output_vector = self.forward_language_model(input_ids, attention_mask, skip_pooling)
                
        if self.projection is not None:
            output_vector = self.projection(output_vector)

        return output_vector

    def pool(self, attention_mask, masked_outputs):
        if self.use_self_attention:
            # Apply multi-head self attention mechanism
            query = self.query_vector.unsqueeze(0).unsqueeze(0)  # (1, 1, n_embd)
            keys = masked_outputs.transpose(0, 1)  # (seq_len, batch_size, n_embd)
            self_attention_mask = attention_mask.logical_not()  # (batch_size, seq_len)
            # Expand the query to match the batch_size of the keys
            query = query.expand(-1, keys.size(1), -1)  # (1, batch_size, n_embd)
            self_attention_outputs, _ = self.self_attention(query, keys, keys, key_padding_mask=self_attention_mask)
            output_vector = self_attention_outputs.squeeze(0).squeeze(0)  # (batch_size, n_embd)
        else:
            # Apply average pooling over the model outputs, with proper masking
            output_vector = (masked_outputs.sum(dim=1) / attention_mask.sum(dim=1).unsqueeze(-1))
        return output_vector


if __name__ == "__main__":
    model = ModelH()
    texts = ["Hello, world! How are you doing?", "I'm doing great! How about you? Are you too doing great?"]

    # Add padding token to the tokenizer
    model.tokenizer.pad_token = model.tokenizer.eos_token
    # Tokenize the input texts and create attention masks
    encoding = model.tokenizer.batch_encode_plus(texts, return_tensors='pt', padding=True, truncation=True)
    input_ids = encoding['input_ids'].to(model.model.device)
    attention_mask = encoding['attention_mask'].to(model.model.device)
    outputs = model(texts)
    print(outputs)


