import copy

from pretorch_util import assign_visible_gpus
from dataset import get_kmeans_path_from_args

assign_visible_gpus()
import json, argparse
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import torch
from tqdm import tqdm
from Levenshtein import distance as levenshtein_distance
from rouge import Rouge
import os

import csv
import re

def clean_text(text):
    """
    Replace any pattern with <integer> with a space.

    :param text: The input text to be cleaned.
    :return: The cleaned text with <integer> patterns replaced by spaces.
    """
    # Replace <integer> patterns with a space
    cleaned_text = re.sub(r'<\d+>', ' ', text)
    
    # Decode the text twice to handle double-encoded sequences
    cleaned_text = bytes(cleaned_text, "utf-8").decode("unicode_escape").encode("latin1").decode("utf-8")
    
    return cleaned_text

def load_kmeans_model(path):
    import pickle
    with open(path, 'rb') as f:
        kmeans = pickle.load(f)
    return kmeans

def get_sequence_of_actions(text_str, embedder, nlp, kmeans):
    text_sentences = get_sentencized_text(text_str, nlp)
    
    if len(text_sentences) == 0:
        print("WOOOOOOP")
        print(text_str)
        print(text_sentences)
        raise ValueError("No sentences found in text")
    text_embeds = embedder.encode(text_sentences, convert_to_numpy=True)
    sequence_of_actions = kmeans.predict(text_embeds).tolist()
    return sequence_of_actions, text_sentences

def get_sentencized_text( text_str, nlp):
    return [sent.text for sent in nlp(text_str).sents]


def load_spacy():
    import spacy # lazy load cuz takes a few seconds which is annoying during quick debugging
        
    nlp = spacy.load("en_core_web_sm", disable=['tagger', 'parser', 'ner', 'lemmatizer'])
    nlp.add_pipe('sentencizer')
    return nlp

def get_sentence_embedder():
    return SentenceTransformer('all-mpnet-base-v2')

def obtain_codes_for_generations(texts, clean=False, token_limits = [-1], tokenizer = "gpt2", args=None, json_path=None):
    # Load necessary models
    nlp = load_spacy()
    embedder = get_sentence_embedder()
    # kmeans = load_kmeans_model('pickles/train_newenwiki_a285310_s42/kmeans_mpnetc1024.pkl')
    kmeans = load_kmeans_model(get_kmeans_path_from_args(args))

    if type(tokenizer) == str:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)

    og_texts = copy.deepcopy(texts)
    texts = [clean_text(t) if clean else t for t in texts]

    for t_dict in tqdm(texts, desc="Obtain actions"):
        t_dict['token_limit_to_action_sequence'] = {}
        context_tokens = t_dict['ctx_tokens']

        for t in token_limits:
            if 'token_limit_to_action_sequence' in t_dict and t in t_dict['token_limit_to_action_sequence']:
                continue
            # t is dict with 'pred_tokens', 'pred_text', 'ctx_tokens', 'ctx_text', 'true_tokens', 'true_text'])
            pred_tokens = t_dict['pred_tokens']
            true_tokens = t_dict['true_tokens']

            if t == -1:
                # take all the tokens
                pred_text = t_dict['pred_text']
                true_text = t_dict['true_text']
            else:
                pred_text = tokenizer.decode(pred_tokens[:t], skip_special_tokens=True)
                true_text = tokenizer.decode(true_tokens[:t], skip_special_tokens=True)
                #print("TRUE TEXT", true_tokens[:t])
            
            pred_sequence_of_actions, _ = get_sequence_of_actions(pred_text, embedder, nlp, kmeans)
            true_sequence_of_actions, _ = get_sequence_of_actions(true_text, embedder, nlp, kmeans)

            t_dict['token_limit_to_action_sequence'][t] = {
                'pred_sequence_of_actions': pred_sequence_of_actions,
                'true_sequence_of_actions': true_sequence_of_actions,
                'pred_text': pred_text,
                'true_text': true_text
            }
    if json_path is not None and (og_texts != texts):
        with open(json_path, 'w') as f:
            json.dump(texts, f)

    return texts


def import_json(filename):
    """
    Function to import a JSON file given a filename.
    
    Args:
    filename (str): The path to the JSON file to be imported.
    
    Returns:
    dict: The JSON data as a dictionary.
    """
    with open(filename, 'r') as file:
        data = json.load(file)
    return data

def parse_args():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument('--filename', type=str, required=True, help='The path to the JSON file to be imported.')
    parser.add_argument('--token_limits', type=int, nargs='+', default=[-1], help='List of token limits.')
    parser.add_argument('--metrics', type=str, nargs='+', required=True, help='List of metrics.')
    parser.add_argument('--tokenizer', type=str, choices=["gpt2","allenai/OLMo-1B"])
    return parser.parse_args()

def main():
    import argparse
    import sys
    sys.setrecursionlimit(10000) # for rouge, might have to adjust upward

    args = parse_args()
    data = import_json(args.filename)
    print(data[0].keys())

    texts_with_code = obtain_codes_for_generations(data, token_limits=args.token_limits, tokenizer=args.tokenizer)

    results = {t: {} for t in args.token_limits}
    for metric in args.metrics:
        for t in args.token_limits:
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

    print(results)

    # Store results in CSV file
    filename_without_ext = args.filename.rsplit('.', 1)[0].rsplit('/')[-1]
    csv_filename = f'startfullctx_results/{filename_without_ext}.csv'
    os.makedirs('startfullctx_results', exist_ok=True)

    with open(csv_filename, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        headers = ['Metric'] + [str(t) for t in args.token_limits]
        writer.writerow(headers)
        for metric in args.metrics:
            row = [metric] + [results[t][metric] for t in args.token_limits]
            writer.writerow(row)



def evaluate_metric_for_generations(pred_text, true_text, metric):
    if 'levenshtein' in metric:
        dist = levenshtein_distance(pred_text, true_text)
        if metric == "normalized_levenshtein":
            return dist / len(true_text)
        else:
            return dist

    elif "rouge" in metric:
        rouge = Rouge()
        return rouge.get_scores(pred_text, true_text)[0][metric]['f']
    else:
        raise ValueError(f"Metric {metric} not supported")

if __name__ == "__main__":
    main()

