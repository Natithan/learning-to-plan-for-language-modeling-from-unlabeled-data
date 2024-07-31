import json, argparse
from sentence_transformers import SentenceTransformer

import re

from dataset import get_kmeans_path_from_args


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

def load_full_generations_from_json(json_path):
    with open(json_path, 'r') as f:
        generations = json.load(f)
    if type(generations[0]) == str:
        return generations
    else:
        return [g['pred_text'] for g in generations]

def main():
    parser = argparse.ArgumentParser(description="Process a CSV file.")
    parser.add_argument('filename', type=str, help='The path to the CSV file.')
    parser.add_argument('--clean', action="store_true", default=False)
    parser.add_argument('--max_samples', type=int, default=None, help='The maximum number of samples to process.')
    args = parser.parse_args()

    # Load and process the data
    generations = load_full_generations_from_json(args.filename)
    if args.max_samples is not None:
        generations = generations[:args.max_samples]
    
    samples = predict_and_transform_json(generations, args, clean=args.clean)
    save_to_json(samples, 'tmp-output.json')

def save_to_json(data, output_file):
    """
    Save the given data to a JSON file.

    :param data: The data to be saved.
    :param output_file: The file to which the JSON data will be written.
    """
    with open(output_file, 'w') as f:
        json.dump(data, f)
    print(f"Data successfully saved to {output_file}")


def load_kmeans_model(path):
    import pickle
    with open(path, 'rb') as f:
        kmeans = pickle.load(f)
    return kmeans

def get_sequence_of_actions(text_str, embedder, nlp, kmeans):
    text_sentences = get_sentencized_text(text_str, nlp)
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

def predict_and_transform_json(texts, args, clean=False):
    # Load necessary models
    nlp = load_spacy()
    embedder = get_sentence_embedder()
    kmeans = load_kmeans_model(get_kmeans_path_from_args(args))


    texts = [clean_text(t) if clean else t for t in texts]
    samples = []

    from tqdm import tqdm

    for text in tqdm(texts, desc="Processing texts"):
        section_names, sections = get_sequence_of_actions(text, embedder, nlp, kmeans)
        section_names = [str(x) for x in section_names]
        sample = {'section_names': section_names, 'sections': sections, "predicted_section_names": section_names}
        samples.append(sample)

    return samples

if __name__ == "__main__":
    main()
