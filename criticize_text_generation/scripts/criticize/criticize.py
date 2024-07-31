import sys
import os
import math
import json
import argparse
from pretorch_util import assign_visible_gpus
assign_visible_gpus()
import torch
import warnings
# ignore FutureWarning: Using `TRANSFORMERS_CACHE` is deprecated and will be removed in v5 of Transformers. Use `HF_HOME` instead.
# also ignore UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from format_generations import load_full_generations_from_json, predict_and_transform_json


def eval_latent_PPL_alt(model, data, criticised_field):
    total_words = 0
    total_loss = 0.
    transition_logits = model['transition_logits']
    stoi = model['vocabulary']
    for sample in data:
        section_names = sample[criticised_field]
        section_names = ['<bos>'] + section_names + ['<eos>']
        for prev_section_name, section_name in zip(section_names[:-1], section_names[1:]):
            if prev_section_name not in stoi or section_name not in stoi:
                print (f'WARNING: invalid section name in transition {prev_section_name} -> {section_name}!')
                continue
            id1 = stoi[prev_section_name]
            id2 = stoi[section_name]
            total_loss += transition_logits[id1, id2]
            total_words += 1

    return math.exp(-total_loss / total_words)


def main(args):
    # Load model
    model = torch.load(os.path.join(args.critic, 'critic.pt'))

    # Criticise text
    if args.reformat:
        generations = load_full_generations_from_json(args.input_file)
        if args.max_samples is not None:
            generations = generations[:args.max_samples]
        data = predict_and_transform_json(generations, args, clean=args.clean)
    else:
        data = json.load(open(args.input_file))
    latent_PPL = eval_latent_PPL_alt(model, data, args.criticised_field)
    print (f'Latent PPL: {latent_PPL}')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Compute Latent PPL.')
    parser.add_argument('--critic', type=str, required=True,
                        help='Folder containing critic.pt')
    parser.add_argument('--input_file', type=str, required=True,
                        help='Input json file containing predicted_section_names.')
    parser.add_argument('--criticised_field', type=str, default='predicted_section_names',
                        help='The field containing section names to be criticised.')
    parser.add_argument('--reformat', action='store_true')
    parser.add_argument('--clean', action="store_true", default=False)
    parser.add_argument('--max_samples', type=int, default=None, help='The maximum number of samples to process.')
    parser.add_argument('--kmeans_path', type=str, help='Path to the KMeans model.', default='kmeans_mpnetc1024_v0.0.pkl')
    args = parser.parse_args()
    main(args)
