
import json

def print_json_keys(file_path):
    """
    Load a JSON file and print its keys.

    :param file_path: Path to the JSON file.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    if isinstance(data, dict):
        keys = data.keys()
    elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        keys = data[0].keys()
    else:
        print("Unsupported JSON format.")
        return
    
    print("Keys in the JSON file:")
    for key in keys:
        print(key)

def print_json_keys_and_first_values(file_path):
    """
    Load a JSON file, print its keys, and print the values for each key in the first item if it's a list of dictionaries.

    :param file_path: Path to the JSON file.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    if isinstance(data, dict):
        keys = data.keys()
        print("Keys in the JSON file:")
        for key in keys:
            print(key)
    elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        keys = data[0].keys()
        print("Keys in the JSON file:")
        for key in keys:
            print(key)
        
        print("Values for the first entry in the JSON file:")
        for key, value in data[0].items():
            print(f"{key}: {value}")
    else:
        print("Unsupported JSON format.")
        return

def transform_json(input_file, output_file, subsample=-1):
    """
    Transform the JSON file by renaming 'sentences' to 'sections' and 'codes' to 'section_names',
    and discarding other fields.

    :param input_file: Path to the input JSON file.
    :param output_file: Path to the output JSON file.
    """
    with open(input_file, 'r') as infile:
        data = json.load(infile)
    
    if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
        print("Unsupported JSON format.")
        return
    
    if subsample > -1:
        import random
        data = random.sample(data, min(subsample, len(data)))

    transformed_data = []
    for item in data:
        transformed_item = {}
        if 'sentences' in item:
            transformed_item['sections'] = item['sentences']
        else:
            raise KeyError("Key 'sentences' not found in item.")
        
        if 'codes' in item:
            transformed_item['section_names'] = [str(code) for code in item['codes']]
        else:
            raise KeyError("Key 'codes' not found in item.")
        transformed_data.append(transformed_item)
    
    with open(output_file, 'w') as outfile:
        json.dump(transformed_data, outfile)


# Example usage
print("VAL")
transform_json('codes_data/val/coded_GPT2TokenizerFast-tokenized_arts.json',
               'val_coded.json')
print("TRAIN")
transform_json('codes_data/train/coded_GPT2TokenizerFast-tokenized_arts.json',
               'train_coded.json', subsample=-1)
print("TEST")
transform_json('codes_data/test/coded_GPT2TokenizerFast-tokenized_arts.json',
               'test_coded.json')

