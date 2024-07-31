from constants import DEFAULT_PICKLE_DIR
from os.path import join as jn
import pickle
from functools import partial
from tqdm import tqdm as std_tqdm
tqdm = partial(std_tqdm, dynamic_ncols=True)

kmeans_path = jn(DEFAULT_PICKLE_DIR,'train_newenwiki_a285310_s42/kmeans_mpnetc1024.pkl')
coded_sents_path = jn(DEFAULT_PICKLE_DIR, 'train_newenwiki_a285310_s42/sentencized.pkl')
with open(kmeans_path, 'rb') as f:
    kmeans = pickle.load(f)
with open(coded_sents_path, 'rb') as f:
    coded_sents = pickle.load(f)

flat_sents = [sent for article in coded_sents for sent in article['sentences']]
labels = kmeans.labels_
unique_labels_ordered_by_frequency = sorted(set(labels), key=lambda x: -list(labels).count(x))
most_frequent_label = unique_labels_ordered_by_frequency[-1]
sents = [s for s, l in zip(flat_sents, labels) if l == most_frequent_label]

second_most_frequent_label = unique_labels_ordered_by_frequency[-2]
sents2 = [s for s, l in zip(flat_sents, labels) if l == second_most_frequent_label]

least_frequent_label = unique_labels_ordered_by_frequency[0]
sents3 = [s for s, l in zip(flat_sents, labels) if l == least_frequent_label]

all_sents = {label: [s for s, l in zip(flat_sents, labels) if l == label] for label in tqdm(unique_labels_ordered_by_frequency)}
