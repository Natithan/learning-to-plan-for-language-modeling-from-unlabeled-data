from constants import DEFAULT_PICKLE_DIR
from os.path import join as jn
import matplotlib; matplotlib.use('TkAgg')
import pickle
kmeans_path = jn(DEFAULT_PICKLE_DIR,'train_enwiki_a28531_s42','kmeans_c64.pkl')
with open(kmeans_path, 'rb') as f:
    kmeans = pickle.load(f)
labels = kmeans.labels_
# plot label frequencies, ordered by frequency
import matplotlib.pyplot as plt
# plt.rcParams['figure.figsize'] = (15, 5)
import numpy as np
from collections import Counter
label_counts = Counter(labels)
label_counts = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
labels, counts = zip(*label_counts)
plt.bar(np.arange(len(labels)), counts)
plt.show()
print(5)