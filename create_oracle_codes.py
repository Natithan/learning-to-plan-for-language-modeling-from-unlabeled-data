import jsonargparse

from constants import EN_WIKI_ARTICLE_COUNT, DEFAULT_PICKLE_DIR
from pretorch_util import assign_visible_gpus; assign_visible_gpus()
from dataset import WikiText103Dataset, FullWikiDataset, NewFullWikiDataset, IterableWrapper, DolmaDataset
import warnings; warnings.filterwarnings("ignore", category=UserWarning, message=".*pydantic*.")

def process_article_batch(article_batch):
    global sbert
    titles = []
    texts = []
    sentence_embds = []
    for article in article_batch:
        title, sents = article['title'], article['sentences']
        sentence_embeddings = sbert.encode(sents)
        # print(f"Processed {title}")
        titles.append(title)
        texts.append(sents)
        sentence_embds.append(sentence_embeddings)
    return titles, texts, sentence_embds


def main():
    args = get_args()
    get_ds_with_oracle_codes(args)


def get_ds_with_oracle_codes(args):
    DatasetClass = {
        "wikitext-103": WikiText103Dataset,
        "enwiki": FullWikiDataset,
        "newenwiki": NewFullWikiDataset,
        "dolma": DolmaDataset
    }[args.data_name]
    dataset = DatasetClass(**args)
    if args.stream:
        dataset = IterableWrapper(dataset)
    return dataset


def get_args():
    parser = jsonargparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--max_articles", type=int, default=EN_WIKI_ARTICLE_COUNT // 1000,
                        help="Number of articles to sample from Wikipedia. If negative, use all articles.")
    parser.add_argument("--cluster_count", type=int, default=1024)
    parser.add_argument("--out_dir", type=str, default=DEFAULT_PICKLE_DIR)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()