from concurrent import futures
from itertools import repeat
# import numpy as np

import pandas as pd
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import nltk


def get_nltk_resource():
    nltk.download("punkt")
    nltk.download("wordnet")
    nltk.download("stopwords")


def __split_stop_stem(title: str | list | float, stop: list[str], stemmer: SnowballStemmer | WordNetLemmatizer):
    if isinstance(title, float):
        return title
    if isinstance(title, list):
        title = " ".join(title)

    title = word_tokenize(title.lower())

    if isinstance(stemmer, SnowballStemmer):
        return " ".join([stemmer.stem(word) for word in title if word not in (stop) and word.isalpha()])
    if isinstance(stemmer, WordNetLemmatizer):
        return " ".join([stemmer.lemmatize(word) for word in title if word not in (stop) and word.isalpha()])


def preprocess_text(datafram: pd.DataFrame, cols: list[str], num_workers: int, stop: list[str], stemmer: SnowballStemmer | WordNetLemmatizer):
    def _fn_poll(col: pd.Series):
        with futures.ProcessPoolExecutor(num_workers) as executor:
            return list(executor.map(__split_stop_stem, tqdm(col.to_numpy()), repeat(stop), repeat(stemmer)))
    return {col: _fn_poll(datafram[col]) for col in cols}
