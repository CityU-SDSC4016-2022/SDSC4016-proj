from concurrent import futures
from itertools import repeat
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
from keras.applications import EfficientNetV2S
from keras.models import Model
from keras.utils import img_to_array, load_img
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from PIL import ImageFile
from tqdm import tqdm


def get_nltk_resource():
    """Download NLTK resources."""

    nltk.download("punkt")
    nltk.download("wordnet")
    nltk.download("stopwords")


def __split_stop_stem(title: str | list | float, stop: list[str], stemmer: SnowballStemmer | WordNetLemmatizer):
    """Split string, filter by stop words and stem words"""

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
    """Preprocess text data"""

    def _fn_poll(col: pd.Series):
        with futures.ProcessPoolExecutor(num_workers) as executor:
            return list(executor.map(__split_stop_stem, tqdm(col.to_numpy()), repeat(stop), repeat(stemmer)))
    return {col: _fn_poll(datafram[col]) for col in cols}


def preprocess_image(series: pd.Series, img_size: tuple[int, int], img_path: str):
    """Preprocess image data"""

    def get_img_feat(filename: Path, model:  Model) -> np.ndarray:
        img = load_img(filename.as_posix(), target_size=img_size)
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        feat = model.predict(img, verbose=0)
        feat = feat.flatten()
        return feat

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    model = EfficientNetV2S(weights='imagenet', include_top=False, input_shape=(img_size[0], img_size[1], 3))

    asin_list = series.to_numpy()
    feat_list = [np.NaN] * len(asin_list)

    path_list = [file for file in Path(img_path).iterdir()]
    name_list = [file.stem for file in path_list]
    join_list, asin_idx, name_idx = np.intersect1d(asin_list, name_list, return_indices=True)

    for index, _ in tqdm(np.ndenumerate(join_list), total=len(join_list)):
        feat_list[asin_idx[index]] = get_img_feat(path_list[name_idx[index]], model)

    return feat_list
