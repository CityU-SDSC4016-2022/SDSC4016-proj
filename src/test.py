import os

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from preprocess import preprocess_text, preprocess_image
from utils import get_image, get_df, sampling_df


def preprocess_product_data(dataframe: pd.DataFrame, ncore: int):
    stemmer = WordNetLemmatizer()
    stop = stopwords.words("english")
    img_size = (32, 32)
    img_path = "./image"

    cols = ["title", "brand", "description"]
    dataframe.dropna(subset=cols, how="all", inplace=True)

    stem = preprocess_text(dataframe, cols, ncore, stop, stemmer)
    feat = preprocess_image(dataframe["asin"], img_size, img_path)

    for col in cols:
        dataframe[col] = stem[col]

    result = dataframe[["asin"]].copy(deep=True)
    result["wordpool"] = dataframe[cols].fillna('').agg(' '.join, axis=1)
    result["image"] = feat

    result.dropna(subset="image", inplace=True)
    # result["wordpool"] = TfidfVectorizer().fit_transform(result['wordpool']).toarray().tolist()
    result["image"] = StandardScaler().fit_transform(result["image"].to_list()).tolist()

    return result


def preprocess_user_data(dataframe: pd.DataFrame, ncore: int):
    stemmer = WordNetLemmatizer()
    stop = stopwords.words("english")

    cols = ["reviewText", "summary"]
    dataframe = dataframe.groupby("asin").filter(lambda x: len(x) >= 5)
    dataframe = dataframe.groupby("reviewerID").filter(lambda x: len(x) >= 2)
    result = dataframe.dropna(subset=cols, how="all").sort_values(by=['unixReviewTime'])

    stem = preprocess_text(result, cols, ncore, stop, stemmer)
    for col in cols:
        dataframe[col] = stem[col]

    result = result[["asin"]].copy(deep=True)
    result["reviewpool"] = dataframe[cols].fillna('').agg(' '.join, axis=1)

    return result


def main():
    ncore = os.cpu_count()
    product_data = get_df("./data/meta_AMAZON_FASHION.json.gz")
    # user_data = get_df("./data/AMAZON_FASHION.json.gz")

    # get_nltk_resource()
    # product_data = sampling_df(product_data, 0.02)
    product_data = preprocess_product_data(product_data, ncore)
    product_data.to_csv("product_data.csv", index=False)

    # user_data = preprocess_user_data(user_data, ncore)
    # user_data.to_csv("user_data.csv", index=False)


if __name__ == "__main__":
    main()
