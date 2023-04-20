import pandas as pd
from keras import backend
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler

from preprocess import preprocess_image, preprocess_text


def preprocess_product_data(dataframe: pd.DataFrame, ncore: int, img_size: tuple[int, int], img_path: str):
    """Preprocess product data"""

    stemmer = WordNetLemmatizer()
    stop = stopwords.words("english")

    cols = ["title", "brand", "description"]
    dataframe.dropna(subset=cols, how="all", inplace=True)

    stem = preprocess_text(dataframe, cols, ncore, stop, stemmer)
    feat = preprocess_image(dataframe["asin"], img_size, img_path)

    for col in cols:
        dataframe[col] = stem[col]

    result = dataframe[["asin"]].copy(deep=True)
    result["wordpool"] = dataframe[cols].fillna("").agg(" ".join, axis=1)
    result["image"] = feat

    result.dropna(subset="image", inplace=True)
    result["image"] = StandardScaler().fit_transform(result["image"].to_list()).tolist()

    backend.clear_session()
    return result


def preprocess_user_data(dataframe: pd.DataFrame, ncore: int):
    """Preprocess user data"""

    stemmer = WordNetLemmatizer()
    stop = stopwords.words("english")

    cols = ["reviewText", "summary"]
    dataframe = dataframe.groupby("asin").filter(lambda x: len(x) >= 5)
    dataframe = dataframe.groupby("reviewerID").filter(lambda x: len(x) >= 2)
    dataframe.dropna(subset=cols, how="all", inplace=True)
    dataframe.sort_values(by=["unixReviewTime"], inplace=True)

    # rate_under = dataframe[dataframe['overall'] != 5]
    # rate_length = round(rate_under.groupby("overall").count()["asin"].mean())
    # rate_five = dataframe[dataframe['overall'] == 5].sample(n=rate_length)
    # dataframe = pd.concat([rate_under, rate_five], ignore_index=True)

    stem = preprocess_text(dataframe, cols, ncore, stop, stemmer)
    for col in cols:
        dataframe[col] = stem[col]

    result = dataframe[["asin", "reviewerID", "overall"]].copy(deep=True)
    result["reviewpool"] = dataframe[cols].fillna("").agg(" ".join, axis=1)

    return result


def preprocess_merge_data(product_data: pd.DataFrame, user_data: pd.DataFrame):
    """Preprocess merged data"""

    merge = pd.merge(user_data, product_data, on='asin', how="inner")

    cols = ["reviewpool", "wordpool"]
    merge["text"] = merge[cols].fillna("").agg(" ".join, axis=1)
    merge.drop(columns=cols, inplace=True)

    merge["text"] = list(TfidfVectorizer().fit_transform(merge["text"]).toarray())
    # merge["text"] = list(TfidfVectorizer(max_features=64).fit_transform(merge["text"]).toarray())

    merge['asin'] = LabelEncoder().fit_transform(merge['asin'])
    merge['reviewerID'] = LabelEncoder().fit_transform(merge['reviewerID'])

    return merge
