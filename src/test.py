import os

import numpy as np
import pandas as pd
from keras.applications import VGG19, EfficientNetV2S, efficientnet_v2, vgg19
from keras.models import Model
from keras.utils import img_to_array, load_img, plot_model
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from PIL import ImageFile
from tqdm import tqdm

from preprocess import get_nltk_resource, preprocess_text
from utils import get_df, sampling_df


def imageFeaturesProcessing(local_df: pd.DataFrame, sample_frac: float = 0):
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    # Setting image size and folder path
    img_size = (32, 32)
    img_folder_path = "./image/"

    # Pre-trained computer vision model (VGG16)
    vgg16_model = VGG19(weights='imagenet', include_top=False, input_shape=(img_size[0], img_size[1], 3))
    # effv2_model = EfficientNetV2S(weights='imagenet', include_top=False, input_shape=(img_size[0], img_size[1], 3))

    # Function to extract image features

    def extract_image_features(filename: str, model: Model):
        img = load_img(filename, target_size=img_size)
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        feat = model.predict(img, verbose=0)
        feat = feat.flatten()
        return feat

    img_list = os.listdir(img_folder_path)
    local_df = local_df if sample_frac else sampling_df(local_df, sample_frac)

    # Creating a new dataframe to store pre-processed image features of products
    new_product_df = pd.DataFrame(columns=['title', 'asin', 'description', 'image'])
    for file in tqdm(img_list):
        asin = os.path.splitext(file)[0]
        # Checking if the product id is present in the dataframe
        if asin in local_df['asin'].to_list():
            try:
                # Extracting image features and adding them to the new_product_df dataframe
                img = extract_image_features(img_folder_path + file, vgg16_model)
                # img = extract_image_features(img_folder_path+file, effv2_model)
                data = {'asin': asin, 'title': local_df[local_df['asin'] == asin]['title'].values[0],
                        'description': local_df[local_df['asin'] == asin]['title'].values[0], 'image': img}
                new_product_df = pd.concat([new_product_df, pd.DataFrame(data)], ignore_index=True)
                # metadata.loc[metadata['asin'] == 'asin', 'imageURL'] = asin+'.jpg'
            except Exception as e:
                print(e)

    # Returning the pre-processed product data
    # new_product_df.to_csv('./image_features.csv', index=False)
    return new_product_df


def preprocess_product_data(dataframe: pd.DataFrame, ncore: int):
    stemmer = WordNetLemmatizer()
    stop = stopwords.words("english")

    cols = ["title", "brand", "description"]
    text = preprocess_text(dataframe, cols, ncore, stop, stemmer)
    for col in cols:
        dataframe[col] = text[col]

    result = dataframe.dropna(subset=cols, how="all")[["asin"]].copy(deep=True)
    result["wordpool"] = dataframe[cols].fillna('').agg(' '.join, axis=1)
    result["image"] = np.nan

    return result


def main():
    ncore = os.cpu_count()
    product_data = get_df("./data/meta_AMAZON_FASHION.json.gz")
    user_data = get_df("./data/AMAZON_FASHION.json.gz")

    # get_nltk_resource()
    product_data = preprocess_product_data(product_data, ncore)
    product_data.to_csv("temp.csv", index=False)


if __name__ == "__main__":
    main()
