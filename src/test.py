import os

import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.utils import plot_model
from sklearn.model_selection import train_test_split

from data import preprocess_merge_data, preprocess_product_data, preprocess_user_data
from model import hybrid_model
from utils import get_df, get_image, sampling_df


def main():
    ncore = os.cpu_count()
    test_size = 0.2
    data_frac = 0.01
    emb_size = 32
    do_hybird = True
    epochs = 100
    batch_size = 32

    product_data = get_df("./data/meta_AMAZON_FASHION.json.gz")
    user_data = get_df("./data/AMAZON_FASHION.json.gz")

    # get_nltk_resource()
    product_data = sampling_df(product_data, data_frac)
    product_data = preprocess_product_data(product_data, ncore)
    # product_data.to_csv("product_data_temp.csv", index=False)

    user_data = preprocess_user_data(user_data, ncore)
    # user_data.to_csv("user_data_temp.csv", index=False)

    merge_data = preprocess_merge_data(product_data, user_data)
    merge_data.to_csv("temp.csv", index=False)

    train_data, test_data = train_test_split(merge_data, test_size=test_size)
    num_user = merge_data['asin'].to_numpy().size
    num_prod = np.unique(merge_data['asin']).size

    text_vec_size = len(merge_data["text"][0])
    img_vec_size = len(merge_data["image"][0])

    model = hybrid_model(num_user, num_prod, text_vec_size, img_vec_size, emb_size, do_hybird)
    plot_model(model, to_file='model_structure.png', show_shapes=True, show_layer_names=True)
    stooper = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=50)

    cols = ["reviewerID", "asin", "text", "image"]
    train_set = [np.stack(train_data[col], 0) for col in cols]
    test_set = [np.stack(test_data[col], 0) for col in cols]

    history = model.fit(train_set, train_data["overall"], validation_split=0.1, epochs=epochs, batch_size=batch_size,
                        verbose=2, shuffle=True, use_multiprocessing=True, workers=ncore, callbacks=[stooper])
    predict = model.predict(test_set, verbose=0, use_multiprocessing=True, workers=ncore)


if __name__ == "__main__":
    main()
