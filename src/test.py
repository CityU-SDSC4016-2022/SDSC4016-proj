import os

import numpy as np
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

from data import preprocess_merge_data, preprocess_product_data, preprocess_user_data
from model import hybrid_model
from utils import get_df, print_struct, sampling_df, save_history, res_evalution, save_result, save_full_result
# from utils import get_image
# from preprocess import get_nltk_resource


def main():
    """Main function."""

    # Arguments
    ncore = os.cpu_count()  # Number of cores to be used for parallel processing
    data_frac = 1           # Data fraction of the dataset to be used for training
    test_size = 0.2         # Data fraction of the dataset to be used for testing
    img_size = (32, 32)     # Image size to be used for preprocessing
    img_path = "./image"    # Image path to be used for preprocessing
    do_hybird = True        # Set to True to use Hybird model
    emb_size = 32           # Embedding size of the model
    epochs = 5000           # Number of epochs to train model
    batch_size = 128        # Batch size of training model
    print_model = False     # Save model structure as image
    csv_par = []            # Parameters for the csv file name

    # Load data from json.gz file
    prod_data_path = "./data/meta_AMAZON_FASHION.json.gz"
    user_data_path = "./data/AMAZON_FASHION.json.gz"
    product_data = get_df(prod_data_path)
    user_data = get_df(user_data_path)

    # Fetch images from Amazon database
    # uid_col = "asin"
    # url_col = ["imageURLHighRes", "imageURL"]
    # get_image(img_path, prod_data_path, url_col, uid_col)

    # Download nltk library resource if not available
    # get_nltk_resource()

    # Preprocessing text and image for product
    # product_data = sampling_df(product_data, data_frac)
    product_data = preprocess_product_data(product_data, ncore, img_size, img_path)

    # Preprocessing text and image for user
    user_data = preprocess_user_data(user_data, ncore)

    # Merging two dataframes
    merge_data = preprocess_merge_data(product_data, user_data)

    # Splitting data into train and test sets
    train_data, test_data = train_test_split(merge_data, test_size=test_size)
    num_user = merge_data['asin'].to_numpy().size
    num_prod = np.unique(merge_data['asin']).size
    text_vec_size = len(merge_data["text"][0])
    img_vec_size = len(merge_data["image"][0])

    # Building model architecture
    model = hybrid_model(num_user, num_prod, text_vec_size, img_vec_size, emb_size, do_hybird)
    print_struct(print_model, model, 'model_structure.png')
    stooper = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=100)

    # Spliting data into X and Y
    cols = ["reviewerID", "asin", "text", "image"]
    train_x = [np.stack(train_data[col], 0) for col in cols]
    test_x = [np.stack(test_data[col], 0) for col in cols]
    train_y = train_data["overall"]
    test_y = test_data["overall"]

    # Training model
    history = model.fit(train_x, train_y, validation_split=0.2, epochs=epochs, batch_size=batch_size,
                        verbose=0, shuffle=True, use_multiprocessing=True, workers=ncore, callbacks=[stooper])

    # Testing model
    predict = model.predict(test_x, verbose=0, use_multiprocessing=True, workers=ncore)

    # Saving results and evaluation
    save_history(history, "history", csv_par)
    save_result(predict, test_y, "result", csv_par)
    save_full_result(predict, test_data, "result_full", csv_par)
    res_evalution(predict, test_y)


if __name__ == "__main__":
    main()
