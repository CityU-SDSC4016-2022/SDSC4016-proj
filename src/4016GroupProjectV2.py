# Import the necessary libraries
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.applications import VGG16, EfficientNetV2S
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import (BatchNormalization, Concatenate, Dense, Dot, Dropout,
                          Embedding, Input, Reshape, StringLookup)
from keras.models import Model
from keras.utils import img_to_array, load_img, plot_model
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from PIL import ImageFile
from sentence_transformers import SentenceTransformer
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

import utils

product_data = utils.get_df('./data/meta_AMAZON_FASHION.json.gz')
user_data = utils.get_df('./data/AMAZON_FASHION.json.gz')


print('--------------------------------product_data----------------------------------------', '\n')
print(product_data.info(), '\n')
print('----------------------------------user_data-----------------------------------------', '\n')
print(user_data.info(), '\n')

# Image feature extraction package


def imageFeaturesProcessing(product_data: pd.DataFrame):
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    # Setting image size and folder path
    img_size = (32, 32)
    img_folder_path = "./image/"

    # Pre-trained computer vision model (VGG16)
    vgg16_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_size[0], img_size[1], 3))
    # effv2_model = EfficientNetV2S(weights='imagenet', include_top=False, input_shape=(img_size[0], img_size[1], 3))

    # Function to extract image features
    def extract_image_features(filename: str, model: Model):
        img = load_img(filename, target_size=img_size)
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = model.predict(img, verbose=0)
        img = img.flatten()
        return img

    # Creating a new dataframe to store pre-processed image features of products
    new_product_df = pd.DataFrame(columns=['title', 'asin', 'description', 'image'])
    for file in os.listdir(img_folder_path):
        asin = file.split('.')[0]
        # Checking if the product id is present in the product_data dataframe
        if asin in list(product_data['asin']):
            try:
                # Extracting image features and adding them to the new_product_df dataframe
                img = extract_image_features(img_folder_path + file, vgg16_model)
                # img = extract_image_features(img_folder_path+file, effv2_model)
                data = {'asin': asin, 'title': product_data[product_data['asin'] == asin]['title'].values[0],
                        'description': product_data[product_data['asin'] == asin]['title'].values[0], 'image': img}
                new_product_df = pd.concat([new_product_df, pd.DataFrame(data)], ignore_index=True)
                # metadata.loc[metadata['asin'] == 'asin', 'imageURL'] = asin+'.jpg'
            except Exception as e:
                print(e)
    # Returning the pre-processed product data
    return new_product_df


product_data = imageFeaturesProcessing(product_data)
print('----------------------------precrocessed_product_data-----------------------------------', '\n')
print(product_data.info(), '\n')


# %%
def dfProcessing(user_data: pd.DataFrame, product_data: pd.DataFrame):
    # Define a list of stop words to remove from the text
    stop = stopwords.words('english')

    # Load the SentenceTransformer model for text embeddings
    # SbertModel = SentenceTransformer('sentence-transformers/all-minilm-l6-v2')

    # Remove rows with missing values in the "title" column from the product data
    product_data = product_data.dropna(subset=['title'])
    print(len(product_data))
    # Keep only relevant columns from the user data
    user_data = user_data[['overall', 'reviewerID', 'asin', 'reviewText', 'unixReviewTime']]

    # Keep only rows in the user data where the "asin" value matches one in the product data
    user_data = user_data[user_data['asin'].isin(list(product_data['asin']))]

    user_data = user_data.groupby("asin").filter(lambda x: x['overall'].count() >= 5)
    user_data = user_data.groupby("reviewerID").filter(lambda x: x['overall'].count() >= 2)
    rating_no_5 = user_data[user_data['overall'] != 5]
    rating_5 = user_data[user_data['overall'] == 5].sample(n=len(user_data[user_data['overall'] == 3]), random_state=1)

    user_data = pd.concat([rating_no_5, rating_5])
    # Sort the user data by review time
    user_data = user_data.sort_values(by=['unixReviewTime'])
    # Merge the user and product data on the "asin" column
    merge_df = pd.merge(user_data, product_data, on='asin', how="left")

    print(merge_df.isna().sum())
    print(len(merge_df))
    # Encode the reviewer ID and product ID columns using LabelEncoder
    user_encoder = LabelEncoder()
    user_ids = user_encoder.fit_transform(merge_df['reviewerID'])
    merge_df['reviewerID'] = user_ids
    product_encoder = LabelEncoder()
    product_ids = product_encoder.fit_transform(merge_df['asin'])
    merge_df['asin'] = product_ids

    # Get the unique product IDs
    unique_product_ids = np.unique(product_ids)

    # Define a Porter stemmer for text processing
    ps = PorterStemmer()
    # Preprocess the "title" column by removing stop words and applying stemming
    merge_df['title'] = merge_df['title'].apply(lambda x: ' '.join([word for word in str(x).split() if word not in (stop)]))
    merge_df['title'] = merge_df['title'].apply(lambda x: ps.stem(x))

    # Get text embeddings for the preprocessed titles
    text_embeddings = TfidfVectorizer(max_features=64).fit_transform(merge_df['title']).toarray()
    # text_embeddings = SbertModel.encode(merge_df['title'])

    # Scale and normalize the image embeddings
    image_embeddings = StandardScaler().fit_transform(merge_df['image'].to_list())
    # Get the overall ratings for each product
    ratings = merge_df['overall']
    # Split the data into training and validation sets using train_test_split
    train_user_ids, val_user_ids, train_product_ids, val_product_ids, train_tfidf_vectors, val_tfidf_vectors, train_images, val_images, train_ratings, val_ratings = train_test_split(user_ids, product_ids, text_embeddings,
                                                                                                                                                                                      image_embeddings, ratings, test_size=0.2,
                                                                                                                                                                                      random_state=42)

    return merge_df, user_ids, unique_product_ids, train_user_ids, val_user_ids, train_product_ids, val_product_ids, train_tfidf_vectors, val_tfidf_vectors, train_images, val_images, train_ratings, val_ratings


def hybirdModel(user_ids, unique_product_ids, do_hybird):
    # Setting the size of embeddings
    embeddings_size = 32
    # Getting the number of unique users and products
    usr, prd = user_ids.shape[0], unique_product_ids.shape[0]

    # Defining input layers
    x_users_in = Input(name="User_input", shape=(1,))
    x_products_in = Input(name="Product_input", shape=(1,))

    # A) Matrix Factorization
    # Embeddings and Reshape layers for user ids
    cf_xusers_emb = Embedding(name="MF_User_Embedding", input_dim=usr, output_dim=embeddings_size)(x_users_in)
    cf_xusers = Reshape(name='MF_User_Reshape', target_shape=(embeddings_size,))(cf_xusers_emb)
    # Embeddings and Reshape layers for product ids
    cf_xproducts_emb = Embedding(name="MF_Product_Embedding", input_dim=prd, output_dim=embeddings_size)(x_products_in)
    cf_xproducts = Reshape(name='MF_Product_Reshape', target_shape=(embeddings_size,))(cf_xproducts_emb)
    # Dot product layer
    cf_xx = Dot(name='MF_Dot', normalize=True, axes=1)([cf_xusers, cf_xproducts])

    # B) Neural Network
    # Embeddings and Reshape layers for user ids
    nn_xusers_emb = Embedding(name="NN_User_Embedding", input_dim=usr, output_dim=embeddings_size)(x_users_in)
    nn_xusers = Reshape(name='NN_User_Reshape', target_shape=(embeddings_size,))(nn_xusers_emb)
    # Embeddings and Reshape layers for product ids
    nn_xproducts_emb = Embedding(name="NN_Product_Embedding", input_dim=prd, output_dim=embeddings_size)(x_products_in)
    nn_xproducts = Reshape(name='NN_Product_Reshape', target_shape=(embeddings_size,))(nn_xproducts_emb)
    # Concatenate and dense layers
    nn_xx = Concatenate()([nn_xusers, nn_xproducts])
    nn_xx = Dense(name="NN_layer", units=16, activation='relu')(nn_xx)
    nn_xx = Dropout(0.1)(nn_xx)

    # If do_hybrid is True, add text-based and image-based models
    if do_hybird:
       ######################### TEXT BASED ############################
        text_in = Input(name="title_input", shape=(64,))
        text_x = Dense(name="title_layer", units=64, activation='relu')(text_in)

    ######################## IMAGE BASED ###########################
        image_in = Input(name="image_input", shape=(512,))
        image_x = Dense(name="image_layer", units=256, activation='relu')(image_in)

        content_xx = Concatenate()([text_x, image_x])
        content_xx = Dense(name="contect_layer", units=128, activation='relu')(content_xx)
        # Merge all
        y_out = Concatenate()([cf_xx, nn_xx, content_xx])
    else:
        y_out = Concatenate()([cf_xx, nn_xx])

    y_out = Dense(name="CF_contect_layer", units=64, activation='linear')(y_out)
    y_out = Dense(name="y_output", units=1, activation='linear')(y_out)

    ########################## OUTPUT ##################################
    # Compile
    if do_hybird == True:
        model = Model(inputs=[x_users_in, x_products_in, text_in, image_in], outputs=y_out, name="Hybrid_Model")
    else:
        model = Model(inputs=[x_users_in, x_products_in], outputs=y_out, name="Hybrid_Model")
    from keras import optimizers
    adam = optimizers.Adam(lr=0.01, decay=0.0001)
    model.compile(optimizer=adam, loss='mae', metrics=[tf.keras.metrics.RootMeanSquaredError(), tf.keras.metrics.mean_absolute_error])

    return model


'''
Plot loss and metrics of keras training.
'''


def utils_plot_keras_training(training):
    metrics = [k for k in training.history.keys() if ("loss" not in k) and ("val" not in k)]
    fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(15, 3))

    # training
    ax[0].set(title="Training")
    ax11 = ax[0].twinx()
    ax[0].plot(training.history['loss'], color='black')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss', color='black')
    for metric in metrics:
        ax11.plot(training.history[metric], label=metric)
    ax11.set_ylabel("Score", color='steelblue')
    ax11.legend()

    # validation
    ax[1].set(title="Validation")
    ax22 = ax[1].twinx()
    ax[1].plot(training.history['val_loss'], color='black')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Loss', color='black')
    for metric in metrics:
        ax22.plot(training.history['val_'+metric], label=metric)
    ax22.set_ylabel("Score", color="steelblue")
    # plt.show()
    plt.savefig("loss.png")


# %%
dataset, user_ids, unique_product_ids, train_user_ids, val_user_ids, train_product_ids, val_product_ids, train_tfidf_vectors, val_tfidf_vectors, train_images, val_images, train_ratings, val_ratings = dfProcessing(
    user_data, product_data)
# %%
do_hybird = True
model = hybirdModel(user_ids, unique_product_ids, do_hybird=do_hybird)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

if do_hybird:
    history = model.fit([train_user_ids, train_product_ids, train_tfidf_vectors, train_images],
                        train_ratings, epochs=100, batch_size=16, verbose=1, shuffle=True,
                        validation_data=([val_user_ids, val_product_ids, val_tfidf_vectors, val_images], val_ratings), callbacks=[es])
    predictions = model.predict([val_user_ids, val_product_ids, val_tfidf_vectors, val_images], verbose=0)
else:
    history = model.fit([train_user_ids, train_product_ids],
                        train_ratings, epochs=100, batch_size=64, verbose=1, shuffle=True,
                        validation_data=([val_user_ids, val_product_ids], val_ratings))
    predictions = model.predict([val_user_ids, val_product_ids], verbose=0)

# %%

# Assuming your model is called "model"
plot_model(model, to_file='model_structure.png', show_shapes=True, show_layer_names=True)

y_pred = []
for y in predictions:
    y_pred.append(y[0])

y_test = []
for y in val_ratings:
    y_test.append(y)


model_ = history.model
utils_plot_keras_training(history)


def evalution(y_pred, y_test):
    import math
    MSE = np.square(np.subtract(y_test, y_pred)).mean()

    RMSE = math.sqrt(MSE)

    MAE = np.mean(np.abs(y_test - y_pred))
    print("Root Mean Square Error: ", RMSE, '\n')

    print("Mean Absolute Error: ", MAE, '\n')

    from scipy.stats import pearsonr
    print("Pearson Correlation : ", pearsonr(y_test, y_pred)[0])
    print("p-value: ", pearsonr(y_test, y_pred)[1])


evalution(y_pred, val_ratings)

# plotting the data
# plt.scatter(y_pred, y_test, s=1)

# # This will fit the best line into the graph
# plt.xlabel("Predict rating")
# plt.ylabel("Real rating")

# plt.plot(np.unique(y_pred), np.poly1d(np.polyfit(y_pred, y_test, 2))(np.unique(y_pred)), color='red')
# %%

print("Recommandation for Top 10 active reviewer: \n")
testset = {}

for n, id_ in enumerate(val_product_ids):
    testset[id_] = {'text': val_tfidf_vectors[n],
                    'image': val_images[n]}
top_10_active_reviewer = dataset["reviewerID"].value_counts(ascending=False).index[:10]

productId = np.unique(val_product_ids)


def createRecommandationSystem(reviewerID, testset):

    recommendation = pd.DataFrame(columns={'UserId', 'ProductId', 'Image', 'Text', 'Predicted Rating'})
    for product in testset:
        data = {"UserId": reviewerID, "ProductId": product, "Image": testset[product]['image'], "Text": testset[product]['text']}
        recommendation = pd.concat([recommendation, pd.DataFrame(data)], ignore_index=True)
    return recommendation


scaler = MinMaxScaler()
acc_list = []
for reviewerID in top_10_active_reviewer:

    print("--- reviewerID", reviewerID, "---")
    Real_buy = dataset[dataset['reviewerID'] == reviewerID][['asin', 'overall']]
    # print("Real Buy : \n",Real_buy)
    Bought = set(Real_buy['asin'].to_list())

    recommendation = createRecommandationSystem(reviewerID, testset)
    recommendation['Predicted Rating'] = scaler.fit_transform(model.predict([np.array(list(recommendation['UserId'], verbose=0)), np.array(
        list(recommendation['ProductId'])), np.array(list(recommendation['Text'])), np.array(list(recommendation['Image']))]))*5

    top_50 = recommendation.sort_values(by='Predicted Rating', ascending=False)[:50]
    top_50_items = set(top_50['ProductId'].to_list())
    print(top_50_items & Bought, 'in top 50 recommendation \n')
    if Bought != {}:
        acc_list.append(len(top_50_items & Bought)/50)
    print("Accuracy : ", len(top_50_items & Bought)/50)

print("Hit ratio @ 50 Accuracy: ", sum(acc_list)/len(acc_list))

# %%
# Assuming your model is called "model"
plot_model(model, to_file='model_structure.png', show_shapes=True, show_layer_names=True)
