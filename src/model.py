import tensorflow as tf
from keras import optimizers
from keras.layers import Concatenate, Dense, Dot, Dropout, Embedding, Input, Reshape
from keras.models import Model


def hybrid_model(num_user: int, num_prod: int, text_vec_size: int, img_vec_size: int, emb_size: int = 32, hybird: bool = True):
    # Defining input layers
    x_user_in = Input(name="User_input", shape=(1,))
    x_prod_in = Input(name="Product_input", shape=(1,))

    # A) Matrix Factorization
    # Embeddings and Reshape layers for user ids
    cf_xusers_emb = Embedding(name="MF_User_Embedding", input_dim=num_user, output_dim=emb_size)(x_user_in)
    cf_xusers = Reshape(name='MF_User_Reshape', target_shape=(emb_size,))(cf_xusers_emb)
    # Embeddings and Reshape layers for product ids
    cf_xproducts_emb = Embedding(name="MF_Product_Embedding", input_dim=num_prod, output_dim=emb_size)(x_prod_in)
    cf_xproducts = Reshape(name='MF_Product_Reshape', target_shape=(emb_size,))(cf_xproducts_emb)
    # Dot product layer
    cf_xx = Dot(name='MF_Dot', normalize=True, axes=1)([cf_xusers, cf_xproducts])

    # B) Neural Network
    # Embeddings and Reshape layers for user ids
    nn_xusers_emb = Embedding(name="NN_User_Embedding", input_dim=num_user, output_dim=emb_size)(x_user_in)
    nn_xusers = Reshape(name='NN_User_Reshape', target_shape=(emb_size,))(nn_xusers_emb)
    # Embeddings and Reshape layers for product ids
    nn_xproducts_emb = Embedding(name="NN_Product_Embedding", input_dim=num_prod, output_dim=emb_size)(x_prod_in)
    nn_xproducts = Reshape(name='NN_Product_Reshape', target_shape=(emb_size,))(nn_xproducts_emb)
    # Concatenate and dense layers
    nn_xx = Concatenate()([nn_xusers, nn_xproducts])
    nn_xx = Dense(name="NN_layer", units=16, activation='relu')(nn_xx)
    nn_xx = Dropout(0.1)(nn_xx)

    # If do_hybrid is True, add text-based and image-based models
    if hybird:
        # 1) TEXT BASED
        text_in = Input(name="title_input", shape=(text_vec_size,))
        text_x = Dense(name="title_layer", units=64, activation='relu')(text_in)

        # 2) IMAGE BASED
        image_in = Input(name="image_input", shape=(img_vec_size,))
        image_x = Dense(name="image_layer", units=256, activation='relu')(image_in)

        content_xx = Concatenate()([text_x, image_x])
        content_xx = Dense(name="contect_layer", units=128, activation='relu')(content_xx)

        # 3) Merge all
        y_out = Concatenate()([cf_xx, nn_xx, content_xx])

    else:
        y_out = Concatenate()([cf_xx, nn_xx])

    y_out = Dense(name="CF_contect_layer", units=64, activation='linear')(y_out)
    y_out = Dense(name="y_output", units=1, activation='linear')(y_out)

    # C) Compile
    if hybird:
        model = Model(inputs=[x_user_in, x_prod_in, text_in, image_in], outputs=y_out, name="Hybrid_Model")
    else:
        model = Model(inputs=[x_user_in, x_prod_in], outputs=y_out, name="Hybrid_Model")

    adam = optimizers.Adam(learning_rate=0.01, decay=0.0001)
    model.compile(optimizer=adam, loss='mae', metrics=[tf.keras.metrics.RootMeanSquaredError(), tf.keras.metrics.mean_absolute_error])

    return model
