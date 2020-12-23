# Builds a Convolutional Net autoencoder
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
import numpy as np

class ConvAutoencoder:
    @staticmethod
    def build(width, height, depth, filters=(32, 64), latent_dim=16, activity_reg=0.0001):
        input_shape = (height, width, depth)
        chan_dim = -1

        inputs = Input(shape=input_shape)
        x = inputs

        for f in filters:
            x = Conv2D(f, (3,3), strides=2, padding="same")(x)
            x = LeakyReLU()(x)
            x = BatchNormalization(axis=chan_dim)(x)

        vol = K.int_shape(x)
        x = Flatten()(x)
        # encoder output
        latent = Dense(latent_dim, name="encoder_reg", activity_regularizer=l1(activity_reg), kernel_regularizer=l2(0.0001))(x)
        latent = Activation("relu", name="encoder")(latent)

        # Decoder
        x = Dense(np.prod(vol[1:]))(latent)
        x = Reshape((vol[1], vol[2], vol[3]))(x)

        for f in filters[::-1]:
            x = Conv2DTranspose(f, (3,3), strides=2, padding="same")(x)
            x = LeakyReLU()(x)
            x = BatchNormalization(axis=chan_dim)(x)

        # apply single Conv2DTranspose to recover original depth of image
        x = Conv2DTranspose(depth, (3,3), padding="same")(x)
        outputs = Activation("sigmoid", name="decoder")(x)

        autoencoder = Model(inputs, outputs, name="autoencoder")

        return autoencoder