import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten, Dense

import utils
from utils import num_classes

def create_MiniVGGNet_2_blocks(input_shape):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), name='conv1_1',
                     padding='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=1))
    model.add(Conv2D(32, (3, 3), name='conv1_2', padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=3))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), name='conv2_1', padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=3))
    model.add(Conv2D(64, (3, 3), name='conv2_2', padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=3))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation("softmax"))

    print(model.summary())

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model, utils.MODEL_PATH