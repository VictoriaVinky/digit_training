from __future__ import absolute_import, division, print_function, unicode_literals

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
from tensorflow.keras.utils import get_file
from tensorflow.keras.models import model_from_json
from tensorflow.keras.callbacks import Callback

import os
import random
import numpy as np
import time
import resource

import utils
from utils import MODEL_PATH, LIST_TEST, MFCC_PATH
from utils import freq_axis, time_axis, channel_axis
from utils import num_classes, epochs, batch_size
from utils import img_rows, img_cols
from utils import N_MELS, N_frames

import training_model

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

# print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# BatchNormalization._USE_V2_BEHAVIOR = False

tf.keras.backend.set_image_data_format('channels_last')

def load_data():
    global x_train, y_train, x_valid, y_valid, input_shape
    mfcc_trains = [line.rstrip() for line in open(utils.LIST_TRAIN)]
    mfcc_valids = [line.rstrip() for line in open(utils.LIST_VALID)]
    # mfcc_tests = [line.rstrip() for line in open(utils.LIST_TEST)]

    # utils.display_data_chart(mfcc_trains, mfcc_valids, mfcc_tests)
    ##################################
    # Load training mfcc
    x_train = np.zeros(shape=(len(mfcc_trains), N_frames, N_MELS))
    y_train = np.zeros(shape=(len(mfcc_trains)))
    X_scale = StandardScaler()
    idx = 0
    for mfcc_file in mfcc_trains:
        mfcc_file_full_path = utils.MFCC_PATH + mfcc_file
        print("##########TRAIN {}:\t{}".format(idx, mfcc_file_full_path))

        x_train_r = np.loadtxt(mfcc_file_full_path)
        X_train = X_scale.fit_transform(x_train_r)

        y_train_r = utils.get_digit(mfcc_file.partition("-")[0])

        x_train[idx] = X_train
        y_train[idx] = int(y_train_r)

        idx += 1
    ##################################
    # Load validation mfcc
    x_valid = np.zeros(shape=(len(mfcc_valids), N_frames, N_MELS))
    y_valid = np.zeros(shape=(len(mfcc_valids)))
    X_scale = StandardScaler()
    idx = 0
    for mfcc_file in mfcc_valids:
        mfcc_file_full_path = utils.MFCC_PATH + mfcc_file
        print("##########VALID {}:\t{}".format(idx, mfcc_file_full_path))

        x_valid_r = np.loadtxt(mfcc_file_full_path)
        X_valid = X_scale.fit_transform(x_valid_r)

        y_valid_r = utils.get_digit(mfcc_file.partition("-")[0])

        x_valid[idx] = X_valid
        y_valid[idx] = int(y_valid_r)

        idx += 1
    # Reformat data
    print("Dinh dang lai du lieu cho phu hop kien truc CNN ...")
    print(tf.keras.backend.image_data_format())
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_valid = x_valid.reshape(x_valid.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

    print("input shape: ", input_shape)

    x_train = x_train.astype('float32')
    x_valid = x_valid.astype('float32')
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_valid = tf.keras.utils.to_categorical(y_valid, num_classes)

def save_model(epoch):
    file_name = model_path + "model_" + str(epoch + 1)
    file_json = file_name + ".json"
    file_h5 = file_name + ".h5"
    # Save model
    print("Save model: " + file_json + " ; " + file_h5)
    # serialize model to JSON
    model_json = model.to_json()
    with open(file_json, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(file_h5)
    print("######################")
    print("#Saved model to disk!#")
    print("######################")


class MemoryCallback(Callback):
    def on_batch_end(self, epoch, logs={}):
        print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

    def on_epoch_end(self, epoch, logs={}):
        save_model(epoch)

def training():
    print("\n##### Training ...")
    start_time = time.clock()
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        callbacks=[MemoryCallback()],
        validation_data=(x_valid, y_valid))

    end_time = time.clock()
    print("Training time (seconds): ", end_time - start_time)

    return history

if __name__ == '__main__':
    print("======DIGIT TRAINING======")
    print("GPU is", "AVAILABLE!" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE!")
    print("\n\n1. Load data training")
    load_data()

    print("\n\n2. Define model")
    global model, model_path
    model, model_path = training_model.create_MiniVGGNet_2_blocks(input_shape)
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
    tf.keras.utils.plot_model(model, utils.RESULT_PATH + '0_model.png', show_shapes=True)

    print("\n\n3. Training")
    history = training()
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    print("Maximum of train accuracy = ", np.max(acc))
    print("Maximum of valid accuracy = ", np.max(val_acc))

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    utils.display_training_result_graph(epochs, acc, val_acc, loss, val_loss)