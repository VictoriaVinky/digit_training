import os
import random
import utils

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

freq_axis = 1
time_axis = 2
channel_axis = 3

num_classes = 10

batch_size = 4
epochs = 250

N_frames = 45
N_MELS = 45

img_rows, img_cols = N_frames, N_MELS
input_shape = (img_rows, img_cols, 1)

DATA_PATH = "./data_train/"
MFCC_PATH = "./train_mfcc/mfcc/"
MODEL_PATH = "./train_mfcc/model/"

# lst_train_mixed.txt: txt file contains list of file names for training
LIST_TRAIN = "./file_list/lst_train_mixed.txt"
# lst_valid_mixed.txt: txt file contains list of file names for validation
LIST_VALID = "./file_list/lst_valid_mixed.txt"
# lst_test_mixed.txt: txt file contains list of file names for testing
LIST_TEST = "./file_list/lst_test_mixed.txt"

RESULT_PATH = "./train_mfcc/result/"
RESULT_FILE = "0_result.txt"
CSV_FILE = "0_result.csv"

RATIO_TRAINING = 0.5
RATIO_VALIDATION = 0.25
RATIO_TESTING = 0.25

lst_user_khong  = ["khong"]
lst_user_mot    = ["mot"]
lst_user_hai    = ["hai"]
lst_user_ba     = ["ba"]
lst_user_bon    = ["bon"]
lst_user_nam    = ["nam"]
lst_user_sau    = ["sau"]
lst_user_bay    = ["bay"]
lst_user_tam    = ["tam"]
lst_user_chin   = ["chin"]

def get_digit(usercode):
    if usercode in lst_user_khong:
        return 0
    elif usercode in lst_user_mot:
        return 1
    elif usercode in lst_user_hai:
        return 2
    elif usercode in lst_user_ba:
        return 3
    elif usercode in lst_user_bon:
        return 4
    elif usercode in lst_user_nam:
        return 5
    elif usercode in lst_user_sau:
        return 6
    elif usercode in lst_user_bay:
        return 7
    elif usercode in lst_user_tam:
        return 8
    elif usercode in lst_user_chin:
        return 9
    else:
        return -1

def get_digit_text(usercode):
    if usercode == 0:
        return "Khong"
    elif usercode == 1:
        return "Mot"
    elif usercode == 2:
        return "Hai"
    elif usercode == 3:
        return "Ba"
    elif usercode == 4:
        return "Bon"
    elif usercode == 5:
        return "Nam"
    elif usercode == 6:
        return "Sau"
    elif usercode == 7:
        return "Bay"
    elif usercode == 8:
        return "Tam"
    elif usercode == 9:
        return "Chin"
    else:
        return -1

def display_data_chart(mfcc_trains, mfcc_valids, mfcc_tests):
    count = np.zeros((3, 10), dtype=int)
    for _file in mfcc_trains:
        count[0][get_digit(_file.partition("-")[0])] += 1
    for _file in mfcc_valids:
        count[1][get_digit(_file.partition("-")[0])] += 1
    for _file in mfcc_tests:
        count[2][get_digit(_file.partition("-")[0])] += 1

    index = np.arange(10)

    plt.figure(figsize=(12, 5))
    
    ax = plt.subplot(1, 3, 1)
    plt.grid(axis='y', alpha=0.25)
    plt.bar(index, count[0], align='center', alpha=0.5, color='b', label='Training data')
    plt.xticks(range(10), range(10))
    plt.yticks(np.arange(85, step=5))
    plt.title("Training data")
    for i in range(10):
        plt.text(i, count[0][i], str(count[0][i]), horizontalalignment='center', fontsize=8)
    
    ax = plt.subplot(1, 3, 2)
    plt.grid(axis='y', alpha=0.25)
    plt.bar(index, count[1], align='center', alpha=0.5, color='g', label='Validation data')
    plt.xticks(range(10), range(10))
    plt.yticks(np.arange(85, step=5))
    plt.title("Validation data")
    for i in range(10):
        plt.text(i, count[1][i], str(count[1][i]), horizontalalignment='center', fontsize=8)

    ax = plt.subplot(1, 3, 3)
    plt.grid(axis='y', alpha=0.25)
    plt.bar(index, count[2], align='center', alpha=0.5, color='r', label='Validation data')
    plt.xticks(range(10), range(10))
    plt.yticks(np.arange(85, step=5))
    plt.title("Testing data")
    for i in range(10):
        plt.text(i, count[2][i], str(count[2][i]), horizontalalignment='center', fontsize=8)

    plt.show()

def display_single_data_chart(mfcc_files):
    count = np.zeros(10, dtype=int)
    for _file in mfcc_files:
        count[get_digit(_file.partition("-")[0])] += 1

    index = np.arange(10)

    plt.figure(figsize=(5, 5))
    
    ax = plt.subplot(1, 1, 1)
    plt.grid(axis='y', alpha=0.25)
    plt.bar(index, count, align='center', alpha=0.5, color='b', label='Training data')
    plt.xticks(range(10), range(10))
    plt.yticks(np.arange(np.max(count) + 10, step=5))
    for i in range(10):
        plt.text(i, count[i], str(count[i]), horizontalalignment='center', fontsize=8)

    plt.show()

def display_training_result_graph(epochs, acc, val_acc, loss, val_loss):
    epochs_range = range(epochs)

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy (' + str(np.max(acc)) + ' - ' + str(np.max(val_acc)) + ')')

    plt.subplot(2, 1, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()