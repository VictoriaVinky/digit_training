from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf

import os
import time
import utils
import pickle
import numpy as np 

from utils import num_classes, epochs
from utils import img_rows, img_cols, input_shape
from utils import N_MELS, N_frames

from sklearn.preprocessing import StandardScaler

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

def loading_testing_data():
    mfcc_tests = [line.rstrip() for line in open(utils.LIST_TEST)]
    # data_chart.display_single_data_chart(mfcc_tests)

    x_test = np.zeros(shape=(len(mfcc_tests), N_frames, N_MELS))
    y_test = np.zeros(shape=(len(mfcc_tests)))
    X_scale = StandardScaler()
    idx = 0
    for mfcc_file in mfcc_tests:
        mfcc_file_full_path = utils.MFCC_PATH + mfcc_file
        print("##########TEST {}:\t{}".format(idx, mfcc_file_full_path))
        
        x_test_r = np.loadtxt(mfcc_file_full_path)
        X_test = X_scale.fit_transform(x_test_r)

        y_test_r = utils.get_digit(mfcc_file.partition("-")[0])

        x_test[idx] = X_test
        y_test[idx] = int(y_test_r)
        
        idx += 1

    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    x_test = x_test.astype('float32')
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    return x_test, y_test

def load_model(file_json, file_h5):
    print("\nLoading model from disk ...")
    json_file = open(file_json, "r")
    loaded_model_json = json_file.read()
    json_file.close()
    model = tf.keras.models.model_from_json(loaded_model_json)
    model.load_weights(file_h5)
    print(" Done!")

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    
    print(model.summary())

    return model

def format(value):
    return "%5.2f" % value

def formatf(value):
    return "%f" % value

def formati(value):
    return "%4d" % value

def eval_model(model, result_path, x, y):
    print("Begin Evaluate Model .............\n")
    dateNow = time.strftime("%d/%m/%Y")
    timeNow = time.strftime("%Hh:%Mm:%Ss")

    if os.path.isfile(result_path + utils.RESULT_FILE) == 1:
        file_result = open(result_path + utils.RESULT_FILE, "a")
    else:
        file_result = open(result_path + utils.RESULT_FILE, "w")
    
    file_result.write("\n===========================================================\n")
    file_result.write(dateNow + "\t" + timeNow + "\n")

    if os.path.isfile(result_path + utils.CSV_FILE) == 1:
        file_CSV = open(result_path + utils.CSV_FILE, "a")
    else:
        file_CSV = open(result_path + utils.CSV_FILE, "w")
        col_title = "Date,Time,No. Epochs,Loss,Accuracy," \
                    "Khong_Khong, Khong_Mot, Khong_Hai, Khong_Ba, Khong_Bon, Khong_Nam, Khong_Sau, Khong_Bay, Khong_Tam, Khong_Chin," \
                    "Mot_Khong, Mot_Mot, Mot_Hai, Mot_Ba, Mot_Bon, Mot_Nam, Mot_Sau, Mot_Bay, Mot_Tam, Mot_Chin," \
                    "Hai_Khong, Hai_Mot, Hai_Hai, Hai_Ba, Hai_Bon, Hai_Nam, Hai_Sau, Hai_Bay, Hai_Tam, Hai_Chin," \
                    "Ba_Khong, Ba_Mot, Ba_Hai, Ba_Ba, Ba_Bon, Ba_Nam, Ba_Sau, Ba_Bay, Ba_Tam, Ba_Chin," \
                    "Bon_Khong, Bon_Mot, Bon_Hai, Bon_Ba, Bon_Bon, Bon_Nam, Bon_Sau, Bon_Bay, Bon_Tam, Bon_Chin," \
                    "Nam_Khong, Nam_Mot, Nam_Hai, Nam_Ba, Nam_Bon, Nam_Nam, Nam_Sau, Nam_Bay, Nam_Tam, Nam_Chin," \
                    "Sau_Khong, Sau_Mot, Sau_Hai, Sau_Ba, Sau_Bon, Sau_Nam, Sau_Sau, Sau_Bay, Sau_Tam, Sau_Chin," \
                    "Bay_Khong, Bay_Mot, Bay_Hai, Bay_Ba, Bay_Bon, Bay_Nam, Bay_Sau, Bay_Bay, Bay_Tam, Bay_Chin," \
                    "Tam_Khong, Tam_Mot, Tam_Hai, Tam_Ba, Tam_Bon, Tam_Nam, Tam_Sau, Tam_Bay, Tam_Tam, Tam_Chin," \
                    "Chin_Khong, Chin_Mot, Chin_Hai, Chin_Ba, Chin_Bon, Chin_Nam, Chin_Sau, Chin_Bay, Chin_Tam, Chin_Chin\n"
        file_CSV.write(col_title)
    loss, acc = model.evaluate(x, y)
    
    print("End Evaluate Model \n")
    file_result.write("Loss = " + str(formatf(loss)) + "\n")
    file_result.write("Acc  = " + str(formatf(acc)) + "\n\n")
    
    return loss, acc, file_result, file_CSV, dateNow, timeNow

def recognize_one_file(model, mfcc_file):
    x_data = np.zeros(shape=(1,N_frames, N_MELS))
    X_scale = StandardScaler()
    x_data_r = np.loadtxt(utils.MFCC_PATH + mfcc_file)
    X_data = X_scale.fit_transform(x_data_r)
    x_data[0] = X_data
    x_data = x_data.reshape(x_data.shape[0], img_rows, img_cols, 1)
    x_data = x_data.astype('float32')

    y_data = int(utils.get_digit(mfcc_file.partition("-")[0]))

    y_predict = model.predict(x_data)
    print("y_predict: ", y_predict)
    
    a = np.argmax(y_predict, axis=1)
    print(a)
    counts = np.bincount(a)
    print(counts)
    digit_predict = np.argmax(counts)
    print("digit_predict: ", digit_predict)

    (values, counts) = np.unique(a, return_counts=True)
    print(values, counts)
    digit_predict = values[np.argmax(counts)]
    print("digit_predict: ", digit_predict)

    return (y_data, digit_predict)        

def recognize_list_file(model, nb, loss, acc, file_result, file_CSV, dateNow, timeNow):
    lst_file = [line.rstrip() for line in open(utils.LIST_TEST)]
    print("Tong so file test: ", len(lst_file))

    # Init digit count
    digitCounts = np.zeros(shape=(10, 10), dtype=int)

    # Tong so Test Files
    alls = np.zeros(shape=(10), dtype=int)
    for _file in lst_file:
        alls[utils.get_digit(_file.partition("-")[0])] += 1

    idx = 0
    for mfcc_file in lst_file:
        result = recognize_one_file(model, mfcc_file)
        print("Result: ", result)
        digitCounts[result[0]][result[1]] += 1
        idx += 1
    
    scores = np.zeros(shape=(10, 10), dtype=float)
    for i in range(10):
        for j in range(10):
            scores[i][j] = digitCounts[i][j] * 100.0 / alls[i]
    
    scoreAverage = 0
    for i in range(10):
        scoreAverage += scores[i][i]
    scoreAverage = scoreAverage/10

    file_result.write("Number of epochs = " + str(formati(nb)) + "\n\n")
    file_result.write("\t\tKhong\tMot\tHai\tBa\tBon\tNam\tSau\tBay\tTam\tChin\n")
    for i in range(10):
        text = str(utils.get_digit_text(i)) + "\t"
        for j in range(10):
            text = text + "\t" + str(format(scores[i][j]))
        file_result.write(text + "\n")
    file_result.write("\n")
    file_result.write("Average Score = " + str(format(scoreAverage))+"\n")
    file_result.write("\t\tKhong\tMot\tHai\tBa\tBon\tNam\tSau\tBay\tTam\tChin\n")
    for i in range(10):
        text = str(utils.get_digit_text(i)) + "\t"
        for j in range(10):
            text = text + "\t" + str(formati(digitCounts[i][j]))
        file_result.write(text + "\n")
    file_result.write("\n")

    file_result.close()

    text = str(dateNow) + "," + str(timeNow) + "," + str(formati(nb)) + "," + str(formatf(loss)) + "," + str(formatf(acc))
    for i in range(10):
        for j in range(10):
            text = text + "," + str(format(scores[i][j]))
    file_CSV.write(time.strftime(text + "\n"))
    file_CSV.close()

if __name__ == '__main__':
    print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")
    max_model = epochs

    x_test, y_test = loading_testing_data()

    ebegin = 202
    model_path = utils.MODEL_PATH

    for i in range(ebegin, max_model + 1):
        file_json = model_path + "model_" + str(i) + ".json"
        file_h5 = model_path + "model_" + str(i) + ".h5"
        print("Model: " + file_json + " ; " + file_h5)

        model = load_model(file_json, file_h5)

        loss, acc, file_result, file_CSV, dateNow, timeNow = eval_model(model, utils.RESULT_PATH, x_test, y_test)

        recognize_list_file(model, i, loss, acc, file_result, file_CSV, dateNow, timeNow)