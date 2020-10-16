import numpy as np
seed = 123
np.random.seed(seed)
from tensorflow.compat.v1 import set_random_seed
set_random_seed(seed)
import utility as ut
import pandas as pd
from sklearn import preprocessing
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

def dec_to_bin(x):
    return format(int(x), "b")

def flat_vec(df):
    list_image_flat = []
    for (index_label, row_series) in df.iterrows():
        list_image = []
        j = 0
        while j < len(row_series):
            v = row_series[j] * (2 ** 24 - 1)
            bin_num = dec_to_bin(int(v))
            if len(bin_num) < 24:
                pad = 24 - len(bin_num)
                zero_pad = "0" * pad
                line = zero_pad + str(bin_num)
                n = 8
                rgb = [line[i:i + n] for i in range(0, len(line), n)]
            else:
                n = 8
                line = str(bin_num)
                rgb = [line[i:i + n] for i in range(0, len(line), n)]
            list_image.append(rgb)
            j = j + 1
        list_image_flat.append(list_image)
    return list_image_flat

def flat_vec_test(df):
    list_image_flat = []
    for (index_label, row_series) in df.iterrows():
        list_image = []
        j = 0
        while j < len(row_series):
            if row_series[j] > 1:
                c = 1.0
            elif row_series[j] < 0:
                c = 0.0
            else:
                c = row_series[j]

            v = c * (2 ** 24 - 1)
            bin_num = dec_to_bin(int(v))
            if len(bin_num) < 24:
                pad = 24 - len(bin_num)
                zero_pad = "0" * pad
                line = zero_pad + str(bin_num)
                n = 8
                rgb = [line[i:i + n] for i in range(0, len(line), n)]
            else:
                n = 8
                line = str(bin_num)
                rgb = [line[i:i + n] for i in range(0, len(line), n)]
            list_image.append(rgb)
            j = j + 1
        list_image_flat.append(list_image)
    return list_image_flat

def get_image_size(num_col):
    matx = 2
    i = False
    while i == False:
        size = matx * matx
        if size >= num_col:
            padding = size-num_col
            i = True
        else:
            matx = matx + 1
    return matx, padding

def rgb_img(list_image_flat_train, num_col):
    x = 0
    list_rgb = []
    size, padding = get_image_size(num_col)
    vec = [[0, 0, 0]] * padding
    while x < len(list_image_flat_train):
        y = 0
        list_img = []
        while y < len(list_image_flat_train[x]):
            z = 0
            img = []
            while z < len(list_image_flat_train[x][y]):
                bin_num = (list_image_flat_train[x][y][z])
                int_num = int(bin_num, 2)
                img.append(int_num)
                z = z + 1
            list_img.append(img)
            y = y + 1
        list_img = list_img + vec
        new_img = np.asarray(list_img)
        new_img = new_img.reshape(size, size, 3)
        list_rgb.append(new_img)
        x = x + 1
    return list_rgb

def rgb_img_test(list_image_flat_train, num_col):
    x = 0
    list_rgb = []
    size, padding = get_image_size(num_col)
    vec = [[0, 0, 0]] * padding
    while x < len(list_image_flat_train):
        y = 0
        list_img = []
        while y < len(list_image_flat_train[x]):
            z = 0
            img = []
            while z < len(list_image_flat_train[x][y]):
                bin_num = (list_image_flat_train[x][y][z])
                int_num = int(bin_num, 2)
                img.append(int_num)
                z = z + 1
            list_img.append(img)
            y = y + 1
        list_img = list_img + vec
        new_img = np.asarray(list_img)
        new_img = new_img.reshape(size, size, 3)
        list_rgb.append(new_img)
        x = x + 1
    return list_rgb

namedataset = "receipt" # change with name of the dataset
df = pd.read_csv('kometa_fold/'+namedataset+'feature.csv', header=None)
fold1, fold2, fold3 = ut.get_size_fold(namedataset)
df = df.iloc[:, :-1] # remove target column
num_col = len(df. columns)
X_1 = df[:fold1]
X_2 = df[fold1:(fold1+fold2)]
X_3 = df[(fold1+fold2):]

f = 0

for f in range(3):
    print("Fold n.", f)
    if f == 0:
        X_train = X_1.append(X_2)
        X_test = X_3
    elif f == 1:
        X_train = X_2.append(X_3)
        X_test = X_1
    elif f == 2:
        X_train = X_1.append(X_3)
        X_test = X_2

    dataframe_train = X_train
    dataframe_test = X_test

    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    scaler.fit(dataframe_train.values.astype(float))

    train_norm = scaler.transform(dataframe_train.values.astype(float))
    test_norm = scaler.transform(dataframe_test.values.astype(float))

    train_norm = pd.DataFrame(train_norm)
    test_norm = pd.DataFrame(test_norm)

    print("prepare fold n.", f)
    list_image_flat_train = flat_vec(train_norm)
    list_image_flat_test = flat_vec_test(test_norm)

    print("make fold n.", f)
    X_train = rgb_img(list_image_flat_train, num_col)
    X_test = rgb_img_test(list_image_flat_test, num_col)

    np.save("image/"+namedataset+"/"+namedataset+"_train_fold_"+str(f) + ".npy", X_train)
    np.save("image/"+namedataset+"/"+namedataset+"_test_fold_" + str(f) + ".npy", X_test)

