import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import soundfile as sf
from python_speech_features import mfcc, logfbank

# forming a pandas dataframe from the metadata file
data = pd.read_csv("F:/Rashmama office/Denoiser-master/Denoiser-master/compressed_dataset/UrbanSound8K.csv")

x_train = []
x_test = []
y_train = []
y_test = []
path = "F:/Rashmama office/noise github/UrbanSound8K/UrbanSound8K/audio/fold"

for i in tqdm(range(len(data))):
    fold_no = str(data.iloc[i]["fold"])
    file = data.iloc[i]["slice_file_name"]
    label = data.iloc[i]["classID"]
    filename = os.path.join(path + fold_no, file)
    y, sr = sf.read(filename)

    # Extracting MFCC and filter bank features
    mfcc_features = mfcc(y, sr)
    filterbank_features = logfbank(y, sr)

    # Reshaping features
    features = np.hstack((mfcc_features.mean(axis=0), filterbank_features.mean(axis=0)))

    if fold_no != '10':
        x_train.append(features)
        y_train.append(label)
    else:
        x_test.append(features)
        y_test.append(label)

print('Length of Data: ', len(x_train) + len(x_test))
x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# Reshaping into 2d to save in csv format
x_train_2d = np.reshape(x_train, (x_train.shape[0], x_train.shape[1]))
x_test_2d = np.reshape(x_test, (x_test.shape[0], x_test.shape[1]))

# Saving the data numpy arrays
np.savetxt("train_data.csv", x_train_2d, delimiter=",")
np.savetxt("test_data.csv", x_test_2d, delimiter=",")
np.savetxt("train_labels.csv", y_train, delimiter=",")
np.savetxt("test_labels.csv", y_test, delimiter=",")
