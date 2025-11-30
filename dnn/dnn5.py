from __future__ import print_function
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
np.random.seed(1337)
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, SimpleRNN, GRU
from keras.datasets import imdb
from keras.utils import to_categorical
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
from sklearn import metrics
from sklearn.preprocessing import Normalizer
import h5py
from keras import callbacks
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger

traindata = pd.read_csv('kdd/binary/Training.csv', header=None)
import os
print("Current working directory:", os.getcwd())
testdata = pd.read_csv('kdd/binary/Testing.csv', header=None)


X = traindata.iloc[:,1:42]
Y = traindata.iloc[:,0]
C = testdata.iloc[:,0]
T = testdata.iloc[:,1:42]

scaler = Normalizer().fit(X)
trainX = scaler.transform(X)

scaler = Normalizer().fit(T)
testT = scaler.transform(T)

y_train = np.array(Y)
y_test = np.array(C)

X_train = np.array(trainX)
X_test = np.array(testT)


batch_size = 64

model = Sequential()
model.add(Dense(1024,input_dim=41,activation='relu'))
model.add(Dropout(0.01))
model.add(Dense(768,activation='relu'))
model.add(Dropout(0.01))
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.01))
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.01))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.01))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
checkpointer = callbacks.ModelCheckpoint(filepath="kddresults/dnn5layer/checkpoint-{epoch:02d}.keras", verbose=1, save_best_only=True, monitor='loss')
csv_logger = CSVLogger('kddresults/dnn5layer/training_set_dnnanalysis.csv',separator=',', append=False)
model.fit(X_train, y_train, batch_size=batch_size, epochs=100, callbacks=[checkpointer,csv_logger])
model.save("kddresults/dnn5layer/dnn5layer_model.keras")
