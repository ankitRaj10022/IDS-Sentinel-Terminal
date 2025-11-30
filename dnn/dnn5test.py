from __future__ import print_function
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
np.random.seed(1337)
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
from sklearn import metrics
from sklearn.preprocessing import Normalizer
import h5py
from keras import callbacks
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger

traindata = pd.read_csv('kdd/binary/Training.csv', header=None)
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

'''
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
checkpointer = callbacks.ModelCheckpoint(filepath="kddresults/dnn5layer/checkpoint-{epoch:02d}.hdf5", verbose=1, save_best_only=True, monitor='loss')
csv_logger = CSVLogger('kddresults/dnn5layer/training_set_dnnanalysis.csv',separator=',', append=False)
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=100, callbacks=[checkpointer,csv_logger])
model.save("kddresults/dnn5layer/dnn5layer_model.hdf5")
'''

score = []
name = []
from sklearn.metrics import confusion_matrix
import os
for file in os.listdir("kddresults/dnn5layer/"):
  if file.endswith(('.hdf5', '.h5', '.keras', '.weights.h5')):
    model.load_weights("kddresults/dnn5layer/"+file)
    y_train1 = y_test
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    accuracy = accuracy_score(y_train1, y_pred)
    recall = recall_score(y_train1, y_pred , average="binary")
    precision = precision_score(y_train1, y_pred , average="binary")
    f1 = f1_score(y_train1, y_pred, average="binary")
    print("----------------------------------------------")
    print("accuracy")
    print("%.3f" %accuracy)
    print("recall")
    print("%.3f" %recall)
    print("precision")
    print("%.3f" %precision)
    print("f1score")
    print("%.3f" %f1)
    score.append(accuracy)
    name.append(file)


model.load_weights("kddresults/dnn5layer/"+name[score.index(max(score))])
pred = (model.predict(X_test) > 0.5).astype(int)
proba = model.predict(X_test)
np.savetxt("dnnres/dnn5predicted.txt", pred)
np.savetxt("dnnres/dnn5probability.txt", proba)

accuracy = accuracy_score(y_test, pred)
recall = recall_score(y_test, pred , average="binary")
precision = precision_score(y_test, pred , average="binary")
f1 = f1_score(y_test, pred, average="binary")


print("----------------------------------------------")
print("accuracy")
print("%.3f" %accuracy)
print("precision")
print("%.3f" %precision)
print("recall")
print("%.3f" %recall)
print("f1score")
print("%.3f" %f1)
