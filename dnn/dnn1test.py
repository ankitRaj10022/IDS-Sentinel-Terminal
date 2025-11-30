from __future__ import print_function
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
np.random.seed(1337)
from keras.preprocessing import sequence
from keras.models import Sequential, load_model
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
testdata = pd.read_csv('kdd/binary/Testing.csv', header=None)


X = traindata.iloc[:,1:42]
Y = traindata.iloc[:,0]
C = testdata.iloc[:,0]
T = testdata.iloc[:,1:42]

trainX = np.array(X)
testT = np.array(T)

trainX.astype(float)
testT.astype(float)

scaler = Normalizer().fit(trainX)
trainX = scaler.transform(trainX)

scaler = Normalizer().fit(testT)
testT = scaler.transform(testT)

y_train = np.array(Y)
y_test = np.array(C)


X_train = np.array(trainX)
X_test = np.array(testT)


batch_size = 64

model = Sequential()
model.add(Dense(1024,input_dim=41,activation='relu'))
model.add(Dropout(0.01))
model.add(Dense(1))
model.add(Activation('sigmoid'))

# model.save('dnn1layer_model.keras', include_optimizer=True)
# model = load_model()
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])



score = []
name = []
from sklearn.metrics import confusion_matrix
import os
for file in os.listdir("kddresults/dnn1layer/"):
  if file.endswith('.keras'):
    try:
        temp_model = load_model("kddresults/dnn1layer/"+file, compile=False)
        model.set_weights(temp_model.get_weights())
    except:
        model.load_weights("kddresults/dnn1layer/"+file, skip_mismatch=True, by_name=True)

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


best_model_path = "kddresults/dnn1layer/"+name[score.index(max(score))]
try:
    temp_model = load_model(best_model_path, compile=False)
    model.set_weights(temp_model.get_weights())
except:
    model.load_weights(best_model_path, skip_mismatch=True, by_name=True)
proba = model.predict(X_test)
pred = (proba > 0.5).astype(int)
np.savetxt("dnnres/dnn1predicted.txt", pred)
np.savetxt("dnnres/dnn1probability.txt", proba)

accuracy = accuracy_score(y_test, pred)
recall = recall_score(y_test, pred , average="binary")
precision = precision_score(y_test, pred , average="binary")
f1 = f1_score(y_test, pred, average="binary")


print("----------------------------------------------")
print("accuracy")
print("%.3f" %accuracy)
print("precision")
print("%.3f" %precision)
print("racall")
print("%.3f" %recall)
print("f1score")
print("%.3f" %f1)

'''
model.load_weights("kddresults/dnn1layer/"+name[score.index(max(score))])
pred = model.predict_classes(X_test)
proba = model.predict_proba(X_test)
np.savetxt("dnnres/dnn1predicted.txt", pred)
np.savetxt("dnnres/dnn1probability.txt", proba)

accuracy = accuracy_score(y_test, pred)
recall = recall_score(y_test, pred , average="binary")
precision = precision_score(y_test, pred , average="binary")
f1 = f1_score(y_test, pred, average="binary")


print("----------------------------------------------")
print("accuracy")
print("%.3f" %accuracy)
print("precision")
print("%.3f" %precision)
print("racall")
print("%.3f" %recall)
print("f1score")
print("%.3f" %f1)
print(model.summary())
'''
