import numpy as np
import matplotlib.pyplot as plt
import h5py
import os

import keras
from keras import utils
from keras.utils.np_utils import to_categorical

from keras.datasets import mnist

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout


from keras.callbacks import EarlyStopping, ModelCheckpoint
#Data preprocessing

(x_train, y_train), (X_test, Y_test) = mnist.load_data()
x_train  = x_train.astype(np.float32)/255
X_test = X_test.astype(np.float32)/255

x_train = np.expand_dims(x_train, -1)
X_test = np.expand_dims(X_test, -1)

y_train = utils.to_categorical(y_train)
Y_test = utils.to_categorical(Y_test)



model = Sequential()

model.add(Conv2D(32, (3,3), input_shape = (28,28,1), activation='relu'))
model.add(MaxPool2D((2,2)))

model.add(Conv2D(64,(3,3), activation='relu'))
model.add(MaxPool2D((2,2)))

model.add(Flatten())

model.add(Dropout(0.25))

model.add(Dense(10, activation="softmax"))

# model.summary()

model.compile(optimizer='adam', loss = 'categorical_crossentropy',metrics=['accuracy'])

es =EarlyStopping(monitor='val_acc', min_delta=0.01, patience=4, verbose=1)
# print(os.getcwd())
# f = h5py.File("bestmodel.h5")
filepath = "K:\\sem_5\\DBMS\\project\\Automated_cheque_processing\\"

mc = ModelCheckpoint("./bestmodel", monitor="val_acc", verbose=1, save_best_only=True, mode='auto')
cb = [es,mc]

his = model.fit(x_train, y_train, epochs = 10, validation_split=0.3, callbacks=cb)
model.save("bestmodel.h5")

model_S = keras.models.load_model(filepath+"bestmodel.h5")


score = model_S.evaluate(X_test, Y_test)
print(score[1])


