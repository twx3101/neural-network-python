import numpy as np
np.random.seed(123)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils

from data_utils_fer2013 import get_FER2013_data

#create model

model = Sequential()
model.add(Convolution2D(32, 3, 3,
                        activation="relu",
                        input_shape=(1,48,48)))
model.add(Convolution2D(32,3,3,activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation="softmax"))
model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=['accuracy'])

#load data

data = get_FER2013_data(49000, 1000, 1000, True)

#fit model

model.fit(data['X_train'],
          data['Y_train'],
          batch_size=32,
          nb_epoch=10,
          verbose=1)

#test model

scores = model.evaluate(data['X_test'],
                        data['Y_train'],
                        verbose=0)
