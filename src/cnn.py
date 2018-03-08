import numpy as np
np.random.seed(123)
import keras
import h5py
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import Adadelta
from keras.models import load_model
from keras import optimizers
from src.data_utils_fer2013 import get_FER2013_data


model = Sequential()
model.add(Convolution2D(20, 5, 5, border_mode="same",
			input_shape=(48, 48, 1)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Convolution2D(50, 5, 5, border_mode="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(500))
model.add(Activation("relu"))

		# softmax classifier
model.add(Dense(7))
model.add(Activation("softmax"))
opt = optimizers.SGD(lr=0.01)
model.compile(loss='categorical_crossentropy',
            optimizer=opt,
            metrics=['accuracy'])


data = get_FER2013_data(20000, 8000, 9, True)

#fit model
X_train = data['X_train'].reshape(data['X_train'].shape[0], 48, 48, 1)
X_val = data['X_val'].reshape(data['X_val'].shape[0], 48, 48, 1)
Y_train = np_utils.to_categorical(data['y_train'], 7)
Y_val = np_utils.to_categorical(data['y_val'], 7)


model.fit(X_train,
          Y_train,
          batch_size=32,
          nb_epoch=100,
          verbose=1)

#test model

scores = model.evaluate(X_val,
                        Y_val,
                        verbose=0)
print(scores)

#model.save('model.h5')