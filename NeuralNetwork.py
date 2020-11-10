from keras.layers import Dense
from keras.models import Sequential
import numpy as np


def get_model():
    model = Sequential()
    model.add(Dense(16, activation='relu', kernel_initializer='random_normal'))
    model.add(Dense(32, activation='relu', kernel_initializer='random_normal'))
    model.add(Dense(64, activation='relu', kernel_initializer='random_normal'))
    model.add(Dense(32, activation='relu', kernel_initializer='random_normal'))
    model.add(Dense(16, activation='relu', kernel_initializer='random_normal'))
    model.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def nn(X, Y):
    model = get_model()
    model.fit(X, Y, epochs=100, verbose=0, batch_size=256)
    return model


def nn_predict(models, X):
    pred = [model.predict(X) for model in models]
    pred = np.array(pred).reshape((len(models), len(X)))
    return pred
