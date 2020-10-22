import numpy as np


def sigmoid(X, w):
    return 1 / (1 + np.exp(-np.dot(X, w)))


def log_reg(X, Y, eta=1, max_steps=500):
    # w = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X), X)), np.transpose(X)), Y)
    w = np.random.rand(X.shape[1], 1)
    step = 0
    Y = np.reshape(Y, (Y.shape[0], 1))
    while step < max_steps:
        p = sigmoid(X, w)
        inv = np.dot(np.transpose(X), np.dot(np.diagflat(np.multiply(p, 1 - p)), X))
        if np.linalg.det(inv) == 0:
            break
        temp = np.linalg.inv(inv)
        w_new = w + eta * np.dot(temp, np.dot(np.transpose(X), Y - p))
        w = w_new
        step += 1
    return w.reshape((X.shape[1],))


def log_reg_predict(ws, X):
    w = np.array(ws)
    return sigmoid(X, w.T).T
