import random

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal


def shuffle_together(arrays):
    ind = list(range(len(arrays[0])))
    random.shuffle(ind)
    return [X[ind] for X in arrays]


def normal_data_generator(d, n, mu0, covar0, mu1, covar1, shuffle=True):
    X = np.concatenate((np.random.multivariate_normal(mu0, covar0, int(n / 2)),
                        np.random.multivariate_normal(mu1, covar1, int(n / 2))))
    X_new = np.ones((X.shape[0], X.shape[1] + 1))
    X_new[:, 1:] = X
    X = X_new
    Y = np.concatenate((np.zeros((int(n / 2),)), np.ones((int(n / 2),))))
    if shuffle:
        X, Y = shuffle_together((X, Y))
    return X, Y


def get_probability_distr(X, mu0, covar0, mu1, covar1):
    pdf1 = multivariate_normal(mean=mu1, cov=covar1).pdf(X[:, 1:])
    pdf0 = multivariate_normal(mean=mu0, cov=covar0).pdf(X[:, 1:])
    return pdf1 / (pdf0 + pdf1)


def extend_array(arr, n):
    idx = [min(i, len(arr) - 1) for i in range(n)]
    return arr[idx]


def max_metric(errors):
    return max(errors)


def avg_metric(errors):
    return sum(errors) / len(errors)


def sse(errors):
    return sum([x ** 2 for x in errors]) / len(errors)


def majority_voting(probs):
    return np.round(np.sum(np.round(probs), axis=0) / probs.shape[0])


def average_voting(probs):
    return np.round(np.sum(probs, axis=0) / probs.shape[0])


def plot(fn, name, plt_name, fn_names):
    for f, n in zip(fn, fn_names):
        plt.plot(f, label=n)
    plt.xlabel("Number of training examples")
    plt.ylabel(plt_name)
    plt.title(name + " - " + plt_name)
    plt.legend()
    plt.savefig(name + " " + plt_name + ".jpg")
    plt.show()
