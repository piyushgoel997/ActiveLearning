import random
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal


def sigmoid(X, w):
    return 1 / (1 + np.exp(-np.dot(X, w)))


def log_reg(X, Y, eta=1, max_steps=100):
    # w = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X), X)), np.transpose(X)), Y)
    w = np.random.rand(X.shape[1], )
    step = 0
    Y = np.array(Y)
    while step < max_steps:
        p = sigmoid(X, w)
        inv = np.dot(np.transpose(X), np.dot(np.diagflat(np.multiply(p, 1 - p)), X))
        if np.linalg.det(inv) == 0:
            break
        temp = np.linalg.inv(inv)
        w_new = w + eta * np.dot(temp, np.dot(np.transpose(X), Y - p))
        w = w_new
        step += 1
    return w


def shuffle_together(arrays):
    ind = list(range(len(arrays[0])))
    random.shuffle(ind)
    return [X[ind] for X in arrays]


def calc_accuracy(y_test, test_pred, thresh=0.5):
    correct = 0
    y_test_ = list(y_test)
    test_pred_ = list(test_pred)
    for i in range(y_test.shape[0]):
        if test_pred_[i] > thresh:
            if y_test_[i] == 1:
                correct += 1
        else:
            if y_test_[i] == 0:
                correct += 1
    return correct / y_test.shape[0]


def roc(y_test, probs):
    r = [(0, 0)]
    prev_c = None
    zeros = 0
    ones = 0
    total_ones = sum(list(y_test))
    total_zeros = len(list(y_test)) - total_ones
    for _, c in sorted(zip(probs, list(y_test)), reverse=True):
        if not (prev_c is None or prev_c == c):
            tpr = ones / total_ones
            fpr = zeros / total_zeros
            r.append((fpr, tpr))
        if c == 0:
            zeros += 1
        else:
            ones += 1
        prev_c = c
    r.append((1, 1))
    return r


def auc(r):
    area = 0
    prev = 0
    for a, b in r:
        area += (b - prev) * (1 - a)
        prev = b
    return area


def compare_posterirors(true, pred, distance_metric):
    errors = [abs(t - p) for t, p in zip(true, pred)]
    return distance_metric(errors)


def max_metric(errors):
    return max(errors)


def avg_metric(errors):
    return sum(errors) / len(errors)


def sse(errors):
    return sum([x ** 2 for x in errors]) / len(errors)


def gen_data(d, mu0, covar0, mu1, covar1, n):
    X = np.concatenate((np.random.multivariate_normal(mu0, covar0, int(n / 2)),
                        np.random.multivariate_normal(mu1, covar1, int(n / 2))))
    X_new = np.ones((X.shape[0], X.shape[1] + 1))
    X_new[:, 1:] = X
    X = X_new
    Y = np.concatenate((np.zeros((int(n / 2),)), np.ones((int(n / 2),))))
    X, Y = shuffle_together((X, Y))
    return X, Y


def random_sampling(indices, probs):
    return len(indices) + 1


def uncertainty_sampling(indices, probs):
    ind = -1
    min_diff = 0.5
    for i in range(len(probs)):
        if i in indices:
            continue
        diff = abs(0.5 - probs[i])
        if diff < min_diff:
            min_diff = diff
            ind = i
    return ind


def active_learning(X, Y, X_true, Y_true, true, sampling=random_sampling, iters=100):
    indices = set()
    for a in range(4):
        indices.add(a)
    acc = []
    aucs = []
    comparison = []
    for j in range(iters):
        # print(len(indices))
        w = log_reg(X[list(indices)], Y[list(indices)])
        true_probs = sigmoid(X_true, w)
        acc.append(calc_accuracy(Y_true, true_probs))
        aucs.append(auc(roc(Y_true, true_probs)))
        comparison.append(compare_posterirors(true, true_probs, max_metric))

        prev_len = len(indices)
        probs = sigmoid(X, w)
        indices.add(sampling(indices, probs))
        if len(indices) == prev_len:
            break
        final_result = sigmoid(X_true, w)
    return acc, aucs, comparison, final_result


def extend_array(arr, n):
    idx = [min(i, len(arr) - 1) for i in range(n)]
    return arr[idx]


def plot(rs, us, name, plt_name):
    plt.plot(rs, label="Random Sampling")
    plt.plot(us, label="Uncertainty Sampling")
    plt.xlabel("Number of training examples")
    plt.ylabel(plt_name)
    plt.title(name + " - " + plt_name)
    plt.legend()
    plt.savefig(name + " " + plt_name + ".jpg")
    plt.show()


def run(d, mu0, covar0, mu1, covar1, name="", num_exp=100):
    errors = []
    errors1 = []
    accuracies = []
    accuracies1 = []
    roc_aucs = []
    roc_aucs1 = []
    X_true, Y_true = gen_data(d, mu0, covar0, mu1, covar1, 10_000)
    pdf1 = multivariate_normal(mean=mu1, cov=covar1).pdf(X_true[:, 1:])
    pdf0 = multivariate_normal(mean=mu0, cov=covar0).pdf(X_true[:, 1:])
    true = pdf1 / (pdf0 + pdf1)
    for i in range(num_exp):
        print("Experiment number:", i + 1)
        exp_start_time = time.time()
        X, Y = gen_data(d, mu0, covar0, mu1, covar1, 200)
        acc, aucs, comparison, final_result = active_learning(X, Y, X_true, Y_true, true, random_sampling)
        acc1, aucs1, comparison1, final_result1 = active_learning(X, Y, X_true, Y_true, true, uncertainty_sampling)
        print(len(comparison1))
        if len(errors) == 0:
            errors = np.array(comparison)
            errors1 = extend_array(np.array(comparison1), len(errors))
            accuracies = np.array(acc)
            accuracies1 = extend_array(np.array(acc1), len(accuracies))
            roc_aucs = np.array(aucs)
            roc_aucs1 = extend_array(np.array(aucs1), len(roc_aucs))
        else:
            errors += np.array(comparison)
            errors1 += extend_array(np.array(comparison1), len(errors))
            accuracies += np.array(acc)
            accuracies1 += extend_array(np.array(acc1), len(accuracies))
            roc_aucs += np.array(aucs)
            roc_aucs1 += extend_array(np.array(aucs1), len(roc_aucs))
        print("Time taken:", time.time() - exp_start_time)
    errors /= num_exp
    errors1 /= num_exp
    accuracies /= num_exp
    accuracies1 /= num_exp
    roc_aucs /= num_exp
    roc_aucs1 /= num_exp

    # plotting the final posterior learned in the last experiment.
    # plt.plot(X_true[:, 1:], final_result, '.', markersize=1, label="Random Sampling")
    # plt.plot(X_true[:, 1:], final_result1, '.', markersize=1, label="Uncertainty Sampling")
    # plt.plot(X_true[:, 1:], true, '.', markersize=1, label="True Posterior")
    # plt.title(name + " - Posteriors")
    # plt.legend()
    # plt.savefig(name + " - Posteriors.jpg")
    # plt.show()

    plot(errors, errors1, name, "Posterior Diff (SSE)")
    plot(accuracies, accuracies1, name, "Accuracy")
    plot(roc_aucs, roc_aucs1, name, "ROC AUC")

    # plt.plot(acc, label="Random Sampling")
    # plt.plot(acc1, label="Uncertainty Sampling")
    # plt.xlabel("Number of training examples")
    # plt.ylabel("Accuracy")
    # plt.title(name + " - Accuracy")
    # plt.legend()
    # plt.savefig(name + " - Accuracy.jpg")
    # plt.show()
    #
    # plt.plot(aucs, label="Random Sampling")
    # plt.plot(aucs1, label="Uncertainty Sampling")
    # plt.xlabel("Number of training examples")
    # plt.ylabel("ROC AUC")
    # plt.title(name + " - ROC AUC")
    # plt.legend()
    # plt.savefig(name + " - ROC AUC.jpg")
    # plt.show()
    #
    # plt.plot(comparison, label="Random Sampling")
    # plt.plot(comparison1, label="Uncertainty Sampling")
    # plt.xlabel("Number of training examples")
    # plt.ylabel("Posterior Diff - SSE")
    # plt.title(name + " - Posterior Diff")
    # plt.legend()
    # plt.savefig(name + " - Posterior Diff.jpg")
    # plt.show()
    #
    # plt.plot(X_true[:, 1:], final_result, '.', markersize=1, label="Random Sampling")
    # plt.plot(X_true[:, 1:], final_result1, '.', markersize=1, label="Uncertainty Sampling")
    # plt.plot(X_true[:, 1:], true, '.', markersize=1, label="True Posterior")
    # plt.title(name + " - Posteriors")
    # plt.legend()
    # plt.savefig(name + " - Posteriors.jpg")
    # plt.show()


start_time = time.time()
d = 1
mu0 = np.array([1])
mu1 = np.array([2])
covar0 = np.array([1.5]) * np.identity(d)
covar1 = np.array([1]) * np.identity(d)
run(d, mu0, covar0, mu1, covar1, name="Less Separation", num_exp=100)
print("Total time taken:", time.time() - start_time)

# d = 1
# mu0 = np.array([1])
# mu1 = np.array([5])
# covar0 = np.array([1.5]) * np.identity(d)
# covar1 = np.array([1]) * np.identity(d)
# run(*gen_data(d, mu0, covar0, mu1, covar1), name="More separation")

# d = 4
# mu0 = np.array([1, 2, 3, 4])
# mu1 = np.array([2.5, 3.5, 4.5, 5.5])
# covar0 = np.array([1.5, 1.2, .5, 1]) * np.identity(d)
# covar1 = np.array([1, 1.8, 0.9, 1.9]) * np.identity(d)
# run(*gen_data(d, mu0, covar0, mu1, covar1), name="4D")
