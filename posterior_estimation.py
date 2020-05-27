import math
import random
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal


def sigmoid(X, w):
    return 1 / (1 + np.exp(-np.dot(X, w)))


def log_reg(X, Y, eta=1, max_steps=200):
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
    pr = probs[0]
    for i in range(len(pr)):
        if i in indices:
            continue
        diff = abs(0.5 - pr[i])
        if diff < min_diff:
            min_diff = diff
            ind = i
    return ind


def query_by_committee(indices, probs):
    # using vote entropy
    # each row of probs is the probs of each committee member.
    ind = -1
    max_entropy = 0
    C = probs.shape[0]
    for i in range(probs.shape[1]):
        if i in indices:
            continue
        vote0 = 0
        vote1 = 0
        for p in probs[:, i]:
            if p > 0.5:
                vote1 += 1
            else:
                vote0 += 1
        entropy = 0
        if vote0 != 0:
            entropy -= (vote0 / C) * math.log(vote0 / C)
        if vote1 != 0:
            entropy -= (vote1 / C) * math.log(vote1 / C)

        if entropy > max_entropy:
            max_entropy = entropy
            ind = i
    return ind


def majority_voting(probs):
    return np.round(np.sum(np.round(probs), axis=0) / probs.shape[0])


def active_learning(X, Y, X_true, Y_true, true, sampling=random_sampling, iters=100, metric=sse,
                    num_committee=1):
    indices = set()
    for a in range(4):
        indices.add(a)
    acc = []
    aucs = []
    comparison = []
    for j in range(iters):
        # print(len(indices))
        w = []
        for _ in range(num_committee):
            w.append(log_reg(X[list(indices)], Y[list(indices)]))
        w = np.array(w)
        # w = log_reg(X[list(indices)], Y[list(indices)])
        # Majority voting is used to decide the outcome of the committee.
        true_probs = majority_voting(sigmoid(X_true, w.T).T)
        acc.append(calc_accuracy(Y_true, true_probs))
        aucs.append(auc(roc(Y_true, true_probs)))
        comparison.append(compare_posterirors(true, true_probs, metric))

        prev_len = len(indices)
        probs = sigmoid(X, w.T).T
        indices.add(sampling(indices, probs))
        if len(indices) == prev_len:
            break
        # final_result = sigmoid(X_true, w)
    return acc, aucs, comparison


def extend_array(arr, n):
    idx = [min(i, len(arr) - 1) for i in range(n)]
    return arr[idx]


def plot(fn, name, plt_name, fn_names):
    for f, n in zip(fn, fn_names):
        plt.plot(f, label=n)
    plt.xlabel("Number of training examples")
    plt.ylabel(plt_name)
    plt.title(name + " - " + plt_name)
    plt.legend()
    plt.savefig(name + " " + plt_name + ".jpg")
    plt.show()


def run(d, mu0, covar0, mu1, covar1, name="", num_exp=100):
    errors = [[], [], []]
    accuracies = [[], [], []]
    roc_aucs = [[], [], []]
    X_true, Y_true = gen_data(d, mu0, covar0, mu1, covar1, 10_000)
    pdf1 = multivariate_normal(mean=mu1, cov=covar1).pdf(X_true[:, 1:])
    pdf0 = multivariate_normal(mean=mu0, cov=covar0).pdf(X_true[:, 1:])
    true = pdf1 / (pdf0 + pdf1)
    for i in range(num_exp):
        print("Experiment number:", i + 1)
        exp_start_time = time.time()
        X, Y = gen_data(d, mu0, covar0, mu1, covar1, 200)
        acc, aucs, comparison = active_learning(X, Y, X_true, Y_true, true, random_sampling)
        acc1, aucs1, comparison1 = active_learning(X, Y, X_true, Y_true, true, uncertainty_sampling)
        print("Num of Queries made for Uncertainty Sampling:", len(comparison1))
        acc2, aucs2, comparison2 = active_learning(X, Y, X_true, Y_true, true, query_by_committee, num_committee=11)
        print("Num of Queries made for Query by Committee:", len(comparison2))
        if len(errors[0]) == 0:
            errors[0] = np.array(comparison)
            errors[1] = extend_array(np.array(comparison1), len(errors[0]))
            errors[2] = extend_array(np.array(comparison2), len(errors[0]))
            accuracies[0] = np.array(acc)
            accuracies[1] = extend_array(np.array(acc1), len(accuracies[0]))
            accuracies[2] = extend_array(np.array(acc2), len(accuracies[0]))
            roc_aucs[0] = np.array(aucs)
            roc_aucs[1] = extend_array(np.array(aucs1), len(roc_aucs[0]))
            roc_aucs[2] = extend_array(np.array(aucs2), len(roc_aucs[0]))
        else:
            errors[0] += np.array(comparison)
            errors[1] += extend_array(np.array(comparison1), len(errors[0]))
            errors[2] += extend_array(np.array(comparison2), len(errors[0]))
            accuracies[0] += np.array(acc)
            accuracies[1] += extend_array(np.array(acc1), len(accuracies[0]))
            accuracies[2] += extend_array(np.array(acc2), len(accuracies[0]))
            roc_aucs[0] += np.array(aucs)
            roc_aucs[1] += extend_array(np.array(aucs1), len(roc_aucs[0]))
            roc_aucs[2] += extend_array(np.array(aucs2), len(roc_aucs[0]))
        print("Time taken:", time.time() - exp_start_time)

    errors = [e / num_exp for e in errors]
    accuracies = [a / num_exp for a in accuracies]
    roc_aucs = [r / num_exp for r in roc_aucs]

    fn_names = ["Random Sampling", "Uncertainty Sampling", "Query By Committee"]
    plot(errors, name, "Posterior Diff (SSE)", fn_names)
    plot(accuracies, name, "Accuracy", fn_names)
    plot(roc_aucs, name, "ROC AUC", fn_names)


start_time = time.time()
d = 1
mu0 = np.array([1])
mu1 = np.array([2])
covar0 = np.array([2]) * np.identity(d)
covar1 = np.array([2]) * np.identity(d)
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
# mu1 = np.array([1.1, 1.8, 3.3, 3.6])
# covar0 = np.array([1.5, 1.2, .5, 1]) * np.identity(d)
# covar1 = np.array([1, 1.8, 0.9, 1.9]) * np.identity(d)
# run(d, mu0, covar0, mu1, covar1, name="4D", num_exp=2)
