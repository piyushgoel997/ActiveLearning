import time

import numpy as np
from scipy.stats import multivariate_normal

from DataUtils import gen_data, extend_array, sse, plot, shuffle_together
from Evaluation import calc_accuracy, roc, auc, compare_posterirors
from LogisticRegression import log_reg, log_reg_predict
from NeuralNetwork import nn, nn_predict
from SamplingMethods import random_sampling, uncertainty_sampling, query_by_committee2, query_by_committee1


def active_learning(X, Y, X_true, Y_true, true, learner, sampling=random_sampling, iters=100, metric=sse,
                    num_committee=1):
    print(sampling.__name__)
    [method, predictor] = learner
    indices = set()
    for a in range(4):
        indices.add(a)
    acc = []
    aucs = []
    comparison = []
    for j in range(iters):
        models = []
        for _ in range(num_committee):
            models.append(method(X[list(indices)], Y[list(indices)]))
        probs = predictor(models, X_true)
        true_probs = np.sum(probs, axis=0) / probs.shape[0]
        a = calc_accuracy(Y_true, true_probs)
        acc.append(a)
        print("Iter:", j, "-", a)
        aucs.append(auc(roc(Y_true, true_probs)))
        comparison.append(compare_posterirors(true, true_probs, metric))

        prev_len = len(indices)
        probs = predictor(models, X)
        indices.add(sampling(indices, probs))
        if len(indices) == prev_len:
            break
        del models
    return np.array(acc), np.array(aucs), np.array(comparison)


def run1(d, mu0, covar0, mu1, covar1, name="", num_exp=100):
    errors = [[], [], []]
    accuracies = [[], [], []]
    roc_aucs = [[], [], []]
    avg_samples = [0, 0]
    X_true, Y_true = gen_data(d, mu0, covar0, mu1, covar1, 10_000)
    pdf1 = multivariate_normal(mean=mu1, cov=covar1).pdf(X_true[:, 1:])
    pdf0 = multivariate_normal(mean=mu0, cov=covar0).pdf(X_true[:, 1:])
    true = pdf1 / (pdf0 + pdf1)
    for i in range(num_exp):
        print("\nExperiment number:", i + 1)
        exp_start_time = time.time()
        X, Y = gen_data(d, mu0, covar0, mu1, covar1, 10_000)
        acc, aucs, comparison = active_learning(X, Y, X_true, Y_true, true, random_sampling)
        # comparison1 = []
        # while len(comparison1) < 50:
        #     X, Y = gen_data(d, mu0, covar0, mu1, covar1, 1000)
        acc1, aucs1, comparison1 = active_learning(X, Y, X_true, Y_true, true, uncertainty_sampling)
        print("Num of Queries made for Uncertainty Sampling:", len(comparison1))
        avg_samples[0] += len(comparison1)
        # comparison2 = []
        # while len(comparison2) < 50:
        #     X, Y = gen_data(d, mu0, covar0, mu1, covar1, 1000)
        acc2, aucs2, comparison2 = active_learning(X, Y, X_true, Y_true, true, query_by_committee2, num_committee=11)
        print("Num of Queries made for Query by Committee:", len(comparison2))
        avg_samples[1] += len(comparison2)
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
    avg_samples = [a / num_exp for a in avg_samples]
    print()
    print("Average number of samples used for Uncertainty Sampling:", avg_samples[0])
    print("Average number of samples used for Query by Committee:", avg_samples[1])
    fn_names = ["Random Sampling", "Uncertainty Sampling", "Query by Committee"]
    plot(errors, name, "Posterior Diff (SSE)", fn_names)
    plot(accuracies, name, "Accuracy", fn_names)
    plot(roc_aucs, name, "ROC AUC", fn_names)


def run2(X, Y, true, learner, sampling_methods, num_committees, name="", num_exp=50, iters=100):
    X, Y, true = shuffle_together([X, Y, true])
    X_new = np.ones((X.shape[0], X.shape[1] + 1))
    X_new[:, 1:] = X
    X = X_new
    X_true = X[:10_000]
    Y_true = Y[:10_000]
    true = true[:10_000]
    num_queries = np.zeros((len(sampling_methods)))
    accuracy = []
    roc_auc = []
    posterior_diff = []
    for i in range(num_exp):
        print("\nExperiment number:", i + 1)
        exp_start_time = time.time()
        start = 10_000 + (10_000 * i)
        end = start + 10_000
        x, y = X[start:end], Y[start:end]
        exp_result = [active_learning(x, y, X_true, Y_true, true, learner, sampling=s, num_committee=c, iters=iters) for
                      s, c in
                      zip(sampling_methods, num_committees)]
        for j, (a, ra, pd) in enumerate(exp_result):
            if i == 0:
                accuracy.append(extend_array(a, iters))
                roc_auc.append(extend_array(ra, iters))
                posterior_diff.append(extend_array(pd, iters))
            else:
                accuracy[j] += extend_array(a, iters)
                roc_auc[j] += extend_array(ra, iters)
                posterior_diff[j] += extend_array(pd, iters)
        q = np.array([len(er[0]) for er in exp_result])
        num_queries += q
        print("Number of queries:")
        for s in range(len(sampling_methods)):
            print(sampling_methods[s].__name__, q[s])
        print("Time taken", time.time() - exp_start_time)

    for i in range(len(accuracy)):
        accuracy[i] /= num_exp
        roc_auc[i] /= num_exp
        posterior_diff[i] /= num_exp
    num_queries /= num_exp
    print()
    print("Average number of queries:")
    for s in range(len(sampling_methods)):
        print(sampling_methods[s].__name__, num_queries[s])
    fn_names = [f.__name__ for f in sampling_methods]
    plot(posterior_diff, name, "Posterior Diff (SSE)", fn_names)
    plot(accuracy, name, "Accuracy", fn_names)
    plot(roc_auc, name, "ROC AUC", fn_names)


start_time = time.time()

print("Starting with Neural Network")
run2(np.load("x.npy"), np.load("y.npy"), np.load("true.npy"), [nn, nn_predict],
     [random_sampling, uncertainty_sampling, query_by_committee1, query_by_committee2], [1, 1, 10, 10],
     name="NN_2Comp4D", num_exp=5, iters=100)

# print("Starting with Logistic Regression")
# run2(np.load("x.npy"), np.load("y.npy"), np.load("true.npy"), [log_reg, log_reg_predict],
#      [random_sampling, uncertainty_sampling, query_by_committee1, query_by_committee2], [1, 1, 10, 10],
#      name="LR_2Comp4D", num_exp=1)

print("Total time taken:", time.time() - start_time)

# d = 1
# mu0 = np.array([0])
# mu1 = np.array([1])
# covar0 = np.array([1]) * np.identity(d)
# covar1 = np.array([1]) * np.identity(d)
# run1(d, mu0, covar0, mu1, covar1, name="Less Separation", num_exp=100)

