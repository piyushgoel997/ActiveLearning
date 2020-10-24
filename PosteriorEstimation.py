import argparse
import time

import numpy as np

from DataUtils import normal_data_generator, extend_array, sse, plot, get_probability_distr
from Evaluation import calc_accuracy, roc, auc, compare_posterirors
# from LogisticRegression import log_reg, log_reg_predict
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
        # print("Iter:", j, "-", a)
        aucs.append(auc(roc(Y_true, true_probs)))
        comparison.append(compare_posterirors(true, true_probs, metric))

        prev_len = len(indices)
        probs = predictor(models, X)
        indices.add(sampling(indices, probs))
        if len(indices) == prev_len:
            break
        del models
    print("Final accuracy", acc[-1], ", Final AUC", aucs[-1],
          ", Final diff between the learned and the actual posterior", comparison[-1])
    return np.array(acc), np.array(aucs), np.array(comparison)


def run(X, Y, true, learner, sampling_methods, num_committees, name="", num_exp=50, iters=100):
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
                      s, c in zip(sampling_methods, num_committees)]
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--gendata', nargs='?', const=True, default=False)
    args = parser.parse_args()

    start_time = time.time()

    if args.gendata:
        print("Generating data")

        mu0, covar0, = np.array([1, 2, 3, 4]), np.array([1.6, 1.2, 1.5, 1]) * np.identity(4)
        mu1, covar1 = np.array([3, 3, 6, 1]), np.array([1, 2, 0.5, 1.9]) * np.identity(4)

        X, Y = normal_data_generator(4, 10_000_000, mu0, covar0, mu1, covar1)
        true = get_probability_distr(X, mu0, covar0, mu1, covar1)

        np.save("x.npy", X)
        np.save("y.npy", Y)
        np.save("true.npy", true)

        print("Time taken for data generation:", time.time() - start_time)
    else:
        X = np.load("x.npy")
        Y = np.load("y.npy")
        true = np.load("true.npy")
        print("Time taken for loading data:", time.time() - start_time)

    start_time = time.time()
    print("Starting with Neural Network")
    run(X, Y, true, [nn, nn_predict],
        [random_sampling, uncertainty_sampling, query_by_committee1, query_by_committee2], [1, 1, 10, 10],
        name="NN_2Comp4D", num_exp=5, iters=100)

    print("Starting with Logistic Regression")
    # run(X, Y, true, [log_reg, log_reg_predict],
    #     [random_sampling, uncertainty_sampling, query_by_committee1, query_by_committee2], [1, 1, 10, 10],
    #     name="LR_2Comp4D", num_exp=1)

    print("Total time taken:", time.time() - start_time)
