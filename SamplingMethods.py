import math
import numpy as np


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


def query_by_committee1(indices, probs):
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


def query_by_committee2(indices, probs):
    # using Average KL Divergence
    # each row of probs is the probs of each committee member.
    ind = -1
    max_kld = 0
    C = probs.shape[0]
    for i in range(probs.shape[1]):
        if i in indices:
            continue
        pc = np.sum(probs[:, i])/probs.shape[0]
        kl_div = 0
        for p in probs[:, i]:
            if p == 0:
                continue
            kl_div += p*math.log(p/pc)
        if kl_div > max_kld:
            max_kld = kl_div
            ind = i
    return ind
