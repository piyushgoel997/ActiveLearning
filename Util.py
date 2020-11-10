import numpy as np
import matplotlib.pyplot as plt


def smooth(arr, n):
    curr = 0
    result = []
    for i in range(len(arr)):
        curr += arr[i]
        if i < n:
            result.append(curr/(i + 1))
        else:
            curr -= arr[i-n]
            result.append(curr/n)
    return result


f = open("saved2/output2.txt").read()
lines = f.split("\n")

acc = np.zeros((4, 100))
auc = np.zeros((4, 100))
pd = np.zeros((4, 100))

names = ["random_sampling", "uncertainty_sampling_hard", "uncertainty_sampling_soft", "query_by_committee2"]
# names = ["random_sampling", "uncertainty_sampling", "query_by_committee1", "query_by_committee2"]
i = -1

for l in lines:
    words = l.split()
    if len(words) == 0:
        continue
    print(words[0])
    try:
        i = names.index(words[0])
        continue
    except:
        if not words[0] == "Iter:":
            continue
        acc[i][int(words[1])] += float(words[3])
        auc[i][int(words[1])] += float(words[4])
        pd[i][int(words[1])] += float(words[5])

acc /= 10
auc /= 10
pd /= 10


def make_plot(arr, names):
    for y, n in zip(arr, names):
        plt.plot(smooth(y, 5), label=n)
    plt.legend()
    plt.savefig("saved2/smoothed_pd2.jpg")
    plt.show()


# make_plot(auc, names)
make_plot(pd, names)
