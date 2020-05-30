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
