import numpy as np


def classify(predict, thresholds=[0, 0.5, 1.5, 2.5, 3.5]):
    predict = max(predict.item(), thresholds[0])
    for i in reversed(range(len(thresholds))):
        if predict >= thresholds[i]:
            return i


def quadratic_weighted_kappa(conf_mat):
    assert conf_mat.shape[0] == conf_mat.shape[1]
    cate_num = conf_mat.shape[0]

    # Quadratic weighted matrix
    weighted_matrix = np.zeros((cate_num, cate_num))
    for i in range(cate_num):
        for j in range(cate_num):
            weighted_matrix[i][j] = 1 - float(((i - j)**2) / ((cate_num - 1)**2))

    # Expected matrix
    ground_truth_count = np.sum(conf_mat, axis=1)
    pred_count = np.sum(conf_mat, axis=0)
    expected_matrix = np.outer(ground_truth_count, pred_count)

    # Normalization
    conf_mat = conf_mat / conf_mat.sum()
    expected_matrix = expected_matrix / expected_matrix.sum()

    observed = (conf_mat * weighted_matrix).sum()
    expected = (expected_matrix * weighted_matrix).sum()
    return (observed - expected) / (1 - expected)


def accuracy(predictions, targets, c_matrix=None):
    predictions = predictions.data
    targets = targets.data
    for i, p in enumerate(predictions):
        predictions[i] = classify(p)

    # update confusion matrix
    if c_matrix is not None:
        for i, p in enumerate(predictions):
            c_matrix[int(targets[i])][int(p)] += 1

    correct = (predictions == targets).sum().item()
    return correct / predictions.size(0)


if __name__ == "__main__":
    conf_mat = np.array([
        [37, 8, 5, 0, 0],
        [8, 32, 8, 2, 0],
        [6, 6, 31, 5, 2],
        [1, 1, 5, 39, 4],
        [1, 1, 4, 10, 34]
    ])
    print(quadratic_weighted_kappa(conf_mat))
