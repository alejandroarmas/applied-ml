import numpy as np

def mse(actual, predicted):
    actual = np.array(actual)
    predicted = np.array(predicted)
    differences = actual - predicted
    squared_differences = differences ** 2
    return squared_differences.mean()


def d_mse_r_slope(actual_x, actual_y, predicted):
    n = len(actual_x)
    assert n == len(predicted)
    assert n == len(actual_y)

    differences = actual_y - predicted
    return 2 * np.sum(differences) / n


def d_mse_r_intercept(actual_y, predicted):
    n = len(actual_y)
    assert n == len(predicted)

    differences = actual_y - predicted
    return 2 * np.sum(differences) / n