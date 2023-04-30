import numpy as np
from utils.shuffle import unison_shuffled_copies


def pearson_corr_coef(x, y):
    n = len(x)
    assert n == len(y)
    numerator = n * np.sum(x * y) - np.sum(x) * np.sum(y)
    denominator = np.sqrt((n * np.sum(x ** 2) - np.sum(x) ** 2) * (n * np.sum(y ** 2) - np.sum(y) ** 2))
    return numerator / denominator
