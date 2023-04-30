import numpy as np


def unison_shuffled_copies(a, b):
    # assert len(a) == len(b)
    # p = np.random.permutation(len(a))
    # return a[p], b[p]
    indices = np.arange(a.shape[0])
    np.random.shuffle(indices)

    return a[indices], b[indices]