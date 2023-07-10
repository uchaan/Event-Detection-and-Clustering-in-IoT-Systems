import pandas as pd
import numpy as np


def ssa(data, window_size, num_components):
    # Construct trajectory matrix
    N = len(data)
    L = N - window_size + 1
    X = np.zeros((window_size, L))

    for i in range(L):
        X[:, i] = data[i : i + window_size]

    # Singular value decomposition
    U, S, V = np.linalg.svd(X)

    # Construct the time-delayed embedding matrix
    embedding_matrix = np.dot(U[:, :num_components], np.diag(S[:num_components]))

    return embedding_matrix
