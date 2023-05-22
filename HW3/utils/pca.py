import numpy as np


def pca(X: np.array, k: int):

    # Step 1: Standardize the data
    X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    # Step 2: Calculate the covariance matrix
    covariance_matrix = np.cov(X_std.T)


    # Step 3: Calculate the eigenvectors and eigenvalues
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # Step 4: Sort the eigenvectors by their eigenvalues
    idx = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, idx]


    # Step 5: Select the top k eigenvectors and form a matrix
    top_k_eigenvectors = eigenvectors[:, :k]

    # Step 6: Transform the data into the new subspace
    new_data = X_std.dot(top_k_eigenvectors)
    return new_data