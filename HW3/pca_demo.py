import numpy as np
from utils.pca import pca 

def main():
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(X)
    Z = pca(X, 1)
    print(Z)


if __name__ == '__main__':
    main() 