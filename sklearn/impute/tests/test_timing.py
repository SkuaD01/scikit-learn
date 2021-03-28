import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
import time



def large_random_matrix(x, y):
    np.random.seed(1)

    X = np.random.random([x, y])
    X = pd.DataFrame(X).mask(X < 0.1)

    imputer = KNNImputer(n_neighbors=5)

    imputer.fit_transform(X)


def test_1K_by_100():
    # The original implementation takes over 0.42 seconds
    start = time.time()
    large_random_matrix(1000, 100)
    end = time.time()

    print("\nTime: ", end-start)

    assert end-start > 0.1


def test_2K_by_500():
    # The original implementation takes over 8.05 seconds
    start = time.time()
    large_random_matrix(2000, 500)
    end = time.time()

    print("\nTime: ", end-start)

    assert end-start > 5


def test_K_by_K():
    # The original implementation takes over 33.48 seconds
    start = time.time()
    large_random_matrix(1000, 5000)
    end = time.time()

    print("\nTime: ", end-start)

    assert end-start > 0


def test_3K_by_1K():
    # The original implementation takes over 33.48 seconds
    start = time.time()
    large_random_matrix(3000, 1000)
    end = time.time()

    print("\nTime: ", end-start)

    assert end-start > 30


def test_10K_by_100():
    # The original implementation takes over 49.96 seconds
    start = time.time()
    large_random_matrix(10000, 100)
    end = time.time()

    print("\nTime: ", end-start)

    assert end-start > 45
