import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
import time


# Total Original Time:  97.91s
# Total new Time:       48.07s

def large_random_matrix(x, y):
    np.random.seed(1)

    X = np.random.random([x, y])
    X = pd.DataFrame(X).mask(X < 0.1)

    imputer = KNNImputer(n_neighbors=5)

    imputer.fit_transform(X)


def test_1K_by_100():
    # The original implementation takes 0.4349205493927002 seconds
    # The new implementation takes      0.3454127311706543 seconds
    start = time.time()
    large_random_matrix(1000, 100)
    end = time.time()

    print("\nTime: ", end-start)

    assert end-start > 0


def test_2K_by_500():
    # The original implementation takes 6.568829774856567 seconds
    # The new implementation takes      3.880154848098755 seconds
    start = time.time()
    large_random_matrix(2000, 500)
    end = time.time()

    print("\nTime: ", end-start)

    assert end-start > 0


def test_1K_by_5K():
    # The original implementation takes 17.162961959838867 seconds
    # The new implementation takes      12.638985633850098 seconds
    # Sofia time:                       10.330341100692749 seconds
    start = time.time()
    large_random_matrix(1000, 5000)
    end = time.time()

    print("\nTime: ", end-start)

    assert end - start > 0
    
def test_10K_by_4():
    # The original implementation takes 2.436619281768799 seconds
    # The new implementation takes      1.8823344707489014 seconds
    start = time.time()
    large_random_matrix(10000, 4)
    end = time.time()

    print("\nTime: ", end-start)

    assert end-start > 0


def test_3K_by_1K():
    # The original implementation takes 28.86723804473877 seconds
    # The new implementation takes      18.428542137145996 seconds
    start = time.time()
    large_random_matrix(3000, 1000)
    end = time.time()

    print("\nTime: ", end-start)

    assert end-start > 0


# def test_10K_by_5K ():
#     # The original implementation takes ___________ seconds
#     # The new implementation takes      1471.1210680007935 seconds
#     start = time.time()
#     large_random_matrix(10000, 5000)
#     end = time.time()
#
#     print("\nTime: ", end-start)
#
#     assert end-start > 0
