import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
import time
import profile


# Total Original Time:  97.91s
# Total new Time:       48.07s

def gen_matrix(x, y, p=0.1):
    np.random.seed(1)
    X = np.random.random([x, y])
    X = pd.DataFrame(X).mask(X < 0.1)
    return X


def run_test(X, old=False):
    start = time.time()
    if (old):
        imputer = KNNImputer(n_neighbors=5, n_jobs=1)
    else:
        imputer = KNNImputer(n_neighbors=5)
    imputer.fit_transform(X)
    end = time.time()

    return start,end


def test_1K_by_100():
    X = gen_matrix(1000,100)

    start, end = run_test(X)
    old_start, old_end = run_test(X, True)

    print("\nOld time:", round((old_end-old_start),4) ,", New time:", round((end-start),4),",", str(round(((old_end-old_start)/(end-start) - 1)*100, 2))+"% increase")

    assert (end-start) < (old_end-old_start)


def test_2K_by_500():
    X = gen_matrix(2000, 500)

    start, end = run_test(X)
    old_start, old_end = run_test(X, True)

    print("\nOld time:", round((old_end-old_start),3),", New time:", round((end-start),3),",", str(round(((old_end-old_start)/(end-start) - 1)*100, 2))+"% increase")

    assert (end-start) < (old_end-old_start)


def test_1K_by_5K():
    X = gen_matrix(1000, 5000)

    start, end = run_test(X)
    old_start, old_end = run_test(X, True)

    print("\nOld time:", round((old_end-old_start),3),", New time:", round((end-start),3),",", str(round(((old_end-old_start)/(end-start) - 1)*100, 2))+"% increase")

    assert (end-start) < (old_end-old_start)


def test_10K_by_4():
    X = gen_matrix(10000, 4)

    start, end = run_test(X)
    old_start, old_end = run_test(X, True)

    print("\nOld time:", round((old_end-old_start),3),", New time:", round((end-start),3),",", str(round(((old_end-old_start)/(end-start) - 1)*100, 2))+"% increase")

    assert (end-start) < (old_end-old_start)


def test_3K_by_1K():
    X = gen_matrix(3000, 1000)

    start, end = run_test(X)
    old_start, old_end = run_test(X, True)

    print("\nOld time:", round((old_end-old_start),3),", New time:", round((end-start),3),",", str(round(((old_end-old_start)/(end-start) - 1)*100, 2))+"% increase")

    assert (end-start) < (old_end-old_start)


def test_10K_by_100():
    X = gen_matrix(10000, 100)

    start, end = run_test(X)
    old_start, old_end = run_test(X, True)

    print("\nOld time:", round((old_end-old_start),3),", New time:", round((end-start),3),",", str(round(((old_end-old_start)/(end-start) - 1)*100, 2))+"% increase")

    assert (end-start) < (old_end-old_start)