import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
import time
import profile
import pytest


# Total Original Time:  97.91s
# Total new Time:       48.07s

perc = [0,0.1,0.5,0.9,1]
small_n = [100,300,500]
large_n = [1000,3000,5000]

def gen_matrix(x, y, p):
    np.random.seed(1)
    X = np.random.random([x, y])
    X = pd.DataFrame(X).mask(X <= p)
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

def relative_assert(new, old):
    assert pytest.approx(new).__lt__(pytest.approx(old, abs=1))

def output_res(old_end,old_start,end,start):
    print("\nOld time:", round((old_end-old_start),4) ,", New time:", round((end-start),4),",", str(round(((old_end-old_start)/(end-start) - 1)*100, 2))+"% improved")

@pytest.mark.parametrize("n,p",[(n, p) for p in perc for n in small_n])
def test_time_1_by_n(n,p):
    X = gen_matrix(1, n, p)

    start, end = run_test(X)
    old_start, old_end = run_test(X, old=True)

    output_res(old_end,old_start,end,start)

    relative_assert(end-start, old_end-old_start)

@pytest.mark.parametrize("N,p",[(N, p) for p in perc for N in large_n])
def test_time_1_by_N(N,p):
    X = gen_matrix(1, N, p)

    start, end = run_test(X)
    old_start, old_end = run_test(X, old=True)

    output_res(old_end,old_start,end,start)

    relative_assert(end-start, old_end-old_start)

@pytest.mark.parametrize("n,p",[(n, p) for p in perc for n in small_n])
def test_time_n_by_1(n,p):
    X = gen_matrix(n, 1, p)

    start, end = run_test(X)
    old_start, old_end = run_test(X, old=True)

    output_res(old_end,old_start,end,start)

    relative_assert(end-start, old_end-old_start)

@pytest.mark.parametrize("N,p",[(N, p) for p in perc for N in large_n])
def test_time_N_by_1(N,p):
    X = gen_matrix(N, 1, p)

    start, end = run_test(X)
    old_start, old_end = run_test(X, old=True)

    output_res(old_end,old_start,end,start)

    relative_assert(end-start, old_end-old_start)

@pytest.mark.parametrize("n1,n2,p",[(n1,n2,p) for p in perc for n1 in small_n for n2 in small_n])
def test_time_n_by_n(n1,n2,p):
    X = gen_matrix(n1, n2, p)

    start, end = run_test(X)
    old_start, old_end = run_test(X, old=True)

    output_res(old_end,old_start,end,start)

    relative_assert(end-start, old_end-old_start)

@pytest.mark.parametrize("n,N,p",[(n,N,p) for p in perc for n in small_n for N in large_n])
def test_time_n_by_N(n,N,p):
    X = gen_matrix(n, N, p)

    start, end = run_test(X)
    old_start, old_end = run_test(X, old=True)

    output_res(old_end,old_start,end,start)

    relative_assert(end-start, old_end-old_start)

@pytest.mark.parametrize("N,n,p",[(N,n,p) for p in perc for n in small_n for N in large_n])
def test_time_N_by_n(N,n,p):
    X = gen_matrix(N, n, p)

    start, end = run_test(X)
    old_start, old_end = run_test(X, old=True)

    output_res(old_end,old_start,end,start)

    relative_assert(end-start, old_end-old_start)

# @pytest.mark.parametrize("N1,N2,p",[(N1,N2,p) for p in perc for n in large_n for N in large_n])
# def test_time_N_by_N(N1, N2):
#     X = gen_matrix(N1, N2, p)

#     start, end = run_test(X)
#     old_start, old_end = run_test(X, old=True)

#     output_res(old_end,old_start,end,start)

#     relative_assert(end-start, old_end-old_start)
