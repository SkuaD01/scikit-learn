import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
import time
import profile
import pytest


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

def output_res(old_end,old_start,end,start):
    print("\nOld time:", round((old_end-old_start),4) ,", New time:", round((end-start),4),",", str(round(((old_end-old_start)/(end-start) - 1)*100, 2))+"% improved")


# @pytest.mark.parametrize("n",[100,200,300,400,500])
# def test_1_by_n(n):
#     X = gen_matrix(1, n)

#     start, end = run_test(X)
#     old_start, old_end = run_test(X, old=True)

#     output_res(old_end,old_start,end,start)

#     assert (end-start) < (old_end-old_start)

# @pytest.mark.parametrize("N",[1000,2000,3000,4000,5000,10000])
# def test_1_by_N(N):
#     X = gen_matrix(1, N)

#     start, end = run_test(X)
#     old_start, old_end = run_test(X, old=True)

#     output_res(old_end,old_start,end,start)

#     assert (end-start) < (old_end-old_start)

# @pytest.mark.parametrize("n",[100,200,300,400,500])
# def test_n_by_1(n):
#     X = gen_matrix(n, 1)

#     start, end = run_test(X)
#     old_start, old_end = run_test(X, old=True)

#     output_res(old_end,old_start,end,start)

#     assert (end-start) < (old_end-old_start)

# @pytest.mark.parametrize("N",[1000,2000,3000,4000,5000,10000])
# def test_N_by_1(N):
#     X = gen_matrix(N, 1)

#     start, end = run_test(X)
#     old_start, old_end = run_test(X, old=True)

#     output_res(old_end,old_start,end,start)

#     assert (end-start) < (old_end-old_start)

@pytest.mark.parametrize("n1,n2",[(n1,n2) for n1 in [100,200,300,400,500] for n2 in [100,200,300,400,500]])
def test_n_by_n(n1,n2):
    X = gen_matrix(n1, n2)

    start, end = run_test(X)
    old_start, old_end = run_test(X, old=True)

    output_res(old_end,old_start,end,start)

    assert (end-start) < (old_end-old_start)

# @pytest.mark.parametrize("n,N",[(n,N) for n in [100,200,300,400,500] for N in [1000,2000,3000,4000,5000]])
# def test_n_by_N(n,N):
#     X = gen_matrix(n, N)

#     start, end = run_test(X)
#     old_start, old_end = run_test(X, old=True)

#     output_res(old_end,old_start,end,start)

#     assert (end-start) < (old_end-old_start)

# @pytest.mark.parametrize("N,n",[(N,n) for n in [100,200,300,400,500] for N in [1000,2000,3000,4000,5000]])
# def test_N_by_n(N,n):
#     X = gen_matrix(N, n)

#     start, end = run_test(X)
#     old_start, old_end = run_test(X, old=True)

#     output_res(old_end,old_start,end,start)

#     assert (end-start) < (old_end-old_start)

# def test_N_by_n():
#     X = gen_matrix(10000, 100)

#     start, end = run_test(X)
#     old_start, old_end = run_test(X, old=True)

#     output_res(old_end,old_start,end,start)

#     assert (end-start) < (old_end-old_start)