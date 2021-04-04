import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
import time
import profile
import pytest

##################### THIS TEST SUITE WILL TAKE APPROXIMATELY 15 MINUTES, BUT VARIES BY MACHINE ############################

# Initialise constants
epsilon = 0.75 # Seconds of breathing room for smaller sized n tests
perc = [0.1,0.5,0.9] # We want to test a varying amount of missing values to impute, each of these represent the precrentage of missing values in X
small_n = [5,300,500]
large_n = [1000,3000,5000]

# generate random array of size x by y with (p*100)% missing values
def gen_matrix(x, y, p):
    np.random.seed(1)
    X = np.random.random([x, y])
    X = pd.DataFrame(X).mask(X <= p)
    return X

# Run (simulated) old time or new time, return start and end times
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
    # Since smaller times will yield more sporatic results, an amount of seconds of amount epsilon are accounted for
    assert ((new == pytest.approx(old, abs=epsilon)) or (new < old))

def output_res(old_end,old_start,end,start):
    print("\nOld time:", round((old_end-old_start),4) ,", New time:", round((end-start),4),",", str(round(((old_end-old_start)/(end-start) - 1)*100, 2))+"% improved")

# Test arrays of size 1 by a small n with varying missing values
@pytest.mark.parametrize("n,p",[(n, p) for p in perc for n in small_n])
def test_time_1_by_n(n,p):
    X = gen_matrix(1, n, p)

    start, end = run_test(X)
    old_start, old_end = run_test(X, old=True)

    output_res(old_end,old_start,end,start)

    relative_assert(end-start, old_end-old_start)

# Test arrays of size 1 by a large n with varying missing values
@pytest.mark.parametrize("N,p",[(N, p) for p in perc for N in large_n])
def test_time_1_by_N(N,p):
    X = gen_matrix(1, N, p)

    start, end = run_test(X)
    old_start, old_end = run_test(X, old=True)

    output_res(old_end,old_start,end,start)

    relative_assert(end-start, old_end-old_start)

# Test arrays of a small n by 1 with varying missing values
@pytest.mark.parametrize("n,p",[(n, p) for p in perc for n in small_n])
def test_time_n_by_1(n,p):
    X = gen_matrix(n, 1, p)

    start, end = run_test(X)
    old_start, old_end = run_test(X, old=True)

    output_res(old_end,old_start,end,start)

    relative_assert(end-start, old_end-old_start)

# Test arrays of a large n by 1 with varying missing values
@pytest.mark.parametrize("N,p",[(N, p) for p in perc for N in large_n])
def test_time_N_by_1(N,p):
    X = gen_matrix(N, 1, p)

    start, end = run_test(X)
    old_start, old_end = run_test(X, old=True)

    output_res(old_end,old_start,end,start)

    relative_assert(end-start, old_end-old_start)

# Test arrays of a small n by a small n with varying missing values
@pytest.mark.parametrize("n1,n2,p",[(n1,n2,p) for p in perc for n1 in small_n for n2 in small_n])
def test_time_n_by_n(n1,n2,p):
    X = gen_matrix(n1, n2, p)

    start, end = run_test(X)
    old_start, old_end = run_test(X, old=True)

    output_res(old_end,old_start,end,start)

    relative_assert(end-start, old_end-old_start)

# Test arrays of a small n by a large n with varying missing values
@pytest.mark.parametrize("n,N,p",[(n,N,p) for p in perc for n in small_n for N in large_n])
def test_time_n_by_N(n,N,p):
    X = gen_matrix(n, N, p)

    start, end = run_test(X)
    old_start, old_end = run_test(X, old=True)

    output_res(old_end,old_start,end,start)

    relative_assert(end-start, old_end-old_start)

# Test arrays of a large n by a small n with varying missing values
### This is the most important test case since it is the most likely scenario for usage
### (More likely to be testing a large number of features)
@pytest.mark.parametrize("N,n,p",[(N,n,p) for p in perc for n in small_n for N in large_n])
def test_time_N_by_n(N,n,p):
    X = gen_matrix(N, n, p)

    start, end = run_test(X)
    old_start, old_end = run_test(X, old=True)

    output_res(old_end,old_start,end,start)

    relative_assert(end-start, old_end-old_start)

# Test arrays of a large n by a large n with varying missing values
# (This takes a VERY long time to run, since they are more often called individually)
## @pytest.mark.parametrize("N1,N2,p",[(N1,N2,p) for p in perc for N1 in large_n for N2 in large_n])
## def test_time_N_by_N(N1, N2, p):
##     X = gen_matrix(N1, N2, p)

##     start, end = run_test(X)
##     old_start, old_end = run_test(X, old=True)

##     output_res(old_end,old_start,end,start)

##     relative_assert(end-start, old_end-old_start)
