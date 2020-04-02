"""
Generic helper functions for any channel.

List of functions:

- randseq_uniform: Creates a uniform random sequence
- randseq_non_uniform: samples a random sequence from 
a given distribution 
- hamming_dist: computes hamming distance
- check_lattice_point: check if a distribution corresponds
to a degenerate one (a lattice point)
- make_categorical: given a sequence, make the degenerate
distribution corresponding to it
- make_vector: given a lattice point, compute the 
corresponding sequence

"""
import random
import numpy as np
import math
import scipy.special as sp
import numba
from numba import jit, prange




def randseq_uniform(n,A):
    """
    Creates a random sequence with symbols in {0,1,...,A-1} of length n
    
    Parameters
    ----------
    - n: integer denoting the length of the sequence
    - A: integer denoting size of the symbol alphabet
    
    Returns
    -------
    - output: 1-D numpy array of length n for the random sequence
    
    """
    output = randseq_non_uniform(n,1/A*np.ones(A))
    return output

@jit(nopython = True, parallel = True)
def randseq_non_uniform(n, p):
    """
    This is a helper function for creating random sequence. This function returns
    a float array which further needs to be typecast as an integer array outside,
    since numba doesn't support typecasting inside the jitted function.
    
    Create a n length sequence from alphabet of size A with prior distribution P,
    where Pr(X_i = a) = P[i,a]
    
    Parameters
    ----------
    - n: integer denoting the length of random sequence
    - P: numpy array of size n*A, where P[i,a] denotes the probability
         of the i^th symbol being a
         
    Returns
    -------
    - output: 1-D numpy float array of length n with symbols in {0,1,...,A}
    
    """
    
    if p.sum() > 1:
        raise ValueError('Invalid distribution, probabilities sum to more than 1')
    
    intervals = np.zeros(p.shape[0]+1)
    
    intervals[-1] = 1
    
    for i in range(1,p.shape[0]):
        intervals[i] = intervals[i-1] + p[i-1]
    
    output = 0 * np.arange(n)
    for i in prange(n):
        r = random.random()
        
        for j in range(intervals.shape[0]):
            if r>=intervals[j] and r <= intervals[j+1]:
                output[i] = j
                break
                
    return output


@jit(nopython = True)
def hamming_error_rate(a,b):
    """
    Returns the hamming distance between vectors a and b
    
    Parameters
    ----------
    - a,b: numpy arrays of the same length
    
    Returns
    -------
    - dist: hamming distance between a and b
    
    """
    dist = 0
    for i in range(a.shape[0]):
        if a[i]!=b[i]:
            dist += 1
    
    return dist/a.shape[0]


@jit(nopython = True)
def check_lattice_point(P):
    """
    Check if a distribution P is a lattice point
    
    Parameters
    ----------
    - P: N*A numpy array of distribution
    
    Returns
    -------
    - True or False
    """
    
    if np.all((P*P).sum(axis=1) == 1):
        return True
    
    else:
        return False

@jit(nopython = True)
def make_categorical(X,A):
    """
    Given a vector, make the categorical matrix corresponding to it.
    
    For example, X = [1,0,2] corresponds to categorical matrix 
    [0 1 0]
    [1 0 0]
    [0 0 1]
    
    Parameters
    ----------
    - X: numpy array of length N
    - A: size of the symbol alphabet
    
    Returns
    -------
    - cat_X: N*A numpy matrix, the caegorical representation of X.
    
    """
    
    N = X.shape[0]
    
    cat_X = np.zeros((X.shape[0],A))
    
    for i in range(N):
        cat_X[i,X[i]] = 1
    
    return cat_X

@jit(nopython = True)
def make_vector(P):
    """
    Given a lattice point P, make the vector corresponding to it.
    
    For example, X = [1,0,2] corresponds to categorical matrix 
    [0 1 0]
    [1 0 0]
    [0 0 1]
    
    Parameters
    ----------
    - P: N*A numpy array of lattice point distribution
    
    Returns
    -------
    - X: numpy array of length N such that cat(X) = P 
    
    """
    if check_lattice_point(P):
        X = 0 * np.arange(P.shape[0])
        for i in range(P.shape[0]):
            for a in range(P.shape[1]):
                if P[i,a] == 1:
                    X[i] = a
        return X
    
    else:
        raise ValueError('Not a lattice point distribution')


@jit(nopython = True)
def edit_dist(A, B):
    """
    Given two vectors, compute edit distance (number of insertions, deletions, substitutions
    to transform one vector into the other).
    
    Parameters
    ----------
    - A: numpy array of length N1
    - B: numpy array of length N2
    
    Returns
    -------
    - edit distance between A and B divided by length of A
    
    """
    N1 = A.shape[0]
    N2 = B.shape[0]
    array = np.zeros((N1+1,N2+1))
    for i in range(0,N1):
        array[i+1,0]=i+1
    for j in range(0,N2):
        array[0,j+1]=j+1
    for i in range(0,N1):
        for j in range(0,N2):
            sub_cost = 1
            if A[i] == B[j]:
                sub_cost = 0
            array[i+1,j+1]=min([array[i+1,j]+1,array[i,j+1]+1,array[i,j]+sub_cost])        
    return array[N1,N2]/N1


####### NOT REALLY USING THE BELOW FUNCTIONS #######


"""


def edit_dist(a, b):
    string1 = a
    string2 = b
    n1 = len(string1)
    n2 = len(string2)
    array = np.zeros((n1+1,n2+1))
    for i in range(0,n1):
        array[i+1,0]=i+1
    for j in range(0,n2):
        array[0,j+1]=j+1
    for i in range(0,n1):
        for j in range(0,n2):
            if string1[i] == string2[j]:
                array[i+1,j+1]=array[i,j]
            else:
                array[i+1,j+1]=1+min([min([array[i+1,j],array[i,j+1]]),array[i,j]])        
    return array[n1,n2]

def shift_array(array):
    out = np.roll(array,1)
    out[0]=0
    return out

def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)


"""


