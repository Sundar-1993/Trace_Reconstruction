"""
Functions specific to deletion channel.

List of functions:

- dc: simulate fractional deletion channel
- lambda_forward: computes log(lambda(P_{1:i},Y_{1:j})) forall i,j
- lambda_reverse: computes log(lambda(P_{i+1:N},Y_{j+1:M})) forall i,j
- lambda_grad_ia: computes the partial derivative of lambda(P,Y) w.r.t P[i,a]
- lambda_grad: computes the partial derivative matrix of lambda(P,Y) wrt P

"""
import random
import numpy as np
import math

from numba import jit,prange

from helper_functions import *
             

@jit(nopython = True)
def dc(seq, delta):
    """
    Takes in a sequence and returns a deleted version,
    where each symbol is deleted with probability p
    
    Parameters
    ----------
    - seq: 1-D numpy array corresponding to the sequence
    - p: deletion probability
    
    Returns
    -------
    - out: 1D numpy array which is a deleted version of seq
    
    """
    
    n = seq.shape[0]
    
    out = 0*np.arange(seq.shape[0]) - 1
    
    idx = 0
    for i in range(seq.shape[0]):
        r = random.random()
        if r > delta:
            out[idx] = seq[i]
            idx += 1
            
    rel_idx = np.where(out != -1)
    
    return out[rel_idx]
    
@jit(nopython = True)
def lambda_forward(P,Y,delta):
    """
    Computes log(lambda(P_{1:i},Y_{1:j})) forall i in {0,1,...,N} ,j in {0,1,...,M}

    Parameters
    ----------
    - P: N*A numpy matrix of probabilities, each row sums to 1
    - Y: numpy array of length M for the trace
    - delta: float deletion probability

    Returns
    -------
    - log_F: numpy matrix of size (N+1)*(M+1) where i,j^th entry is log(lambda(P_{1:i},Y_{1:j}))

    """
    N = P.shape[0]
    M = Y.shape[0]
    log_F = - 1e50 * np.ones((N+1,M+1))
    for i in range(0,N+1):
        for j in range(0,M+1):
            if (j==0):
                log_F[i,j] = i*math.log(delta)
            elif (i>=j):
                temp1 = math.log(delta) + log_F[i-1,j]
                temp2 = math.log(1-delta) + math.log(1e-100 + P[i-1,Y[j-1]]) + log_F[i-1,j-1]
                temp_max = max(temp1,temp2)
                log_F[i,j] = temp_max + math.log(math.exp(temp1-temp_max) + math.exp(temp2-temp_max))
            else:
                log_F[i,j] = -1e50
    return log_F   


@jit(nopython = True)
def lambda_reverse(P,Y,delta):
    """
    Computes log(lambda(P_{i+1:N},Y_{j+1:M})) forall i in {0,1,...,N} ,j in {0,1,...,M}

    Parameters
    ----------
    - P: N*A numpy matrix of probabilities, each row sums to 1
    - Y: numpy array of length M for the trace
    - delta: float deletion probability

    Returns
    -------
    - log_F: numpy matrix of size (N+1)*(M+1) where i,j^th entry is log(lambda(P_{i+1:N},Y_{j+1:M}))

    """
    log_F = lambda_forward(P[::-1],Y[::-1],delta)
    log_F = log_F[::-1,::-1]
    
    return log_F   

@jit(nopython=True)
def lambda_grad_ia(lambda_for,lambda_rev,Y,delta,i,a):
    """
    Compute the partial derivative of lambda(P,Y) w.r.t P[i,a]
    
    Parameters
    ----------
    - lambda_for: numpy matrix of size (N+1)*(M+1) where i,j^th entry is log(lambda(P_{1:i},Y_{1:j}))
    - lambda_rev: numpy matrix of size (N+1)*(M+1) where i,j^th entry is log(lambda(P_{i+1:N},Y_{j+1:M}))
    - Y: the observed trace
    - delta: float corresponding to deletion probability
    - i: integer in {1,2,...,N}
    - a: integer in {0,1,...,A-1}
    
    Returns
    -------
    - out: log of partial derivative of lambda(P,Y) w.r.t P[i,a]
    """
    
    N,M = lambda_for.shape
    N -= 1
    M -= 1
    
    temp = np.zeros(M+1)
    for m in range(M+1):
        temp1 = math.log(delta) + lambda_for[i-1,m] + lambda_rev[i,m]
        if m>0 and (Y[m-1] == a):
            temp2 = math.log(1-delta) + lambda_for[i-1,m-1] + lambda_rev[i,m]
    
        else:
            temp2 = -1e100
        
        temp[m] = max(temp1,temp2) + math.log(math.exp(temp1-max(temp1,temp2))+math.exp(temp2-max(temp1,temp2)))
    
    temp_max = temp.max()

    out = temp_max + math.log((np.exp(temp-temp_max)).sum())
    
    return out


@jit(nopython = True,parallel = True)
def lambda_grad(P,Y,delta):
    """
    Compute the derivative of lambda(P,Y) w.r.t P
    
    Parameters
    ----------
    - P: numpy matrix of size N*A denoting the categorical probabilities
    - Y: the observed trace
    - delta: float corresponding to deletion probability

    
    Returns
    -------
    - out: N*A numpy matrix where entry [i,a] is log of derivative of lambda(P,Y) w.r.t P[i+1,a]
    """
    
    N,A = P.shape
    M = Y.shape[0]
    
    lambda_for = lambda_forward(P,Y,delta)
    lambda_rev = lambda_reverse(P,Y,delta)
    
    out = np.zeros((N,A))
    
    for i in prange(N):
        for a in range(A):
            out[i,a] = lambda_grad_ia(lambda_for,lambda_rev,Y,delta,i+1,a)
    
    return out
    

def log_likelihood_gain(Xhat,X,Y,delta,A):
    """
    Function to compute likelihood gain of estimated seq Xhat over X,
    defined as log Pr(Y|Xhat) - log Pr(Y|X).
    
    Parameters
    ----------
    - Xhat: numpy array, estimated sequence of length N
    - X: numpy array of length N, actual input sequence
    - Y: the observation
    - delta: the deletion probability
    - A: the size of the alphabet
    
    Returns
    -------
    - log_gain = log Pr(Y|Xhat) - log Pr(Y|X)
    
    """
    
    temp1 = lambda_forward(make_categorical(Xhat,A),Y,delta)[-1,-1]
    temp2 = lambda_forward(make_categorical(X,A),Y,delta)[-1,-1]
    
    log_gain = temp1 - temp2
    
    return log_gain