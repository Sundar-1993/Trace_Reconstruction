"""
Metaheuristics and algorithms for inference over deletion channels. 
Note that these are metaheuristics, and can be easily generalized
to any channel. When using these for other channels, the 
algorithm for computing lambda, its gradient and the channel parameters
(such as delta) change. The metaheuristic remains the same otherwise.

- exact_ml: for true maximum likelihood sequence
- symbolwise_map: self-explanatory
- coordinate_refinement_greedy: self-explanatory
- proj_grad_asc: projected gradient ascent

Helper function for these also included at the end.

"""
import random
import numpy as np
import math

from numba import jit,prange

from helper_functions import *

from itertools import product


def exact_ml(N,A,Y,lambda_forward,delta):
    """
    Function to compute exact maximum-likelihood sequence for deletion channel
    
    Parameters
    ----------
    - N: length of input sequence
    - A: the alphabet size
    - Y: numpy array of length M, the observation
    - lambda_forward: algorithm to compute log Pr(Y_{1:j}|X_{1:i}) for all i,j
    - delta: deletion probability
    
    Returns
    -------
    - best_seq: most likely input sequence
    
    """

    list_seq = [i for i in product(range(A), repeat = N)]
    
    list_seq = np.array(list_seq)
    
    best_seq = exact_ml_computation(N,A,Y,lambda_forward,list_seq,delta)
    
    return best_seq

@jit(nopython = True)
def exact_ml_computation(N,A,Y,lambda_forward,list_seq,delta):
    """
    jitted support function for ML calculation
    
    Parameters
    ----------
    - N: length of input sequence
    - A: the alphabet size
    - Y: numpy array of length M, the observation
    - lambda_forward: algorithm to compute log Pr(Y_{1:j}|X_{1:i}) for all i,j
    - list_seq: list of all possible N length sequences
    - delta: deletion probability
    
    Returns
    -------
    - best_seq: most likely input sequence
    
    """
    
    max_val = -1e100
    
    best_seq = list_seq[0]
    
    for i in range(list_seq.shape[0]):
        
        temp = make_categorical(list_seq[i],A)
        
        lambda_val = lambda_forward(temp,Y,delta)[-1,-1]
        
        if lambda_val > max_val:
            max_val = lambda_val
            best_seq = list_seq[i]
            
    return best_seq
    

@jit(nopython = True)
def symbolwise_map(P_prior,Y,lambda_grad,delta):
    """
    Function to compute symbolwise MAP
    
    Parameters
    ----------
    - P_prior: N*A numpy array of prior probability distribution
    - Y: numpy array of length M, the observation
    - lambda_grad: function to compute the log of gradient of lambda
    - delta: the deletion probability
    
    Returns
    -------
    - X: symbolwise MAP estimate of the input
    
    """
    N = P_prior.shape[0]
    A = P_prior.shape[1]
    
    G = lambda_grad(P_prior,Y,delta)
    
    temp = np.zeros_like(G[0,:])     
            
    for a in range(A):
        if P_prior[0,a] == 0:
            temp[a] = -1e100
        else:
            temp[a] = G[0,a] + math.log(P_prior[0,a])

    lambda_val = max(temp) + math.log(np.exp(temp-max(temp)).sum())
                                            # compute the log lambda from gradient
    
    
    log_P_post = np.log(P_prior+1e-100) + G - lambda_val  # formula for symbolwise MAP
    P_post = np.exp(log_P_post) 
    
    X = decode_from_P(P_post)
    
    return X


@jit(nopython = True)
def cood_refinement_greedy(P_init,Y,lambda_grad, delta, max_iter = 10, print_conv_iter = False):
    """
    The metaheuristic for greedy coordinate refinement.
    
    Parameters
    ----------
    - P_init: N*A numpy array for initial distribution
    - Y: numpy array of length M for observation vector
    - lambda_grad: a jitted function that returns a numpy array
                   of size N*A where the (i,a)th entry is 
                   equal to log d/dP_{(i+1)a} lambda(P,Y)
    - delta: deletion probability
    - max_iter: maximum number of refinement iterations
    
    Returns
    -------
    - a numpy array of length N which is an estimate of input X,
    given observation Y.
    
    Notes
    -----
    This function is jitted, meaning that this function isn't 
    written in a "pythonic" fashion, disregarding broadcasting
    and other python capabilities that typically correspond to
    good programming practices. In fact, you will see the usage
    of many for loops etc., which can definitely be avoided with
    usual python programming. However, writing this function
    this way allows us to jit and compile the function that makes 
    it extremely fast during runtime.
    
    """
    
    P = P_init * np.ones_like(P_init)
    
    N = P_init.shape[0]
    A = P_init.shape[1]
    M = Y.shape[0]
    
    for it in range(max_iter):
        
        vis_indices = np.zeros(N)            # initialize all vertices as unvisited
        
        while (vis_indices.sum() < N):

            G = lambda_grad(P,Y,delta)       # compute gradient
            
            temp = np.zeros_like(G[0,:])     
            
            for a in range(A):
                if P[0,a] == 0:
                    temp[a] = -1e100
                else:
                    temp[a] = G[0,a] + math.log(P[0,a])
                    
            lambda_val = max(temp) + math.log(np.exp(temp-max(temp)).sum())
                                            # compute the log lambda val from gradient
            
            gain = G - lambda_val           # gain measures how much each gradient val
                                            # increases the function by
    
            
            if check_lattice_point(P):
                if gain.max() <= 1e-10:         # if no gradient gives gain, then fixed point
                    
                    if print_conv_iter:
                        print('converged in ',it+1, 'iterations')
                    return make_vector(P)
            
            for i in range(N):              # remove visited indices from picture
                if vis_indices[i] == 1:
                    gain[i,:] -= 1e100
            
            gain_max = -1.0

            for i in range(N):
                for a in range(A):
                    if gain[i,a] >= gain_max:  # choose i,a that gives max gain
                        gain_max = gain[i,a]
                        i_star,a_star = i,a
            
            
            P[i_star,:] = 0
            P[i_star,a_star] = 1
            
            vis_indices[i_star] = 1
            
    if print_conv_iter:       
        print('converged in ',it+1, 'iterations')
    return make_vector(P)


@jit(nopython = True)
def proj_grad_asc(P_prior,Y,lambda_grad,delta,step_size = 0.1,tolerance = 1e-6,max_grad_steps = 100):
    """
    Function for projected gradient ascent.
    
    Parameters
    ----------
    - P_prior: N*A numpy array of starting probability distribution
    - Y: numpy array of length M, the observation
    - lambda_grad: function to compute the log of gradient of lambda
    - delta: the deletion probability
    - step_size: float
    - tolerance: float in [0,1]
    - max_grad_steps: max number of gradient steps
    
    Returns
    -------
    - X: estimate of the input obtained via projected gradient ascent
    
    """
    N = P_prior.shape[0]
    A = P_prior.shape[1]
    
    P = P_prior * np.ones_like(P_prior)
    
    #lambda_list = []
    lambda_list = np.zeros(max_grad_steps)

    for it in range(max_grad_steps):

        G = lambda_grad(P,Y,delta)
        
        temp = np.zeros_like(G[0,:])     
        for a in range(A):
            if P[0,a] == 0:
                temp[a] = -1e100
            else:
                temp[a] = G[0,a] + math.log(P[0,a])

        lambda_val = max(temp) + math.log(np.exp(temp-max(temp)).sum())
                                                   # compute the log lambda val from gradient
        lambda_list[it] = lambda_val
        
        P = P + step_size * np.exp(G - lambda_val) # gradient ascent step
        
        P = simplex_proj_mat(P)                    # projection step
        
        if it > 5:
            if (np.std(lambda_list[it-5:it])/np.abs(lambda_list[it-5:it].min()) < tolerance):
                break
                
    X = decode_from_P(P) 
    return X


############################# HELPER FUNCTIONS ###################################

@jit(nopython = True)
def decode_from_P(P):
    """
    Given a distribution P, find the most likely vector
    """
    N = P.shape[0]
    A = P.shape[1]
    
    X = np.arange(N)
    
    for i in range(N):
        max_val = -1e100
        for a in range(A):
            if P[i,a] > max_val:
                max_val = P[i,a]
                X[i] = a
    
    return X


@jit(nopython = True)
def simplex_proj(p):
    """
    Function to project a real vector onto the unit simplex hyperplane.
    Algorithm finds closes point on the simplex, i.e., solves the following
    problem
    
    argmin_{x} ||x-p||^2 
    
    s.t. sum(x) = 1
         x_i >= 0 forall i
         
    Check https://eng.ucmerced.edu/people/wwang5/papers/SimplexProj.pdf for
    description of algorithm.
    
    Parameters
    ----------
    - p:  numpy array of length N
    
    Returns
    -------
    - p_proj: numpy array of positive entries such that entries sum to one
    
    """
    A = p.shape[0]
    u = np.sort(p)[::-1]
    
    temp1 = np.zeros(A)
    
    for i in range(A):
        temp1[i] = u[:i+1].sum()
    
    temp2 = (1-temp1) / np.arange(1,A+1)
    
    rho = A
    for i in np.arange(A,0,-1):
        if (u[i-1] + temp2[i-1]) > 0:
            rho = i
            break
    
    lam = temp2[rho-1]
    
    p_proj = np.maximum(np.zeros(A),p+lam)
    return p_proj

@jit(nopython = True)
def simplex_proj_mat(P):
    """
    Function to project a matrix P onto the unit simplex
    
    Parameters
    ----------
    - P: N*A numpy array
    
    Returns
    -------
    - P_proj: N*A numpy array of positive entries such that each row sums to one
    """
    P_proj = np.zeros_like(P)
    for i in range(P.shape[0]):
        P_proj[i,:] = simplex_proj(P[i,:])
    
    return P_proj