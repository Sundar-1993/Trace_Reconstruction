"""
Trace reconstruction algorithms with multiple traces

- iterative symbolwise MAP with multiple traces
- projected gradient ascent
- exact symbolwise map

Helper function for these also included at the end.

"""
import random
import numpy as np
import math

from numba import jit,prange

from helper_functions import *

from itertools import product



@jit(nopython = True)
def symbolwise_map_seq(P_prior,Y_list,lambda_grad,delta):
    """
    Function to compute iterative symbolwise MAP
    
    Parameters
    ----------
    - P_prior: N*A numpy array of prior probability distribution
    - Y_list: list of numpy arrays for the traces
    - lambda_grad: function to compute the log of gradient of lambda
    - delta: the deletion probability
    
    Returns
    -------
    - X: symbolwise MAP estimate of the input
    
    """
    N = P_prior.shape[0]
    A = P_prior.shape[1]
    
    P = np.ones_like(P_prior) * P_prior
    
    for Y in Y_list:
    
        G = lambda_grad(P,Y,delta)

        temp = np.zeros_like(G[0,:])     

        for a in range(A):
            if P[0,a] == 0:
                temp[a] = -1e100
            else:
                temp[a] = G[0,a] + math.log(P[0,a])

        lambda_val = max(temp) + math.log(np.exp(temp-max(temp)).sum())
                                                # compute the log lambda from gradient


        log_P_post = np.log(P+1e-100) + G - lambda_val  # formula for symbolwise MAP
        P = np.exp(log_P_post) 
    
    X = decode_from_P(P)
    
    return X



@jit(nopython = True)
def proj_grad_asc_traces(P_prior,Y_list,lambda_grad,lambda_forward,delta,\
                         step_size = 0.1,tolerance = 1e-6,max_grad_steps = 100):
    """
    Function for projected gradient ascent.
    
    Parameters
    ----------
    - P_prior: N*A numpy array of starting probability distribution
    - Y_list: list of numpy arrays, the traces
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

        G_list = [lambda_grad(P,Y,delta) for Y in Y_list]
        G_vals = [lambda_forward(P,Y,delta)[-1,-1] for Y in Y_list]
        
        G = np.zeros_like(G_list[0])
        
        for i in range(len(Y_list)):
            G += np.exp(G_list[i]-G_vals[i])
        
        G /= len(G_list)
        
        lambda_list[it] = np.array(G_vals).sum()/len(G_vals)
        
        P = P + step_size * G # gradient ascent step
        
        P = simplex_proj_mat(P)                    # projection step
        
        if it > 5:
            if (np.std(lambda_list[it-5:it])/np.abs(lambda_list[it-5:it].min()) < tolerance):
                break
                
    X = decode_from_P(P) 
    return X

def symbolwise_map_exact(P,traces,delta):
    """
    Function for exact symbolwise MAP with multiple traces.
    
    Parameters
    ----------
    - P: N*A numpy array of prior probabilities
    - traces: list of numpy arrays for the traces
    - delta: deletion probability

    Returns
    -------
    - Xhat: estimate of the input obtained via exact symbolwise MAP
    
    """
    N,A = P.shape

    forward_vals = forward_pass(N,A,traces,delta,P)
    backward_vals = backward_pass(N,A,traces,delta,P)
    P_post = compute_posteriors(N,A,traces,delta,P,forward_vals,backward_vals)
    
    X_hat = decode_from_P(P_post)
    return X_hat


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

################### HELPER FUNCTIONS SPECIFICALLY FOR EXACT SYMBOLWISE MAP ###############

@jit(nopython = True)
def log_sum_exp(ns):
    """
    Function to do log sum exp calculation, i.e.
    given a and b, returns log(exp(a)+exp(b))
    
    Parameters
    ----------
    - ns: list of numbers
    
    Returns
    -------
    - log(sum_i(exp(ns[i])))
    
    """
    ns = np.array(ns)
    max_num = max(ns)
    ds = ns - max_num
    sumOfExp = np.exp(ds).sum()
    return max_num + np.log(sumOfExp)


def gen_joint_drifts(N,M_list):
    """
    Given the length of each traces, this generates
    the joint drift values (set of states).
    
    Parameters
    ----------
    - N: length of the input
    
    Returns
    -------
    - out: S*T numpy array, each row is (d_1,d_2,...,d_T)
           a possible drift value. The possible drift values
           range from (0,0,..,0) to (N-M_1,N-M_2,...N-M_T)
    """
    drift_list = [np.arange(0,N-M+1) for M in M_list]
    
    out = cartesian_product(drift_list)
    return out


# Generalized N-dimensional products
def cartesian_product(arrays):
    """
    Generalized N-dimensional products.
    Given a list of arrays (or array of arrays),
    generates their cartesian product.
    
    E.g: A = [0,1], B = [2,3], C = [4,5]
    
    cartesian_product([A,B,C]) = D
    where D = [
    [0,2,4]
    [0,2,5]
    [0,3,4]
    .
    .
    [1,3,5]
    ]
    
    Parameters
    ----------
    - arrays: list of 1-D arrays or 2-D numpy array
    
    Returns
    -------
    - arr: 2-D numpy array where each row is an element of the
           cartesian product
    """
    la = len(arrays)
    dtype = np.find_common_type([a.dtype for a in arrays], [])
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)


def make_edges(states,unit_cube):
    """
    Generate the set of incoming edges for each joint drift state.
    The set of edges are generated as indices.
    
    Eg: (3,4,2) is connected to (3,4,2),(2,4,2),....,(2,3,1).
    If the index for (3,4,2) is 24, say, then the row corresponding
    to (3,4,2) lists out the indices of (3,4,2),.....(2,3,1).
    
    Parameters
    ----------
    - states: S*T numpy array corresponding to name of the states 
    - unit_cube: 2^T * T numpy array each row corresponding to
                 T-bit the binary representation of elements from
                 0 to 2^T-1.
                 
    Returns
    -------
    - edges: S * 2^T numpy array. Entry edges[i,t] corresponds to the
             index of the state (states[i] - unit_cube[t])
    
    """
    S,T = states.shape
    
    max_drifts = states.max(axis = 0)
    
    number_system = np.arange(T)
    
    number_system[T-1] = 1
    for i in range(1,T):
        number_system[T-1-i] = number_system[T-i] * (max_drifts[T-i]+1)

    edges = -1*np.ones((S,2**T),dtype = int)
    
    for i,s in enumerate(states):
        for j,c in enumerate(unit_cube):
            if (s - c).min() >= 0:
                edges[i,j] = i - (number_system*c).sum()
                
    return edges    


@jit(nopython = True)
def one_forward_pass(t,states,init_vals,traces,delta,unit_cube,edges,P):
    """
    Perform forward pass from t to t+1.
    
    Parameters
    ----------
    - states: 2-D numpy array of size S*T denoting joint drift states
    - init_val: 1-D numpy array of length S denoting log
                probabilities (forward values) at time 't'
    - traces:  list of traces
    - delta: deletion probability
    - unit_cube: 2-D numpy array of shape 2**T,T corresponding to binary representation of
                 0, 1,....2**T-1
    - edges:  2-D numpy array of size S * 2^T entry edge[i,j] corresponds to
              the index of the state (state[i]-unit_cube[j])
                 
                 
    Returns
    -------
    - next_vals: 1-D numpy array of length S denoting log
                 probabilities (forward values) at time 't+1'
    
    """
    
    next_vals = -1e100 * np.ones(init_vals.shape)
    
    S,T = states.shape
    
    for i in range(len(states)):
        for j in range(len(unit_cube)):
#             print('State, unitcube subtracted',s,c)
            if (states[i] - unit_cube[j]).min() >= 0:
                
                state_idx = edges[i,j]
                
                tau = np.where(unit_cube[j] == 0)
                
                if len(tau[0]) == 0:
                    next_vals[i] = log_sum_exp([next_vals[i],\
                                                init_vals[state_idx]+T*np.log(delta)])
                    
                else:
                    
                    flag = 0
                    temp = []
                    for k in tau[0]:
                        if (t-states[i,k] >=0 and t-states[i,k]< len(traces[k])):
                            temp.append(traces[k][t-states[i][k]])
                        else:
                            flag = 1
                            break

                    if (len(set(temp)) == 1 and flag == 0):
                        bit = temp[0]
                        next_vals[i] = log_sum_exp([next_vals[i],init_vals[state_idx]\
                                                    +(T-len(temp)) * np.log(delta)\
                                                    + np.log(1-delta) * (len(temp))
                                                    + np.log(P[t,bit]+1e-100)])
                
    return next_vals
        

def forward_pass(N,A,traces,delta,P):
    
    T = len(traces)
    
    unit_cube = cartesian_product(np.vstack([[0,1]]*T))
    states = gen_joint_drifts(N,[len(Y) for Y in traces])
    init_vals = -1e100*np.ones(states.shape[0])
    init_vals[0] = 0
    edges = make_edges(states,unit_cube)

    S,T = states.shape
    
    forward_vals = np.zeros((S,N+1))
    
    forward_vals[:,0] = 1.0*init_vals
    
    for t in range(0,N):
        forward_vals[:,t+1] =  one_forward_pass(t,states,forward_vals[:,t]\
                                              ,traces,delta,unit_cube,edges,P)
    return forward_vals


@jit(nopython = True)
def one_backward_pass(t,states,init_vals,traces,delta,unit_cube,edges,P):
    """
    Perform backward pass from t+1 to t.
    
    Parameters
    ----------
    - states: 2-D numpy array of size S*T denoting joint drift states
    - init_val: 1-D numpy array of length S denoting log
                probabilities (forward values) at time 't+1'
    - traces:  list of traces
    - delta: deletion probability
    - unit_cube: 2-D numpy array of shape 2**T,T corresponding to binary representation of
                 0, 1,....2**T-1
    - edges:  2-D numpy array of size S * 2^T entry edge[i,j] corresponds to
              the index of the state (state[i]-unit_cube[j])
                 
                 
    Returns
    -------
    - next_vals: 1-D numpy array of length S denoting log
                 probabilities (backward values) at time 't'
    
    """
    
    next_vals = -1e100 * np.ones_like(init_vals)
    
    S,T = states.shape
    
    for i in range(len(states)):
        for j in range(len(unit_cube)):
#             print('State, unitcube subtracted',s,c)
            if (states[i] - unit_cube[j]).min() >= 0:
                
                state_idx = edges[i,j]
                
                tau = np.where(unit_cube[j] == 0)
                
                if len(tau[0]) == 0:
                    next_vals[state_idx] = log_sum_exp([next_vals[state_idx],\
                                                init_vals[i]+T*np.log(delta)])
                    
                else:
                    flag = 0
                    temp = []
                    for k in tau[0]:
                        if (t-states[i,k] >=0 and t-states[i,k]< len(traces[k])):
                            temp.append(traces[k][t-states[i][k]])
                        else:
                            flag = 1
                            break

                    if (len(set(temp)) == 1 and flag == 0):
                        bit = temp[0]
                        next_vals[state_idx] = log_sum_exp([next_vals[state_idx],init_vals[i]\
                                                    + (T-len(temp)) * np.log(delta)\
                                                    + np.log(1-delta) * (len(temp))\
                                                    + np.log(P[t,bit] + 1e-100)])
                
    return next_vals
        

def backward_pass(N,A,traces,delta,P):
    
    T = len(traces)
    
    unit_cube = cartesian_product(np.vstack([[0,1]]*T))
    states = gen_joint_drifts(N,[len(Y) for Y in traces])
    init_vals = -1e100*np.ones(states.shape[0])
    init_vals[-1] = 0
    edges = make_edges(states,unit_cube)

    S,T = states.shape
    
    backward_vals = np.zeros((S,N+1))
    
    backward_vals[:,-1] = 1.0*init_vals
    
    for t in range(N-1,-1,-1):
        backward_vals[:,t] =  one_backward_pass(t,states,backward_vals[:,t+1]\
                                              ,traces,delta,unit_cube,edges,P)
    return backward_vals

@jit(nopython = True)
def one_compute_posterior(t,a,states,forward_vals,backward_vals,traces,delta,unit_cube,edges,P):
    """
    Compute posterior probabilty Pr(observation|x_t = a)
    
    Parameters
    ----------
    - states: 2-D numpy array of size S*T denoting joint drift states
    - init_val: 1-D numpy array of length S denoting log
                probabilities (forward values) at time 't'
    - traces:  list of traces
    - delta: deletion probability
    - unit_cube: 2-D numpy array of shape 2**T,T corresponding to binary representation of
                 0, 1,....2**T-1
    - edges:  2-D numpy array of size S * 2^T entry edge[i,j] corresponds to
              the index of the state (state[i]-unit_cube[j])
                 
                 
    Returns
    -------
    - next_vals: 1-D numpy array of length S denoting log
                 probabilities (forward values) at time 't+1'
    
    """
    
    init_vals = forward_vals.copy()
    next_vals = -1e100 * np.ones(init_vals.shape)
    
    S,T = states.shape
    
    for i in range(len(states)):
        for j in range(len(unit_cube)):
#             print('State, unitcube subtracted',s,c)
            if (states[i] - unit_cube[j]).min() >= 0:
                
                state_idx = edges[i,j]
                
                tau = np.where(unit_cube[j] == 0)
                
                if len(tau[0]) == 0:
                    next_vals[i] = log_sum_exp([next_vals[i],\
                                                init_vals[state_idx]+T*np.log(delta)])
                    
                else:
                    
                    flag = 0
                    temp = []
                    for k in tau[0]:
                        if (t-states[i,k] >=0 and t-states[i,k]< len(traces[k])):
                            temp.append(traces[k][t-states[i][k]])
                        else:
                            flag = 1
                            break

                    if (len(set(temp)) == 1 and flag == 0 and temp[0] == a):
                        bit = a
                        next_vals[i] = log_sum_exp([next_vals[i],init_vals[state_idx]\
                                                    +(T-len(temp)) * np.log(delta)\
                                                    + np.log(1-delta) * (len(temp))])
    
    
    out = next_vals + backward_vals
    
    return log_sum_exp(list(out))
        

def compute_posteriors(N,A,traces,delta,P,forward_vals,backward_vals):
    
    T = len(traces)
    
    unit_cube = cartesian_product(np.vstack([[0,1]]*T))
    states = gen_joint_drifts(N,[len(Y) for Y in traces])
    init_vals = -1e100*np.ones(states.shape[0])
    init_vals[-1] = 0
    edges = make_edges(states,unit_cube)

    S,T = states.shape
    
    posteriors = np.zeros((N,A))
    
    for t in range(0,N):
        for a in range(0,A):
            posteriors[t,a] =  one_compute_posterior(t,a,states,forward_vals[:,t],\
                                                        backward_vals[:,t+1],traces,delta,unit_cube,edges,P)
            posteriors[t,a] += np.log(P[t,a]+1e-500)
    
        posteriors[t] -= log_sum_exp(list(posteriors[t]))
    
    posteriors = np.exp(posteriors)
    
    return posteriors


