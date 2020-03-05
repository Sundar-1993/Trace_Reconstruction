import random
import numpy as np
import math
import time

from numba import jit,prange

from helper_functions import *
from deletion_functions import *
from inference_metaheuristics import *
from trace_reconstruction_heuristics import *

from tqdm import trange
import multiprocessing as mp
from itertools import repeat

import matplotlib
import matplotlib.pyplot as plt

import seaborn as sns
sns.set()
sns.set_style('whitegrid')

def one_iter(N,A,T_s,delta, method = None, method_params = None):
    
    X = randseq_uniform(N,A)
    Y_list = []
    
    hamming_error_rates = []
    
    for t in range(1,max(T_s)+1):
        Y_list.append(dc(X,delta))
        
        if t in T_s:
            if method == 'proj_grad_asc_traces':
                Xhat = proj_grad_asc_traces(method_params['P_init'],Y_list,lambda_grad,\
                                     lambda_forward,delta,step_size = 0.1,\
                                     tolerance = 1e-6,max_grad_steps = 100)
            elif method == 'symbolwise_map_seq':
                Xhat = symbolwise_map_seq(method_params['P_init'],Y_list,lambda_grad,delta)
                
            elif method == 'symbolwise_map_exact':
                Xhat = symbolwise_map_exact(method_params['P_init'],Y_list,delta)
            else:
                raise ValueError('Method not implemented')
            
            hamming_error_rates.append(hamming_error_rate(Xhat,X))
    
#     print(X,Xhat)
    return np.array(hamming_error_rates)


##### Warming up numba and the functions #####
import warnings
warnings.filterwarnings('ignore')

print('Warming up the numba functions....')

N = 10
A = 2
T_s = [1]
delta = 0.2

method = 'proj_grad_asc_traces'
method_params = {}
method_params['P_init'] = 1/A * np.ones((N,A))
one_iter(N,A,T_s,delta, method, method_params)

method = 'symbolwise_map_seq'
method_params = {}
method_params['P_init'] = 1/A * np.ones((N,A))
one_iter(N,A,T_s,delta, method, method_params)

method = 'symbolwise_map_exact'
method_params = {}
method_params['P_init'] = 1/A * np.ones((N,A))
one_iter(N,A,T_s,delta, method, method_params)

print('Warming up completed.')

def gen_error_rates(N,A,T_s,delta_vec, method = None, method_params = None, hyperiters = 100,process_per_hyperiter = 100):
    
    results = {}
    results['summary'] = ("Hamming error rates and likelihood gains for a blocklength of {}, "
    "an alphabet size {} using the method {}".format(N,A,method))
    
    results['delta_vec'] = delta_vec
    
    hamming_error_list = np.zeros((len(delta_vec),len(T_s)))
    #likelihood_gain_list = np.zeros((len(delta_vec),hyperiters*process_per_hyperiter))
    
    for idx in trange(len(delta_vec),desc = 'Delta values'):
#         print('Computing for delta = ',delta)
        delta = delta_vec[idx]
        time.sleep(0.4)
        pool = mp.Pool(mp.cpu_count())
        for it in trange(hyperiters,desc = 'Hyperiters'):
            temp = pool.starmap(one_iter, zip(repeat(N),repeat(A),repeat(T_s),delta*np.ones(process_per_hyperiter),\
                                              repeat(method),repeat(method_params)))
            temp = np.array(temp)
            hamming_error_list[idx,:] += temp.sum(axis = 0)
            #likelihood_gain_list[idx,it*process_per_hyperiter:(it+1)*process_per_hyperiter] = temp[:,1]
        pool.close()
    
    hamming_error_list /= hyperiters * process_per_hyperiter
    results['hamming_error_list'] = hamming_error_list
#     results['likelihood_gain_list'] = likelihood_gain_list
    
    return results


import warnings
warnings.filterwarnings('ignore')

N = 500
A = 2
T_s = [1,2,4,8,12,20]
delta_vec = np.arange(0.1,0.6,0.1)

hyperiters = 40
process_per_hyperiter = 40

errors = {}

methods = ['symbolwise_map_seq','proj_grad_asc_traces']

for method in methods:
    print('*'*50,'\n',method,'\n','*'*50)
    method_params = {}
    method_params['P_init'] = 1/A * np.ones((N,A))
    errors[method] = gen_error_rates(N,A,T_s,delta_vec, method, method_params,hyperiters,process_per_hyperiter)


np.save('errors_BL{}.npy'.format(N),errors)