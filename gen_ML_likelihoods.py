#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 21:43:25 2019

@author: sundar
"""

import random           
import numpy as np
import math
import os
import scipy.special as sp
# import matplotlib.pyplot as plt
# import matplotlib

from itertools import repeat
import multiprocessing as mp
from tqdm import tqdm

from helper_fns_delchannel import *
from ML_heuristics import *

# import seaborn as sns
# sns.set()
# sns.set_style('whitegrid')

def one_iter(n,delta,conv_iter = 10, method = 'cood_switch_greedy', init=None, step_size = 0.1,\
             grad_iter = 100, tolerance = 1e-3):
    if init == 'randint':
        p = np.random.uniform(0,1,n)
        p[p>=0.5] = 1.0
        p[p<0.5] = 0.0
        init = p
        
    superseq = randseq(n)
    X = np.array(list(superseq),dtype = 'float')
    trace = dc(superseq,delta)
    Y = np.array(list(trace),dtype = 'int')
    if method == 'cood_switch_greedy':
        rec,_,_ = cood_switch_greedy(n, Y, conv_iter = conv_iter, p = init)
    elif method == 'MAP':
        rec = prob_to_seq(MAP_del(n, Y, init = init))
    elif method == 'ML':
        rec = prob_to_seq(ML_grad_asc(n, Y, step_size = step_size, tolerance = tolerance,\
                                      grad_iter = grad_iter, init = init))
    elif method == 'MAP_with_cood_switch':
        rec,_,_ = cood_switch_greedy(n, Y, conv_iter = conv_iter, p = MAP_del(n, Y, init = init)[:,1])
    elif method == 'ML_with_cood_switch':
        rec,_,_ = cood_switch_greedy(n, Y, conv_iter = conv_iter, \
                                     p = ML_grad_asc(n, Y, step_size = step_size, tolerance = tolerance, \
                                                     grad_iter = grad_iter, init = init)[:,1])
    elif method == 'exact_ML':
        return -(exact_ML(n,Y)['opt_obj'])/F_array(X,Y)[-1,-1]
    else:
        raise ValueError('Method not implemented')
    
#     reconstructed = ''.join(['1' if rec[i]==1 else '0' for i in range(len(rec))])
#     return hamming_dist(superseq,reconstructed)
    return (F_array(rec,Y)[-1,-1]/F_array(X,Y)[-1,-1])

def gen_likelihoods(n,delta_vec,conv_iter = 10, method = 'cood_switch_greedy', \
                    num_hyperiter = 10, process_per_hyperiter = 100,\
                    init=None, step_size = 0.1, grad_iter = 100, tolerance = 1e-3):
    
    likelihoods = {}

    for delta in delta_vec:
        temp_list = []
        print('Computing error for blocklength {} and probability {}'.format(n,delta))
        pool = mp.Pool(mp.cpu_count())
        for it in tqdm(range(num_hyperiter)):
            temp = pool.starmap(one_iter, zip(repeat(n),delta*np.ones(process_per_hyperiter),\
                                              repeat(conv_iter),repeat(method),repeat(init),\
                                             repeat(step_size), repeat(grad_iter), repeat(tolerance)))
            temp_list.extend(temp)
        pool.close()
        likelihoods['delta_{}'.format(delta)] = np.array(temp_list)
        
    return likelihoods

n = 100
delta_vec = np.arange(0.1,1,0.1)
conv_iter = 20
init = None
step_size = 0.1
tolerance = 1e-3
grad_iter = 200

num_hyperiter = 20
process_per_hyperiter = 120

likelihoods = {}


# print('*'*50,'\n Computing for Exact ML \n','*'*50)

# method = 'exact_ML'
# likelihoods['exact_ML'] = gen_likelihoods(n,delta_vec,conv_iter,method,num_hyperiter,\
#                                              process_per_hyperiter,init,step_size,grad_iter,tolerance)

print('*'*50,'\n Computing for MAP \n','*'*50)

method = 'MAP'
likelihoods['MAP'] = gen_likelihoods(n,delta_vec,conv_iter,method,num_hyperiter,\
                                             process_per_hyperiter,init,step_size,grad_iter,tolerance)
print('*'*50,'\n Computing for ML \n','*'*50)

method = 'ML'
likelihoods['ML'] = gen_likelihoods(n,delta_vec,conv_iter,method,num_hyperiter,\
                                             process_per_hyperiter,init,step_size,grad_iter,tolerance)
print('*'*50,'\n Computing for Cood_switch \n','*'*50)

method = 'cood_switch_greedy'
likelihoods['cood_switch'] = gen_likelihoods(n,delta_vec,conv_iter,method,num_hyperiter,\
                                             process_per_hyperiter,init,step_size,grad_iter,tolerance)

print('*'*50,'\n Computing for Cood_switch_vertex \n','*'*50)

method = 'cood_switch_greedy'
init = 'randint'
likelihoods['cood_switch_vertex'] = gen_likelihoods(n,delta_vec,conv_iter,method,num_hyperiter,\
                                             process_per_hyperiter,init,step_size,grad_iter,tolerance)


np.save('likelihoods.npy',likelihoods)
