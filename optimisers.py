# -*- coding: utf-8 -*-
"""
Implementation of various numerical optimisers.

@author: Sebastian J. Scott
"""
#%% Load modules
import numpy as np
import torch

from tqdm import tqdm
import time


from matplotlib import pyplot as plt #  Not necessary in general

#%% Algorithms for finding minimiser

def truncate_info(info,n):
    info['t_grad_computation'] = info['t_grad_computation'][:(n+1)]
    info['grad_norms'] = info['grad_norms'][:n+1]
    info['step_lengths'] = info['step_lengths'][:(n+1)]
    # info['step_lengths'][n-1] = None
    info['backtrack_its'] = info['backtrack_its'][:n+1]
    info['t_step_length'] = info['t_step_length'][:n+1]
    info['t_iteration_total'] = info['t_iteration_total'][:n+1]
    info['fun_evals'] = info['fun_evals'][:n+1]
    return info

#Gradient descent
def gradient_descent( fun, x0, max_its=10000, grad_tol=1e-12, verbose=False,
                     desc='', store_iters=False, full_info = True, constraint_fun = None):
    """
    Gradient descent algorithm. Solves
        min_x  fun(x)
    where fun is differentiable function with gradient property grad. Algorithm is terminated 
    if either max_its number of iterations is reached, gradient norm is smaller than grad_tol,
    or proposed step length by backtracking linessearch is smalelr than 1e-14
    
    INPUTS
    fun : function with single input, returns real number. Has property grad, 
        Single input and returns output the same dimension as the input
    x0 : starting point of the algoithm.
    max_its : Maximum number of iterations to run for
    grad_tol : Gradient tolerance for early stopping
    verbose : Flag for whether information about the updates should be stated
    desc : Extra information to be prefixed onto printed statements
    store_iters: Flag for whether the iterates should be stored
    
    OUTPUTS
    x - most recent iteration of the algorithm
    flag - statement regarding the convergence of gradient descent
    """
    x = x0.detach().clone()      
    if constraint_fun is not None: # Apply any projections/constraints
                x = constraint_fun(x)
                
    step_next = 1e-2 # Initial step length for backtracking algorithm for first iteration of gradient descent
    if full_info:
        info = { # Dictionary for extra information about GD iterations
            't_grad_computation':200*np.ones(max_its), # array of length of times for each gradient computation
            'grad_norms':200*np.ones(max_its), # array of the gradient norms at each iteration
            'step_lengths':200*np.ones(max_its),# array of step lengths used for each iteration of algorithm
            'backtrack_its':200*np.ones(max_its), # array of number of backtracking iterations used for each step length
            't_step_length':200*np.ones(max_its), # array of length of time for each step length computation
            't_iteration_total':100*np.ones(max_its), # array of length of overall time for each iteration
            'fun_evals':50*np.ones(max_its), # array of length of evaluation of cost function
            'stop_it':None, # array of length of overall time for each iteration
            'all_iters':None}
    else:  
        info = { # Dictionary of minimal amount of information
                'grad_norms' : None, # Final gradiennt norm of solver
                't_total' : None, # Total computation time
                'stop_it' : None # Stopping iteration of algorithm
            }
    if store_iters: info['all_iters'] = [x0] # History of considered points
    
    pbar = tqdm(total=max_its, disable=not(verbose), position=0, ncols=0)
    it = 0
    
    t_start_loop = time.time()
    fun_new = None
    
    for it in range(max_its):
        t_iteration_start = time.time()
        
        # Compute gradient
        grad_tmp = fun.grad(x)
        if grad_tmp is tuple: grad_x , grad_info = grad_tmp
        else: grad_x = grad_tmp
        t_grad = time.time() - t_iteration_start
        
        # Save gradient-based information
        grad_norm = grad_x.norm()
        if full_info:
            info['t_grad_computation'][it] = t_grad
            info['grad_norms'][it] =  grad_norm.detach()
        
        # Early stopping based on gradient norm
        if grad_norm <= grad_tol:
            info['stop_it'] = it+1 # Store stopping iterations
            info['convg_flag'] = 0
            info['t_total'] = time.time() - t_start_loop
            if full_info: 
                info['t_iteration_total'][it] =  time.time() - t_iteration_start
                info['fun_evals'][it] = fun_new # Function evaluation at current iteration
                info['step_lengths'] = info['step_lengths'][:it]
                info['backtrack_its'] = info['backtrack_its'][:it]
                info['t_step_length'] = info['t_step_length'][:it] 
                info = truncate_info(info,it)
            else: 
                info['grad_norms'] = grad_norm.detach()
            pbar.close()
            return x, info
        
        # Determine step length for GD by backtracking linesearch
        t_tmp = time.time()
        step_length, step_next, backtrack_satisfied, backtrack_it, x_new, fun_new, fun_old = backtracking(fun, grad_x, x,search_dir=-grad_x, step_size=step_next, fun_eval = fun_new)
        # step_length, backtrack_it, fun_old, backtrack_satisfied = 5e-2, 0,0,1
        # x_new = x + step_length * grad_x
        
        t_step_length = time.time() - t_tmp
        
        if full_info:
            info['step_lengths'][it] = step_length
            info['backtrack_its'][it] =  backtrack_it
            info['t_step_length'][it] = t_step_length
            info['fun_evals'][it] = fun_old # We know that x_k has been accepted so store the evaluation. Possible we reject the determined step length
            
        # Early stopping based on step length/Armijo condition
        if not backtrack_satisfied:
            info['stop_it'] = it+1 # Store stopping iterations
            info['convg_flag']  = -1
            info['t_total'] = time.time() - t_start_loop
            if full_info: 
                info['t_iteration_total'][it] =  time.time() - t_iteration_start
                info = truncate_info(info,it)
            else: 
                info['grad_norms'] = grad_norm.detach()
            pbar.close()
            return x, info
        
        x  = x_new  # Did not stop early so perform gradient step
        
        if constraint_fun is not None: # Apply any projections/constraints
            x = constraint_fun(x)
        
        if full_info: 
            info['t_iteration_total'][it] =  time.time() - t_iteration_start
        if store_iters: info['all_iters'].append(x.detach().clone())
        
        # Update progress bar
        pbar_info = (desc + ' | Grad comp time:{:.4f}s, Log grad norm:{:.3f}  |  '.format(t_grad,torch.log10(grad_norm.detach().cpu())) + 
                    'Log step length:{:.5f}, Backtrack its:{:}, Backtrack time:{:.4f}s'.format(np.log10(step_length),backtrack_it,t_step_length))
        pbar.postfix = (pbar_info)
        pbar.update()
        
    info['stop_it'] = it+1 # Store stopping iterations
    info['t_total'] = time.time() - t_start_loop
    info['convg_flag']  = 1
    if not full_info: info['grad_norms'] = grad_norm

    info['t_total'] = time.time() - t_start_loop
    return x, info


#Gradient descent
def nesterov( fun, x0, max_its=10000, grad_tol=1e-12, verbose=False,
                     desc='', store_iters=False, full_info = False, constraint_fun = None):
    """
    Nesterov's accelerated gradient descent. Solves
        min_x  fun(x)
    where fun is differentiable function with gradient property grad. Algorithm is terminated 
    if either max_its number of iterations is reached, gradient norm is smaller than grad_tol,
    or proposed step length by backtracking linessearch is smalelr than 1e-14
    
    INPUTS
    fun : function with single input, returns real number. Has property grad, 
        Single input and returns output the same dimension as the input
    x0 : starting point of the algoithm.
    max_its : Maximum number of iterations to run for
    grad_tol : Gradient tolerance for early stopping
    verbose : Flag for whether information about the updates should be stated
    desc : Extra information to be prefixed onto printed statements
    store_iters: Flag for whether the iterates should be stored
    
    OUTPUTS
    x - most recent iteration of the algorithm
    flag - statement regarding the convergence of gradient descent
    """
    x = x0.detach().clone()
    step_next = 1e-2 # Initial step length for backtracking algorithm for first iteration of gradient descent
    if full_info:
        store_iters = True
        info = { # Dictionary for extra information about GD iterations
            't_grad_computation':200*np.ones(max_its), # array of length of times for each gradient computation
            'grad_norms':200*np.ones(max_its), # array of the gradient norms at each iteration
            'step_lengths':200*np.ones(max_its),# array of step lengths used for each iteration of algorithm
            'backtrack_its':200*np.ones(max_its), # array of number of backtracking iterations used for each step length
            't_step_length':200*np.ones(max_its), # array of length of time for each step length computation
            't_iteration_total':100*np.ones(max_its), # array of length of overall time for each iteration
            'fun_evals':50*np.ones(max_its), # array of length of evaluation of cost function
            'stop_it':None} # array of length of overall time for each iteration
    else:  
        info = { # Dictionary of minimal amount of information
                'grad_norms' : None, # Final gradiennt norm of solver
                't_total' : None, # Total computation time
                'stop_it' : None # Stopping iteration of algorithm
            }
    if store_iters: info['all_iters'] = [x0] # History of considered points
    
    pbar = tqdm(total=max_its, disable=not(verbose), position=0, ncols=0)
    it = 0
    
    beta = 0  # Nesterov acceleration parameter
    # lam = 0 # Variable to determine Nesterov parameter
    x_old = torch.empty_like(x0)
    
    t_start_loop = time.time()
    fun_new = None 
    
    for it in range(max_its):
        t_iteration_start = time.time()
        
        # Compute gradient
        grad_tmp = fun.grad(x)
        if grad_tmp is tuple: grad_x , grad_info = grad_tmp
        else: grad_x = grad_tmp
        t_grad = time.time() - t_iteration_start
        
        # Save gradient-based information
        grad_norm = grad_x.norm()
        if full_info:
            info['t_grad_computation'][it] = t_grad
            info['grad_norms'][it] =  grad_norm.detach()
        
        # Early stopping based on gradient norm
        if grad_norm <= grad_tol:
            info['stop_it'] = it # Store stopping iterations
            info['convg_flag'] = 0
            info['t_total'] = time.time() - t_start_loop
            if full_info: 
                info['t_iteration_total'][it] =  time.time() - t_iteration_start
                info['fun_evals'][it] = fun_new # Function evaluation at current iteration
                info['step_lengths'] = info['step_lengths'][:it]
                info['backtrack_its'] = info['backtrack_its'][:it]
                info['t_step_length'] = info['t_step_length'][:it] 
                info = truncate_info(info,it)
            else: 
                info['grad_norms'] = grad_norm.detach()
            pbar.close()
            return x, info
        
        beta = 0.1
        # beta = it/(it+1)
        if it > 0:
            x += beta * (x-x_old)
        # print(beta)
        # Determine step length backtracking linesearch
        t_tmp = time.time()
        step_length, step_next, backtrack_satisfied, backtrack_it, x_new, fun_new, fun_old = backtracking(fun, grad_x, x,search_dir=-grad_x, step_size=step_next, fun_eval = fun_new)
        t_step_length = time.time() - t_tmp
        
        # Update Nesterov parameters
        # lamold = lam
        # lam = (1+np.sqrt(1+4*lam**2))/2 
        # beta = (lamold - 1)/lam
        
        if full_info:
            info['step_lengths'][it] = step_length
            info['backtrack_its'][it] =  backtrack_it
            info['t_step_length'][it] = t_step_length
            info['fun_evals'][it] = fun_old # We know that x_k has been accepted so store the evaluation. Possible we reject the determined step length
            
        # Early stopping based on step length/Armijo condition
        if not backtrack_satisfied:
            info['stop_it'] = it # Store stopping iterations
            info['convg_flag']  = -1
            info['t_total'] = time.time() - t_start_loop
            if full_info: 
                info['t_iteration_total'][it] =  time.time() - t_iteration_start
                info = truncate_info(info,it)
            else: 
                info['grad_norms'] = grad_norm.detach()
            pbar.close()
            return x, info
        
        x_old = x
        x  = x_new  # Did not stop early so perform gradient step
        if constraint_fun is not None: # Apply any projections/constraints
            x = constraint_fun(x)
        if full_info: 
            info['t_iteration_total'][it] =  time.time() - t_iteration_start
        if store_iters: info['all_iters'].append(x.detach().clone())
        
        # Update progress bar
        pbar_info = (desc + ' | Grad comp time:{:.4f}s, Log grad norm:{:.3f}  |  '.format(t_grad,torch.log10(grad_norm.detach().cpu())) + 
                    'Log step length:{:.5f}, Backtrack its:{:}, Backtrack time:{:.4f}s'.format(np.log10(step_length),backtrack_it,t_step_length))
        pbar.postfix = (pbar_info)
        pbar.update()
        
    info['stop_it'] = it # Store stopping iterations
    info['t_total'] = time.time() - t_start_loop
    info['convg_flag']  = 1
    if not full_info: info['grad_norms'] = grad_norm

    info['t_total'] = time.time() - t_start_loop
    return x, info

def backtracking(fun, grad_s, s, search_dir, step_size=1e-2, fun_eval=None):
    """
    Implementation of backtracking linesearch
    fun(s_{k+1}) < f(s_k) - tau* step_size * \| grad(s_k)\|**2
    step_size is the initial step size to consider
    
    INPUTS:
    fun : function of single input that returns a float
    grad_s : Evaluation of gradient at current iterate
    s : current iterate that we want to improve upon
    step_size : Initial step size to be considered
    fun-eval : Optional evaluation of fun(s) 
    
    OUTPUTS:
    step_size - step size that should be utilised
    step_next - step size that should be considered for the next run of backtracking linesearch
    flag - Flag for if the Armijo condition was satisfied
    it - number of backtracking iterations to termiante
    """
    # Specify relevant quantities for the Armijo condition
    if fun_eval is None: fun_s = fun(s)
    else: fun_s = fun_eval
    grad_search_inner = torch.dot(grad_s.ravel(),search_dir.ravel())
    discount = 1e-2
    
    factor_increase = 1.2
    factor_decrease = 0.4
    
    it = 0
    
    # step_size = 1e-1
    # return step_size, 0, True, None, s + step_size*search_dir, None, None
    
    while step_size > 1e-14:
        s_tmp = s + step_size*search_dir # Candidate gradient update
        fun_snew = fun(s_tmp)
            # Check armijo condition
        if fun_snew < fun_s - discount*step_size*grad_search_inner:
            step_next = factor_increase * step_size
            return step_size, step_next, True, it, s_tmp, fun_snew, fun_s
        else: 
            step_size *= factor_decrease
            it += 1
            
    return step_size, 0, False, it, s_tmp, fun_snew, fun_s
            



#%%% BFGS

def _bfgs_search_dir(s, y, g, hessinv_estimate=None):
    r"""Compute the descent direction used in LBFGS using the two loop approach
    """
    assert len(s) == len(y)
    
    device = g.device
    
    q = g.detach().clone()
    alphas = torch.zeros(len(s), device=device)
    rhos = torch.zeros(len(s) , device=device)
    
    for i in reversed(range(len(s))):
        rhos[i] =  1.0 / torch.dot (y[i].ravel() , s[i].ravel())
        alphas[i] = rhos[i] * torch.dot(s[i].ravel() , q.ravel())
        q -= alphas[i]*y[i]
    
    
    if hessinv_estimate is not None: # Utilize initial inverse Hessian if provided
            q = hessinv_estimate(q)
    
    for i in range(len(s)):
        beta = rhos[i] * torch.dot(y[i].ravel(),q.ravel())
        q += s[i] * (alphas[i] - beta)
        
    return - q


def bfgs(fun, x0, max_its=10000, grad_tol=1e-12, verbose=False, 
         num_store=None, hessinv_estimate = None, desc='', store_iters = False,
         full_info = False, store_hessian = False):
    """
    BFGS method. Solves
        min_x  fun(x)
    where fun is differentiable function with gradient grad. Algorithm is terminated 
    if either max_its number of iterations is reached or the gradient norm is smaller than grad_tol
    
    INPUTS
    fun : function with single input, returns real number. Has .grad property
    x0 : starting point of the algoithm. Point is not overwritten.
    max_its : Maximum number of iterations to run for
    grad_tol : Gradient tolerance for early stopping
    verbose : Flag for whether information about the updates should be stated
    num_store: Number of vectors to store for limited memory BFGS
    hessinv_estimate: Initial estimate of the inverted Hessian matrix
    desc : Extra information to be prefixed onto printed statements
    
    OUTPUTS
    x - most recent iteration of the algorithm
    flag - statement regarding the convergence of gradient descent
    """
    x = x0.detach().clone() # Do not overwrite starting point
    step_next = 1. # Starting step length for backtracking algorithm for first iteration of gradient descent
    
    if full_info:
        info = { # Dictionary for extra information about GD iterations
            't_grad_computation':np.empty(max_its), # array of length of times for each gradient computation
            'grad_norms':torch.empty(max_its), # array of the gradient norms at each iteration
            'step_lengths':torch.empty(max_its),# array of step lengths used for each iteration of algorithm
            'backtrack_its':np.empty(max_its), # array of number of backtracking iterations used for each step length
            't_step_length':np.empty(max_its), # array of length of time for each step length computation
            't_iteration_total':np.empty(max_its), # array of length of overall time for each iteration
            'fun_evals':np.empty(max_its), # array of length of evaluation of cost function
            'stop_it':None, # array of length of overall time for each iteration
            'all_iters':None}
    else:  
        info = { # Dictionary of minimal amount of information
                'grad_norms' : None, # Final gradiennt norm of solver
                't_total' : None, # Total computation time
                'stop_it' : None # Stopping iteration of algorithm
            }
    if store_iters: info['all_iters'] = [x0] # History of considered points
    
    t_start_loop = time.time()
    pbar = tqdm(total=max_its, disable=not(verbose), position=0, ncols=0)
    
    ss, ys = [], []
    grad_x = fun.grad(x)
    fun_new = fun(x)
    
    for it in range(max_its):
        t_iteration_start = time.time()
        
        
        t_grad = time.time() - t_iteration_start
        try:    grad_norm = grad_x.norm()
        except: grad_norm = np.linalg.norm(grad_x)
        if full_info:
            info['t_grad_computation'][it] =  t_grad
            info['grad_norms'][it] = grad_norm.detach()
        
        if grad_norm <= grad_tol: # Norm of gradient tolerance reached
            info['t_total'] = time.time() - t_start_loop
            info['stop_it'] = it # Store stopping iterations
            info['convg_flag'] = 0
            if full_info: 
                info['t_iteration_total'][it] =  time.time() - t_iteration_start
                info['fun_evals'][it] = fun_new # Function evaluation at current iteration
                info['step_lengths'] = info['step_lengths'][:it]
                info['backtrack_its'] = info['backtrack_its'][:it]
                info['t_step_length'] = info['t_step_length'][:it] 
                info = truncate_info(info, it)
            else: info['grad_norms'] = grad_norm.detach()
            
            if store_hessian:
                info['ss'] = ss
                info['ys'] = ys
            return x, info
        
        search_dir = _bfgs_search_dir(ss, ys, grad_x, hessinv_estimate) 
        # search_dir2 = -inverse_hessian(fun, x, grad_x)
        # print(torch.norm(search_dir - search_dir2)/search_dir2.norm())
        
        
        
        step_length, step_next, backtrack_satisfied, backtrack_it, x_new, fun_new, fun_old = backtracking(fun, grad_x, x, search_dir=search_dir, step_size=step_next, fun_eval = fun_new)
        t_step_length = time.time() - t_grad - t_iteration_start
        if full_info:
            info['fun_evals'][it] = fun_old # Function evaluation at current iteration
            info['step_lengths'][it] = step_length
            info['backtrack_its'][it] = backtrack_it
            info['t_step_length'][it] = t_step_length
        # Check early stopping criteria
        if not backtrack_satisfied:
            info['stop_it'] = it+1 # Store stopping iterations
            info['convg_flag']  = -1
            if not full_info: info['grad_norms'] = grad_norm.detach()
            info['t_total'] = time.time() - t_start_loop
            if full_info: 
                info['t_iteration_total'][it] =  time.time() - t_iteration_start
                info = truncate_info(info,it)                    
            pbar.close()
            
        
            if store_hessian:
                info['ss'] = ss
                info['ys'] = ys
            
            return x, info
                
        # Early stopping not satisfied, update parameter
        x  = x_new  # Gradient descent
        if store_iters: info['all_iters'].append(x.detach().clone())
        
        grad_x, grad_diff = fun.grad(x) , grad_x
        grad_diff = grad_x - grad_diff
        
        # Update Hessian
        ys.append(grad_diff)
        ss.append(step_length * search_dir)
        if num_store is not None:
            # Throw away factors if they are too many.
            ss = ss[-num_store:]
            ys = ys[-num_store:]
            
        if full_info: info['t_iteration_total'][it] =  time.time() - t_iteration_start
        
        
        # Update progress bar
        pbar_info = ('Grad comp time:{:.4f}s, Log grad norm:{:.3f}  |  '.format(t_grad,torch.log10(grad_norm)) + 
                    'Log step length:{:.5f}, Backtrack its:{:}, Backtrack time:{:.4f}s'.format(np.log10(step_length),backtrack_it,t_step_length))
        pbar.postfix = (pbar_info)
        pbar.update()
    
    info['stop_it'] = it+1 # Store stopping iterations
    info['convg_flag']  = 1
    if not full_info: info['grad_norms'] = grad_norm
    
    if store_hessian:
        info['ss'] = ss
        info['ys'] = ys

    info['t_total'] = time.time() - t_start_loop
    return x, info




#%% Newtons method
from linear_ops import construct_matrix

def inverse_hessian(fun, x, z):
    '''
    Solve fun.hess(x,w) = z for w. Do this by constructing the hessian matrix explicitly.
    '''
    hess_op_vec = lambda w: fun.hess(x,w.view(x.shape)).ravel()
    hess_mat =construct_matrix(hess_op_vec, x.ravel().shape[0], device=x.device)
    
    hess_inv = torch.linalg.inv(hess_mat)
    out_vec = hess_inv @ z.ravel()
    
    return out_vec.view(x.shape)
    
    
    

def newton(fun, x0, max_its=10000, grad_tol=1e-12, verbose=False, 
         num_store=None, hessinv_estimate = None, desc='', store_iters = False,
         full_info = False, store_hessian = False):
    """
    Newtons method. Solves
        min_x  fun(x)
    where fun is differentiable function with gradient grad. Algorithm is terminated 
    if either max_its number of iterations is reached or the gradient norm is smaller than grad_tol
    
    INPUTS
    fun : function with single input, returns real number. Has .grad property
    x0 : starting point of the algoithm. Point is not overwritten.
    max_its : Maximum number of iterations to run for
    grad_tol : Gradient tolerance for early stopping
    verbose : Flag for whether information about the updates should be stated
    num_store: Number of vectors to store for limited memory BFGS
    hessinv_estimate: Initial estimate of the inverted Hessian matrix
    desc : Extra information to be prefixed onto printed statements
    
    OUTPUTS
    x - most recent iteration of the algorithm
    flag - statement regarding the convergence of gradient descent
    """
    
    x = x0.detach().clone() # Do not overwrite starting point
    step_next = 1. # Starting step length for backtracking algorithm for first iteration of gradient descent
    
    if full_info:
        info = { # Dictionary for extra information about GD iterations
            't_grad_computation':np.empty(max_its), # array of length of times for each gradient computation
            'grad_norms':torch.empty(max_its), # array of the gradient norms at each iteration
            'step_lengths':torch.empty(max_its),# array of step lengths used for each iteration of algorithm
            'backtrack_its':np.empty(max_its), # array of number of backtracking iterations used for each step length
            't_step_length':np.empty(max_its), # array of length of time for each step length computation
            't_iteration_total':np.empty(max_its), # array of length of overall time for each iteration
            'fun_evals':np.empty(max_its), # array of length of evaluation of cost function
            'stop_it':None, # array of length of overall time for each iteration
            'all_iters':None}
    else:  
        info = { # Dictionary of minimal amount of information
                'grad_norms' : None, # Final gradiennt norm of solver
                't_total' : None, # Total computation time
                'stop_it' : None # Stopping iteration of algorithm
            }
    if store_iters: info['all_iters'] = [x0] # History of considered points
    
    t_start_loop = time.time()
    pbar = tqdm(total=max_its, disable=not(verbose), position=0, ncols=0)
    
    grad_x = fun.grad(x)
    fun_new = None
    
    for it in range(max_its):
        t_iteration_start = time.time()
        
        
        t_grad = time.time() - t_iteration_start
        try:    grad_norm = grad_x.norm()
        except: grad_norm = np.linalg.norm(grad_x)
        if full_info:
            info['t_grad_computation'][it] =  t_grad
            info['grad_norms'][it] = grad_norm.detach()
        
        if grad_norm <= grad_tol: # Norm of gradient tolerance reached
            info['t_total'] = time.time() - t_start_loop
            info['stop_it'] = it # Store stopping iterations
            info['convg_flag'] = 0
            if full_info: 
                info['t_iteration_total'][it] =  time.time() - t_iteration_start
                info['fun_evals'][it] = fun_new # Function evaluation at current iteration
                # info['step_lengths'][it] = None
                # info['backtrack_its'][it] = None
                # info['t_step_length'][it] = None
                info = truncate_info(info, it)
            else: info['grad_norms'] = grad_norm.detach()

        
        search_dir = -inverse_hessian(fun, x, grad_x)
        
        
        step_length, step_next, backtrack_satisfied, backtrack_it, x_new, fun_new, fun_old = backtracking(fun, grad_x, x, search_dir=search_dir, step_size=step_next, fun_eval = fun_new)
        t_step_length = time.time() - t_grad - t_iteration_start
        if full_info:
            info['fun_evals'][it] = fun_old # Function evaluation at current iteration
            info['step_lengths'][it] = step_length
            info['backtrack_its'][it] = backtrack_it
            info['t_step_length'][it] = t_step_length
        # Check early stopping criteria
        if not backtrack_satisfied:
            info['stop_it'] = it # Store stopping iterations
            info['convg_flag']  = -1
            if not full_info: info['grad_norms'] = grad_norm.detach()
            info['t_total'] = time.time() - t_start_loop
            if full_info: 
                info['t_iteration_total'][it] =  time.time() - t_iteration_start
                info = truncate_info(info,it)                    
            pbar.close()
            return x, info
                
        # Early stopping not satisfied, update parameter
        x  = x_new  # Gradient descent
        if store_iters: info['all_iters'].append(x.detach().clone())
        
        grad_x, grad_diff = fun.grad(x) , grad_x
        grad_diff = grad_x - grad_diff

            
        if full_info: info['t_iteration_total'][it] =  time.time() - t_iteration_start
        
        
        # Update progress bar
        pbar_info = ('Grad comp time:{:.4f}s, Log grad norm:{:.3f}  |  '.format(t_grad,torch.log10(grad_norm)) + 
                    'Log step length:{:.5f}, Backtrack its:{:}, Backtrack time:{:.4f}s'.format(np.log10(step_length),backtrack_it,t_step_length))
        pbar.postfix = (pbar_info)
        pbar.update()
    
    info['stop_it'] = it # Store stopping iterations
    info['convg_flag']  = 1
    if not full_info: info['grad_norms'] = grad_norm


    info['t_total'] = time.time() - t_start_loop
    return x, info


#%%% Solve Hessian system

def cg(A, b, x0=None, tol=1e-6, max_its=10000, verbose=False, full_info = True):
    # Conjugate gradient for solving a symmetric positive definite linear system.
    #  A x = b
    if x0 is None:
        x0 = torch.zeros_like(b)
    t_start = time.time()
    x = x0.detach().clone()
    r = b - A(x)
    p = r # First basis vector
    
    r_norm = r.norm()
    info = {
            'stop_it': None,
            'convg_flag':None,
            't_total':None}
    if full_info:
        info['r_norms'] =  torch.empty(max_its+1, device=x0.device)
        info['r_norms'][0] = r_norm.detach()
        info['xs'] = [x]
        info['basis_coefs'] = torch.empty(max_its, device=x0.device)
    
    if r_norm < tol:
        info['stop_it'] = 0
        info['r_norms'] = r_norm.detach()
        info['convg_flag'] = 0
        info['t_total'] = time.time() - t_start
        return x0, info
    for it in tqdm(range( max_its ), disable=not(verbose)):
        Ap = A(p)
        alpha = r_norm**2 / torch.inner(p.ravel(), Ap.ravel())#torch.sum(p * Ap) # Coefficient of new basis vector
        x = x + alpha * p # Update solution
        r = r - alpha * Ap # Update residual
        r_new_norm = r.norm()
        if full_info: 
            info['r_norms'][it+1] = r_new_norm.detach() # Store residual
            info['xs'].append(x)
            info['basis_coefs'][it] = alpha
        
        if r_new_norm < tol:
            info['stop_it'] = it+2
            info['convg_flag'] = 0
            if full_info: 
                info['r_norms'] = info['r_norms'][:(it+2)]
                info['basis_coefs'] = info['basis_coefs'][:it+1]
            else: 
                info['r_norms'] = r_new_norm.detach()
            info['t_total'] = time.time() - t_start
            return x, info
        beta = (r_new_norm / r_norm)**2
        p = r + beta * p
        r_norm = r_new_norm
    
    if not full_info:info['r_norms'] = r_new_norm.detach()
    info['stop_it'] = max_its
    info['convg_flag'] = 1
    info['t_total'] = time.time() - t_start
    return x, info



# Start of algorithm
def MINRES(A, b, x0=None, tol=1e-6, max_its=10000, verbose=False, full_info = True,
           store_basis_vecs = False, U = None, C = None, stop_crit = None):
    '''
    Apply (R)MINRES algorithm to solve A x = b. Using Lanczos vectors, solves an equivalent tridiagonal system
        y_k = argmin_y \| T_k y - beta e_1 \|^2
    using Givens rotations which exploits structural similarity between T_k and T_{k-1}. 
    A change of basis is also performed, involving the QR decomp of T_k, to avoid
    the storage of all Lanczos vectors in the solution update
        x_k = x_0 + V_k y_k
    and instead we have 
        x_k = x_{k-1} + c * \tilde v_k .

    Parameters
    ----------
    A : Linear operator. Evaluated with a function call A(x)
    b : Tensor, right hand side of the linear system
    x0 : Initial guess of solution. THe zero vector if not specified.
    tol : Early stopping tolerance for the residual norm
    max_its : Maximum number of iterations
    verbose : Flag for displaying progress
    full_info : Flag for storing all previous iterations/basis vectors
    U : Tensor U is recycle space and is such that C=AU satisfies C^TC=I
    C : Tensor C = AU which satisfies satisfies C^TC=I
    M : Tensor M for which the stopping criterion is \|M (A x  - b)\|_2 < tol

    Returns
    -------
    x : Determined solution of the linear system
    info : dictionary containing relevant information of outcome of the solve

    '''
    # print('Stop Hess tol: {}'.format(tol))
    # Based on Large-scale topology optimization using preconditioned Krylov subspace methods with recycling
    t_start = time.time()
    if U is not None:
        recycling = True
    else: 
        recycling = False
    if x0 is None:
        x0 = torch.zeros_like(b)
        
    # We are going to apply Hermitian Lanczos process to get (I-Q)AV_j = V_{j+1} T_j
    x = x0.detach().clone()
    r = b - A(x) # Initial residual
    
    # Adjust initial point and residual for recycled space
    if recycling:
        rtilde = C.T @ r
        x += U @ rtilde
        r -= C @ rtilde # Apply projection to residual
    
    # Variables associated with Lanczos vectors
    beta, oldbeta = r.norm(), 0.
    v1old= r.clone()
    v2old = torch.empty_like(r) # Not utilised first iteration
    vnext = r.clone()
    
    # Variables associated with Givens rotations
    s, c = 0., 1.
    olds_beta, oldc_beta = 0., 0.
    
    zeta = r.norm() # Initial residual norm/the final component of the RHS of the Tridiagonal system
    vtilde2old = torch.zeros_like(b)
    vtilde1old = torch.zeros_like(b)
    btilde2old = torch.zeros_like(b)
    btilde1old = torch.zeros_like(b)
    
    # Vectors for updating the residual vector (needed for alternative stopping criterion)
    Avtilde2old = torch.zeros_like(b)
    Avtilde1old = torch.zeros_like(b)
    Abtilde2old = torch.zeros_like(b)
    Abtilde1old = torch.zeros_like(b)
    
    
    info = {'stop_it': max_its,
            'convg_flag':-1,
            't_total':None}
    if full_info:
        store_basis_vecs = True
        info['r_norms'] = torch.empty(max_its+1, device=x0.device)
        info['r_norms'][0] = zeta
        info['basis_coefs'] = torch.empty(max_its, device=x0.device)
        info['xs'] = [x.clone()]
    if store_basis_vecs:
        info['basis_vecs'] = []
        
    
    for it in range(max_its):
        #===============================================================================
        # ==== Create new basis vector for Krylov subspace             
        v = vnext/beta # v_k
        # Start constructing v_{k+1}
        Avtilde = A(v) #  Reuse memory of residual for new basis vector
        vnext = Avtilde.clone()
        
        # Orthogonalise against recycled space C = AU
        if recycling:
            bhat = C.T @ Avtilde # Reuse memory
            Abtilde = C @ bhat
            vnext -= Abtilde # Make orthogonal to recycle C
            bhat = U @ bhat
            
       
        # Use modified Gram-Schmidt orthognalization and fact that since A is symmetric
        #   we only need to consider the previous 2 Lanczos vectors
        # Orthogonalise against v_k
        alpha = torch.inner(v.ravel(),Avtilde.ravel()) 
        vnext -= alpha/beta*v1old 
        # Orthogonalise against v_{k-1}
        if it > 0:
            vnext -= beta/oldbeta * v2old # r is now the unnormalised v_{k+1}
           
        # Assign variables to prepare for next iteration
        oldbeta = beta 
        beta = vnext.norm() # beta_{k+1}
        v2old = v1old   # Has norm given by oldbeta
        v1old = vnext   # Has norm given by beta
        
        
        # We now need to compute the QR factorisation of the new tridiagonal matrix.
        #   In particular, we need the final column of R 
        #====================================================================================
        #====   Apply the previous two rotations to the new column of the tridiagonal matrix,
        #           ( zeros(k-2), beta_k, alpha_k, beta_{k+1} ).
        #       Resulting vector will have entries denoted
        #           ( zeros(k-3), gamma, delta, epsi, beta_{k+1} )
        gamma =    olds_beta             # gamma_k   =           s_{k-2} beta_k
        delta =  c*oldc_beta + s*alpha   # delta_k   =   c_{k-1} c_{k-2} beta_k + s_{k-1} alpha_k
        epsi  = -s*oldc_beta + c*alpha   # epsilon_k = - s_{k-1} c_{k-2} beta_k + c_{k-1} alpha_k
        
        olds_beta = s * beta
        oldc_beta = c * beta
        
        #====================================================================================
        #==== Determine new plane rotation to annihilate beta_{k+1} and perform said rotation
        #       Final column of R_k will have entries 
        #       (zeros(k-3), gamma, delta, epsibar, 0) where
        epsibar = torch.sqrt(epsi**2 + beta**2)
        #    and was attained with a rotation given by
        c = epsi / epsibar # c_k
        s = beta / epsibar # s_k  
        
        # ==== Apply kth rotation to the basis vector
        update_coef = c * zeta # Coefficient of the solution update vector
        zeta = - s * zeta # Update the residual/final entry of the Tridiagonal system RHS
        
        #==== Update solution
        # x_k = x_{k-1} + c_k zeta_k (I - UC^TA) \tilde v_k
        #   where \tilde v_k is the final column of  V_k R_k^{-1}.
        vtilde = (v - delta * vtilde1old - gamma * vtilde2old) / epsibar
        x +=  update_coef * vtilde 
        vtilde2old = vtilde1old
        vtilde1old = vtilde
        
        
        # Update residual vector
        Avtilde = (Avtilde - delta * Avtilde1old - gamma * Avtilde2old) / epsibar
        r -= update_coef * Avtilde
        Avtilde2old = Avtilde1old
        Avtilde1old = Avtilde
        
        # Update recycling space part of solution
        if recycling:
            btilde = (bhat - delta * btilde1old - gamma * btilde2old ) / epsibar # equal to UC^TA \tilde v_k
            x -= update_coef * btilde
            btilde2old = btilde1old
            btilde1old = btilde
            
            Abtilde = (Abtilde - delta * Abtilde1old - gamma * Abtilde2old) / epsibar
            r += update_coef * Abtilde
            Abtilde2old = Abtilde1old
            Abtilde1old = Abtilde
        
        rnorm = torch.abs(zeta) # Residual
        
        #=== Stopping criteria        
        if full_info: 
            info['basis_coefs'][it] = update_coef
            info['r_norms'][it+1] = rnorm
            info['xs'].append(x.clone())
        if store_basis_vecs:
            info['basis_vecs'].append(v)
            
        # Potentially custom stopping criteria
        if stop_crit is not None:
            stop_flag = stop_crit(r,x)
        else:
            stop_flag = rnorm < tol # Residual sufficiently small
        
        if stop_flag:
            info['stop_it'] = it+2
            if full_info: 
                info['r_norms'] = info['r_norms'][:(it+2)]
                info['basis_coefs'] = info['basis_coefs'][:(it)]
            else: info['r_norms'] = zeta
            info['convg_flag'] = 0
            info['t_total'] = time.time() - t_start
            
            return x, info
            
    if not full_info:
        info['r_norms'] = zeta
    info['stop_it'] = max_its
    info['t_total'] = time.time() - t_start
    return x, info
    