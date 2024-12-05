# -*- coding: utf-8 -*-
"""
Class of bilevel - used to solve bilevel problems


Created on Fri Jul 29 13:55:27 2022

@author: sebsc
"""
#%% Load modules
import numpy as np
import scipy.linalg
import torch
import matplotlib.pyplot as plt
import time

from optimisers import cg, bfgs, gradient_descent, MINRES, nesterov
from functionals import Functional




#%% Default options

def populate_options(options):
    r"""
    Create a dictionary of all options where unspecified options have been included
    """
    defaults = {
            'll_solve': {
                'solver': 'BFGS',
                'max_its': 10000,
                'tol': 1e-8,
                'warm_start' : True,
                'verbose' : False,
                'store_iters': False,
                'num_store': None
            },
            'hess_sys': {
                'max_its': 10000,
                'tol': 1e-8,
                'warm_start' : True,
                'verbose' : False,
                'store_solns': False,
                'solver' : 'MINRES',
                'recycle_strategy' : None,
                'recycle_dim' : 5
            },
            'ul_solve': {
                'solver': 'GD',
                'max_its': 10000,
                'tol': 1e-8,
                'verbose' : True,
                'store_iters':True,
                'full_info':True,
                'num_store': None
            }
        }
    
    # Populate unspecified main options
    out = {**defaults, **options}
    
    # Populate unspecified sub options
    for key in options:
        out[key] = {**defaults[key] , **options[key]}
    return out


#%%% Solve singleton lower level problem
def solve_lower_level_singleton(params, y, ll_fun, x0=None, analytic=None, solver='GD',verbose=1,
                                max_its=10000, grad_tol=1e-12, device=None, store_iters=False,
                                full_info = False, num_store=None):
        """ Solves lower level problem for single data pair"""
        if analytic is not None:
            return analytic( params, y), 'User supplied function used' 
        
        if device is None:
            device = params.device
        
        ll_fun.y = y
        ll_fun.params = params
        
        if x0 is None:
            x0 = torch.zeros(ll_fun._xshape , device=device)
        
        if solver =='GD':
            return gradient_descent(ll_fun, x0, max_its=max_its, grad_tol=grad_tol, desc='LL ', verbose=verbose, store_iters=store_iters, full_info = full_info)
        else:
            return bfgs(ll_fun, x0, max_its=max_its, grad_tol=grad_tol, desc='LL ', verbose=verbose, store_iters=store_iters, full_info = full_info, 
                        num_store=num_store,  store_hessian=True)
      

#%% Solve numerous lower level problems
def solve_lower_levels(params, ys, ll_fun, x0s=None, analytic=None, solver='GD',verbose=True,
                                max_its=10000, grad_tol=1e-8, device=None, store_iters=False,
                                full_info = False, num_store=None):
    #Assume all data is given as either s-n-n tensor where s is number of samples
    if device is None: device = ys.device
    dat_num = len(ys)
    
    if x0s is None: x0s = torch.zeros(dat_num, ll_fun._xshape[0],ll_fun._xshape[1],device=device)
    x_outs = torch.empty(dat_num,ll_fun._xshape[0],ll_fun._xshape[1],device=device)
    info_outs = [None for _ in range(dat_num)]
    
    for ind in range(dat_num):
        x_outs[ind], info_outs[ind] =  solve_lower_level_singleton(params, ys[ind], ll_fun, x0=x0s[ind], analytic=analytic, solver=solver,verbose=verbose,
                                        max_its=max_its, grad_tol=grad_tol, device=device, store_iters=store_iters,
                                        full_info = full_info, num_store=num_store )
        
    return x_outs, info_outs

#%%%

def calculate_recon_and_costs(params, ys , ul_fun, ll_fun, x0s=None, analytic=None, solver='GD',verbose=1,
                                max_its=10000, grad_tol=1e-8, device=None, alg_options=None, store_iters=False):
    """Calculate the associated reconstruction, upper level cost, and lower level cost for a given parameters"""
    recons, infos = solve_lower_levels(params, ys, ll_fun, x0s, analytic=analytic, solver=solver,verbose=verbose,
                                    max_its=max_its, grad_tol=grad_tol, device=device, store_iters=store_iters, num_store=None)

    ul_cost = ul_fun(recons)
    if recons.ndim == 3: 
        ll_cost = torch.empty(len(recons))
        for ind in range(len(recons)):
            ll_cost[ind] = ll_fun(recons[ind])
    else: 
        ll_cost = ll_fun(recons)
    
    return recons,  ul_cost, ll_cost, infos




#%% Solve Hessian system



def solve_hessian_systems(xs, ul, ll, w0s=None, tol=1e-6, max_its=10000, verbose=False, full_info=True, solver='MINRES', 
                          recycle_strategy=None, recycle_dim=15, recycle_basis_vecs=[None] ):
    """Solve   H(x) * w  =  grad ul(x)
      where H is the Hessian of the lower level problem and l is the upper level cost function
      
      recycle_basis_vecs : list of length data_num where each entry is a torch tensor in which 
          each row is a vector over which the recycling subspace should be selected from.
      """
    
    data_num = len(xs)
    ul_grad =  ul.grad(xs)  # gradient of upper level with respect to x evaluated at reconstruction
    
    # Determine starting point
    if w0s is None: w0s = torch.zeros(xs.shape , device=xs.device)
    
    w_outs = torch.empty(w0s.shape,device=xs.device)
    hess_infos = [None for _ in range(data_num)]
    
    # Solve Hessian system for each datapair
    for ind in range(data_num):
        if solver == 'CG':
            hessian = lambda z : ll.hess(xs[ind],z) # Lambda function that returns Hessian around x applied to z
            w, hess_infos[ind] = cg(hessian,ul_grad[ind], x0=w0s[ind], tol=tol, max_its=max_its, verbose=verbose, full_info=full_info)
            w_outs[ind] = w
        elif solver =='MINRES':
            # Specify Hessian
            hessian = lambda z : ll.hess(xs[ind],z.view(xs[0].shape)).ravel() # Lambda function that returns Hessian around x applied to z
            
            # Determine recycling space
            if recycle_basis_vecs[0] is None: # No recycling is being performed
                U = None
                C = None
                M = None
            else:
                C, U, M, t_make_dense, t_decomp, t_save = determine_recycle_space(hessian, recycle_basis_vecs[ind], recycle_dim=recycle_dim, strategy=recycle_strategy)

            # Solve Hessian system
            w, hess_infos[ind] = MINRES(hessian,ul_grad[ind].ravel(), x0=w0s[ind].ravel(), tol=tol, max_its=max_its,
                                        verbose=verbose, full_info=full_info, U=U, C=C)
            w_outs[ind] = w.view(xs[0].shape)
            hess_infos[ind]['U'] = U
    return w_outs, hess_infos


def hessian_fun(H, n, device):
    '''
    Perform interesting queries to properties of the Hessian. Purely for research

    '''
    # Dense form
    Hmat = torch.empty(n,n,device=device)
    for basis_ind in range(n):
        basis_vec = torch.zeros(n,device=device)
        basis_vec[basis_ind] = 1.
        Hmat[:,basis_ind] = H(basis_vec)
    
    # Compute eigenvalues
    evals, _ = torch.linalg.eigh(Hmat) # Hessian is SPD
    plt.plot(evals.flip([0]).cpu())
    plt.title('Hessian evals')
    
    
def make_HU_orthogonal(H,U):
    HU = torch.empty_like(U)
    for ind in range(U.shape[-1]):
        HU[:,ind] = H(U[:,ind]) 
                   
    C, R = torch.linalg.qr(HU) # Skinny QR factorisation. C is n-by-s, R is s-by-s.
    Uactual = U @ torch.linalg.inv(R)
    return C, Uactual    


def construct_dense_matrix(op,m, n=None, device='cpu'):
    if n is None: n=m
    mat = torch.empty(m,n,device=device)
    for ind in range(n):
        e = torch.zeros(n,device=device)
        e[ind] = 1.
        mat[:,ind] = op(e)
    return mat

def determine_recycle_space(H_op, basis_vecs, recycle_dim, strategy=None,
                            size='small', J_op=None, pdim=None):
    '''
    Want to determine a recycle space U such that C = AU satisfies C^T C = I.
    Matrix C is what is utilised in practice so is what shall be returned.
    
    
    op represents the application of a symmetric positive definite matrix
    
    Parameters
    ----------
    basis_vecs : List of unravelled tensors that can be used in recycling. The rows are the L.I. vectors.
    strategy : Method in which the new subspace is determined
    k : Dimension of the recycled subspace to be determined
    size : 'small' or 'large'. Whether to select the k vectors of the smallest or largest coefs

    Returns
    -------
    C : Tensor C = AU, satisfies C^TC=I
    U : Tensor 
    M : None or Tensor, used for alternative stopping criterion of RMINRES

    '''
    # if device is None: device = basis_vecs.device
    if strategy is None: # No recycling, return zero vectors
        # zero_vec = torch.zeros(len(basis_vecs[0]),1, device=basis_vecs[0].device)
        return None, None, None, 0, 0,0
    
    ResPrecond = None
    t_save = 0
    t0 = time.time()
    
    # We need to compute Wt@H@W regardless of the (generalised)(harmonic) Ritz choice rule
    if isinstance(basis_vecs,list):
        W = torch.stack(basis_vecs, dim=1)
    else: W = basis_vecs
    device = W.device
    Wdim = W.shape[-1]
    
    WtHW = torch.empty_like(W) # transpose(AV)
    for ind in range(Wdim):
        WtHW[:,ind] = H_op(W[:,ind])
    WtHW = W.T @ WtHW 
    
    # Cannot recycle more than the dimension of W = [ U V ]
    number_of_recycle_vectors = min(recycle_dim, Wdim)
    
    if strategy == 'ritz':
        # Use the k eigenvectors corresponding to the k smallest eigenvalues WtHW
        t1 = time.time()
        t_make_dense = t1 - t0

        if size=='large':
            _, evecs = scipy.linalg.eigh(WtHW.cpu(), subset_by_index=(Wdim-number_of_recycle_vectors, Wdim-1))
            U_tmp = W @ torch.tensor(evecs,device=device) # Use eigenvectors of the k largest evals
        elif size=='small':
            _, evecs = scipy.linalg.eigh(WtHW.cpu(), subset_by_index=(0,number_of_recycle_vectors-1))
            U_tmp = W @ torch.tensor(evecs,device=device) # Use eigenvectors of the k smallest evals
        elif size =='mix':
            # Use mixture of smallest and largest
            num_small = int(np.floor(number_of_recycle_vectors/2))
            _, evecs_large = scipy.linalg.eigh(WtHW.cpu(), subset_by_index=(Wdim-num_small, Wdim-1))
            _, evecs_small = scipy.linalg.eigh(WtHW.cpu(), subset_by_index=(0,num_small))
            
            U_tmp = W @ torch.hstack([torch.tensor(evecs_small,device=device), torch.tensor(evecs_large, device=device)])
        
        t_decomp = time.time() - t1
        
    elif strategy =='gritz_left':
        # if matlab_eng == None: 
        #     matlab_eng = matlab.engine.start_matlab()
        JW = torch.empty( pdim, W.shape[-1], device=device)
        for ind in range(W.shape[-1]):
            JW[:,ind] = J_op(W[:,ind])
        t_make_dense = time.time() - t0

        V_J,V_H,X,S_J,S_H, t_decomp, t_save = compute_gsvd(JW, WtHW)

        if size=='large':
            U_tmp= W @ V_H[:,-number_of_recycle_vectors:]
            ResPrecond =  (S_J @ torch.diag(1/torch.diag(S_H)))[-number_of_recycle_vectors:,-number_of_recycle_vectors:] @ V_H[:,-number_of_recycle_vectors:].T @ W.T
        
        if size=='small':
            U_tmp= W @ V_H[:,:number_of_recycle_vectors]
            ResPrecond =  (S_J @ torch.diag(1/torch.diag(S_H)))[:number_of_recycle_vectors,:number_of_recycle_vectors] @ V_H[:,:number_of_recycle_vectors].T @ W.T
            
        elif size=='mix':
            # Use mixture of smallest and largest
            num_small = int(np.floor(number_of_recycle_vectors/2))
            
            U_tmp = W @ torch.hstack([V_H[:,:num_small] , V_H[:,-num_small:]])
            ResPrecond =  ((S_J @ torch.diag(1/torch.diag(S_H)))[:num_small,:num_small] @ V_H[:,:num_small].T 
                           + (S_J @ torch.diag(1/torch.diag(S_H)))[-num_small:,-num_small:] @ V_H[:,-num_small:].T )@ W.T
            
        del V_J, V_H, X, S_J,S_H
            
    elif strategy =='gritz_right':
        
        JW = torch.empty( pdim, W.shape[-1], device=device)
        for ind in range(W.shape[-1]):
            JW[:,ind] = J_op(W[:,ind])            
        t_make_dense = time.time() - t0

        V_J,V_H,X,S_J,S_H, t_decomp, t_save = compute_gsvd(JW, WtHW)
        Xinv = torch.linalg.inv(X)
        
        if size=='large':
            U_tmp= W @ Xinv[:,-number_of_recycle_vectors:]
            
            ResPrecond =  (S_J @ torch.diag(1/torch.diag(S_H)))[-number_of_recycle_vectors:,-number_of_recycle_vectors:] @ V_H[:,-number_of_recycle_vectors:].T @ W.T
        
        elif size=='small':
            U_tmp= W @ Xinv[:,:number_of_recycle_vectors] 
            ResPrecond =  (S_J @ torch.diag(1/torch.diag(S_H)))[:number_of_recycle_vectors,:number_of_recycle_vectors] @ V_H[:,:number_of_recycle_vectors].T @ W.T
            
        elif size=='mix':
            # Use mixture of smallest and largest
            num_small = int(np.floor(number_of_recycle_vectors/2))
            
            U_tmp = W @ torch.hstack([Xinv[:,:num_small] , Xinv[:,-num_small:]])
            ResPrecond =  ((S_J @ torch.diag(1/torch.diag(S_H)))[:num_small,:num_small] @ V_H[:,:num_small].T 
                           +(S_J @ torch.diag(1/torch.diag(S_H)))[-num_small:,-num_small:] @ V_H[:,-num_small:].T )@ W.T
            
        del V_J, V_H, X, S_J,S_H
        
        
    elif strategy =='gritz_both':
        
        JW = torch.empty( pdim, W.shape[-1], device=device)
        for ind in range(W.shape[-1]):
            JW[:,ind] = J_op(W[:,ind])
        t_make_dense = time.time() - t0


        V_J,V_H,X,S_J,S_H, t_decomp, t_save = compute_gsvd(JW, WtHW)
        Xinv = torch.linalg.inv(X)
        if size=='large':
            U_tmp= W @ (V_H[:,-number_of_recycle_vectors:] + Xinv[:,-number_of_recycle_vectors:])/2
            ResPrecond = (S_J @ torch.diag(1/torch.diag(S_H)))[-number_of_recycle_vectors:,-number_of_recycle_vectors:] @ V_H[:,-number_of_recycle_vectors:].T @ W.T
        
        if size=='small':
            U_tmp= W @ (V_H[:,:number_of_recycle_vectors] + Xinv[:,:number_of_recycle_vectors])/2
            ResPrecond =  (S_J @ torch.diag(1/torch.diag(S_H)))[:number_of_recycle_vectors,:number_of_recycle_vectors] @ V_H[:,:number_of_recycle_vectors].T @ W.T
            
        elif size=='mix':
            # Use mixture of smallest and largest
            num_small = int(np.floor(number_of_recycle_vectors/2))
            U_tmp = W @ torch.hstack([Xinv[:,:num_small] + V_H[:,:num_small] , Xinv[:,-num_small:] + V_H[:,-num_small:]])/2
            ResPrecond =  ((S_J @ torch.diag(1/torch.diag(S_H)))[:num_small,:num_small] @ V_H[:,:num_small].T 
                           +(S_J @ torch.diag(1/torch.diag(S_H)))[-num_small:,-num_small:] @ V_H[:,-num_small:].T )@ W.T
            
        del V_J, V_H, X, S_J,S_H
            
    elif strategy == 'harmonic_ritz':
        
        if isinstance(basis_vecs,list):
            W = torch.stack(basis_vecs, dim=1)
        else: W = basis_vecs
        device = W.device
        Wdim = W.shape[-1]
        
        WtHHW = torch.empty_like(W) 
        for ind in range(Wdim):
            WtHHW[:,ind] = H_op(H_op(W[:,ind]))
        WtHHW = W.T@ WtHHW
        t_make_dense = time.time() - t0
        
        if size=='large':
            _, evecs = scipy.linalg.eigh(a=WtHW.cpu(),b=WtHHW.cpu(), subset_by_index=(Wdim-number_of_recycle_vectors, Wdim-1))
            U_tmp = W @ torch.tensor(evecs,device=device) # Use eigenvectors of the k largest evals
        elif size=='small':
            _, evecs = scipy.linalg.eigh(a=WtHW.cpu(),b=WtHHW.cpu(), subset_by_index=(0,number_of_recycle_vectors-1))
            U_tmp = W @ torch.tensor(evecs,device=device) # Use eigenvectors of the k smallest evals
        elif size =='mix':
            # Use mixture of smallest and largest
            num_small = int(np.floor(number_of_recycle_vectors/2))
            _, evecs_large = scipy.linalg.eigh(a=WtHW.cpu(),b=WtHHW.cpu(), subset_by_index=(Wdim-num_small, Wdim-1))
            _, evecs_small = scipy.linalg.eigh(a=WtHW.cpu(),b=WtHHW.cpu(), subset_by_index=(0,num_small))
            
            U_tmp = W @ torch.hstack([torch.tensor(evecs_small,device=device), torch.tensor(evecs_large, device=device)])
        t_decomp = time.time() - t0
    
    elif strategy == 'eig':
        # Consider full eigendecomposition of H
        
        H_mat = construct_dense_matrix(H_op, W.shape[0], device=device)
        t_make_dense = time.time() - t0

        if size=='large':
            _, evecs = scipy.linalg.eigh(H_mat.cpu(), subset_by_index=(W.shape[0]-recycle_dim, W.shape[0]-1))
            U_tmp = torch.tensor(evecs,device=device) # Use eigenvectors of the k largest evals
        elif size=='small':
            _, evecs = scipy.linalg.eigh(H_mat.cpu(), subset_by_index=(0,recycle_dim))
            U_tmp = torch.tensor(evecs,device=device) # Use eigenvectors of the k smallest evals
        elif size =='mix':
            # Use mixture of smallest and largest
            num_small = int(np.floor(recycle_dim/2))
            _, evecs_large = scipy.linalg.eigh(H_mat.cpu(), subset_by_index=(W.shape[0]-num_small, W.shape[0]-1))
            _, evecs_small = scipy.linalg.eigh(H_mat.cpu(), subset_by_index=(0,num_small))
            
            U_tmp = torch.hstack([torch.tensor(evecs_small,device=device), torch.tensor(evecs_large, device=device)])
        t_decomp = time.time() - t0
        
    elif strategy == 'gsvd_left':
        
        H_mat = construct_dense_matrix(H_op, W.shape[0], device=device)
        J_mat = construct_dense_matrix(J_op, pdim, W.shape[0], device=device )
        t_make_dense = time.time() - t0

        V_J,V_H,X,S_J,S_H, t_decomp, t_save = compute_gsvd(J_mat, H_mat)
        
        if size=='large':
            U_tmp=  V_H[:,-number_of_recycle_vectors:]
        if size=='small':
            U_tmp=  V_H[:,:number_of_recycle_vectors]
            
        del V_J, V_H, X, S_J,S_H
            
    elif strategy =='gsvd_right':
    
    
        H_mat = construct_dense_matrix(H_op, W.shape[0], device=device)
        J_mat = construct_dense_matrix(J_op, pdim, W.shape[0], device=device )
        t_make_dense = time.time() - t0

        V_J,V_H,X,S_J,S_H, t_decomp, t_save = compute_gsvd(J_mat, H_mat)
        
        if size=='large':
            U_tmp= torch.linalg.inv(X)[:,-number_of_recycle_vectors:]
        elif size=='small':
            U_tmp=  torch.linalg.inv(X)[:,:number_of_recycle_vectors]  
        del V_J, V_H, X, S_J,S_H
        
    elif strategy =='gsvd_both':
        H_mat = construct_dense_matrix(H_op, W.shape[0], device=device)
        J_mat = construct_dense_matrix(J_op, pdim, W.shape[0], device=device )
        t_make_dense = time.time() - t0
    
        V_J,V_H,X,S_J,S_H, t_decomp, t_save = compute_gsvd(J_mat, H_mat)
        
        Xinv = torch.linalg.inv(X)
        if size=='large':
            U_tmp= W @ (V_H[:,-number_of_recycle_vectors:] + Xinv[:,-number_of_recycle_vectors:])/2
        if size=='small':
            U_tmp= W @ (V_H[:,:number_of_recycle_vectors] + Xinv[:,:number_of_recycle_vectors])/2
        elif size=='mix':
            # Use mixture of smallest and largest
            num_small = int(np.floor(number_of_recycle_vectors/2))
            U_tmp = W @ torch.hstack([Xinv[:,:num_small] + V_H[:,:num_small] , Xinv[:,-num_small:] + V_H[:,-num_small:]])/2
            
            
        del V_J, V_H, X, S_J,S_H
    else:
        raise ValueError('Unrecognised value of recycling strategy: '+strategy)
    
    C, U = make_HU_orthogonal(H_op,U_tmp)
    return C, U, ResPrecond, t_make_dense, t_decomp, t_save
        

def compute_gsvd(A, B):
    """ 
    Compute the GSVD of the pair (A,B) where we assume B is square and non-singular.
    
    Follows the construction of "Towards a Generalized Singular Value Decomposition"
    by Paige and Saunders, 1981.
    """
    
    t0 = time.time()
    m = len(A)
    s = len(B)
    stack = torch.cat([A,B])
    
    P,D,Q = torch.svd(stack)
    
    P11 = P[:m]
    P21 = P[m:]
    
    V_A, S_A, W =  torch.linalg.svd(P11, full_matrices=True)
    
    if m > s:
        V_A, S_A, W = V_A[:,:s].flip(1), S_A.flip(0), W.flip(0)
        S_A = torch.diag(S_A)
    else:
        V_A, S_A, W = V_A.flip(1), S_A.flip(0), W.flip(0)    
        S_A = torch.diag(S_A, diagonal=s-m)[:len(S_A)]
        
    X = W @ torch.diag(D) @ Q.T
            
    V_B, S_B = torch.linalg.qr(P21@W.T)
    S_B = S_B.diag().diag() # Matrix of just the diagonal entries
          
    t1 = time.time()
      
    return V_A, V_B, X, S_A, S_B, t1-t0, 0
        
    

#%%% Compute hypergradient


def compute_hypergradient( params, ys, ul_fun, ll_fun, recons=None,  x0s = None, w0s=None, optns={}, device=None, 
                          store_solns=False, recycle_basis_vecs = [None]):
        """Compute hypergradient of the upper level cost function
        INPUTS
        param - Current parameter considered"""
        optns = populate_options(optns)
        
        # Solve lower level problem utilising current parameters
        if recons is None:
            recons,  ll_infos = solve_lower_levels(params, ys, ll_fun, x0s=x0s,
                    solver=optns['ll_solve']['solver'],verbose=optns['ll_solve']['verbose'], max_its=optns['ll_solve']['max_its'], 
                    grad_tol=optns['ll_solve']['tol'], device=device, store_iters=optns['ll_solve']['store_iters'],
                    full_info = False, num_store=optns['ll_solve']['num_store'])
        else: 
            ll_infos = None # Suppress error. Will reuse info from the function object call property later        
        
        # Solve linear system involving the Hessian
        ws, hess_infos = solve_hessian_systems(recons,ul_fun,ll_fun, w0s=w0s, tol=optns['hess_sys']['tol'], solver=optns['hess_sys']['solver'], 
                max_its=optns['hess_sys']['max_its'], verbose=optns['hess_sys']['verbose'], 
                full_info=True,recycle_strategy=optns['hess_sys']['recycle_strategy'],  
                recycle_basis_vecs = recycle_basis_vecs, recycle_dim=optns['hess_sys']['recycle_dim'])
        
        # Calculate gradient of upper level with respect to parameters and calculate new recycled space
        ul_grad = torch.zeros_like(params)
        # Recycling variables
        for ind in range(len(ys)):
            ul_grad -= ll_fun.grad_jac_adjoint(recons[ind], ws[ind])
         
        return ul_grad, recons, ws,  {'ll_info':ll_infos, 'hess_info':hess_infos}




#%%% Bilevel cost function


def create_W_space(V,U):
    if isinstance(V,list):
        V = torch.stack(V, dim=1)      
    elif isinstance(V,dict):
        V = torch.stack(V['basis_vecs'], dim=1)
    if U is None:
        W = V
    else:
        W, _ = torch.linalg.qr(torch.hstack([V, U])) # Make orthogonal
        
    return W

class bilevel_fun(Functional):
    r"""
    Functional used by gradient descent to solve the bilevel problem
    """
    def __init__(self, ul_fun, ll_fun, ys, optns={}, x0s=None, w0s = None):
        super().__init__()
        self.ul_fun = ul_fun # Upper level cost function
        self.ll_fun = ll_fun # Lower level cost function
        self.optns = populate_options(optns)
        self.ys = ys    # Noisy measurements
        self.x0s = x0s # Initial guess for lower level solves
        self.w0s = w0s # Initial guess for Hessian system solves
        self.data_num = len(ys)
        self.recycle_basis_vecs = [None for _ in range(self.data_num)]  # Temporary storage for recycling space
        self.info = {
                    'll_call_infos':[],# Information of reconstruction for function calls (e.g. in backtracking liensearch)
                    'll_infos':[], # Information of reconstruction in the gradient computation
                    'hess_infos':[], # Information of the Hessian lienar system solve
                    'ul_info':None} # Information of optimiser applied to the bielvel function
        self._ll_solves_counter = None
        self._ll_solves_per_ul_it = []
        self._recons = None # Temporary storage of recently computed reconstructions. Avoid computing it twice in gradient call and backtracking linesearch
        self._first_eval_call_made = False # Flag for first iteration of algorithm
        
        
        if self.optns['ll_solve']['store_iters']:
            self._all_recons = []
     
    @property
    def xexact(self):
        return self.ul_fun.xexact
    @xexact.setter
    def xexact(self,newxexact):
        self.ul_fun.xexact = newxexact
    
    @property
    def params(self):
        return self.ll_fun.params
    @params.setter
    def params(self, newparams):
        self.ll_fun.params= newparams      
        
    def __call__(self,params):
        # self.params = params
        
        # After the first evaluation call in the algorithm, have to solve new lower level
        if self._first_eval_call_made: 
            self._ll_solves_counter += 1
            # Calculate reconstructions of all datapairs
            recons, recons_info = solve_lower_levels(params, self.ys, self.ll_fun, x0s=self.x0s, 
                                        solver=self.optns['ll_solve']['solver'],
                                        verbose=self.optns['ll_solve']['verbose'],
                                        max_its=self.optns['ll_solve']['max_its'],
                                        grad_tol=self.optns['ll_solve']['tol'],
                                        num_store=self.optns['ll_solve']['num_store'],
                                        full_info = False,
                                        device=params.device)
            self.info['ll_call_infos'].append(recons_info)
            self._recons = recons.clone()
            
        else: # For the first evaluation call can reuse reconstruction made from the gradient call
            recons = self._recons.clone()
            self.info['ll_call_infos'].append(self.info['ll_infos'][0])
            self._first_eval_call_made = True
            
        if  self.optns['ll_solve']['warm_start']: self.x0s = recons.clone() # Warm start for backtracking linesearch solves as well
        
        return self.ul_fun(recons) 
    
    def grad(self,params):
        '''
        Construct the hypergradient of the bilevel cost function at params
        '''
        # self.params = params
        self._ll_solves_per_ul_it.append(self._ll_solves_counter)
        # Exploit the fact that the most recent call of the function was for the succesful iterate of backtracking, and so we can reuse the associated reconstruction
        ul_grad, recons, ws, infos =  compute_hypergradient(params, self.ys, self.ul_fun, self.ll_fun, optns=self.optns,
                                    recons=self._recons, w0s=self.w0s, device=params.device, recycle_basis_vecs=self.recycle_basis_vecs) 
        
        # Update recycling variables. Construct W = [ V U ] where V is recent Krylov subspace and U the recycle space
        if self.optns['hess_sys']['recycle_strategy'] is not None:
            for data_ind in range(self.data_num):
                self.recycle_basis_vecs[data_ind] = create_W_space(V=infos['hess_info'][data_ind], U=infos['hess_info'][data_ind]['U'])
        
        
        self._ll_solves_counter = 0 # Call of gradient means the start of new iteration of the algorithm so reset counter
            
        if self._first_eval_call_made: # Reused reconstruction from a previous function evaluation call
            infos['ll_info'] = self.info['ll_call_infos'][-1]
        else: # Had to solve lower level (this occurs only on the first gradient call of the algorithm)
            self._ll_solves_counter += 1
            self._recons = recons
        
        self.info['ll_infos'].append(infos['ll_info'])
            
        if self.optns['ll_solve']['warm_start']: self.x0s = recons # Warm start lower level
        if self.optns['hess_sys']['warm_start']: self.w0s = ws # Warm start Hessian system
        if self.optns['ll_solve']['store_iters']: self._all_recons.append(recons)
        self.info['hess_infos'].append(infos['hess_info'])
        return ul_grad
    
    def __repr__(self):
        return 'Bilevel problem with upper level '+str(self.ul_fun) + ' and lower level '+str(self.ll_fun)
    


   

#%% Solve bilevel problem using gradient descent
def bilevel_gradient_descent( ul_fun, ll_fun, ys, x0s=None, w0=None, analytic=None,
                   optns={}, device=None): 
    """Apply gradient descent to upper level cost function using hypergradients"""
    optns = populate_options(optns)
    
    fun = bilevel_fun(ul_fun, ll_fun, ys, optns=optns,x0s=x0s)
    solver = optns['ul_solve']['solver'] # Default opion is GD
    # Update to solutions and reconstruction history are done in compute_hypergradient calls
    if solver =='GD':
        param, ul_info = gradient_descent( fun, x0=fun.params, max_its=optns['ul_solve']['max_its'], grad_tol=optns['ul_solve']['tol'],
                                      verbose=optns['ul_solve']['verbose'], desc='UL ', store_iters=optns['ul_solve']['store_iters'],
                                      full_info = optns['ul_solve']['full_info'])
    elif solver =='BFGS':
        param, ul_info = bfgs( fun, x0=fun.params, max_its=optns['ul_solve']['max_its'], grad_tol=optns['ul_solve']['tol'],
                                      verbose=optns['ul_solve']['verbose'], desc='UL ', store_iters=optns['ul_solve']['store_iters'],
                                      full_info = optns['ul_solve']['full_info'])
    elif solver =='nesterov':
        param, ul_info = nesterov( fun, x0=fun.params, max_its=optns['ul_solve']['max_its'], grad_tol=optns['ul_solve']['tol'],
                                      verbose=optns['ul_solve']['verbose'], desc='UL ', store_iters=optns['ul_solve']['store_iters'],
                                      full_info = optns['ul_solve']['full_info'])
    else:
        raise ValueError('Choice of solver {} not recognised'.format(solver))
        
    if optns['ll_solve']['store_iters']:
        recons = fun._all_recons #+ [fun._recons]
    else:
        recons = fun._recons
    
    ul_info['ll_solves_per_ul_it'] = np.append(np.array(fun._ll_solves_per_ul_it[1:], dtype=np.dtype('int')), fun._ll_solves_counter)
    fun.info['ul_info']= ul_info # Information regarding convergence results of optimiser applied to the bilevel problem
    
    return param, recons, fun.info
