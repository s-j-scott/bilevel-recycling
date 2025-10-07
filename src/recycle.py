# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 23:33:40 2025

@author: sebsc

Functions used to employ recycling in solving a bilevel learning problem
"""
import torch
import numpy as np
import time
import scipy

def make_HU_orthogonal(H,U):
    HU = torch.empty_like(U)
    for ind in range(U.shape[-1]):
        HU[:,ind] = H(U[:,ind]) 
                   
    C, R = torch.linalg.qr(HU) # Skinny QR factorisation. C is n-by-s, R is s-by-s.
    Uactual = U @ torch.linalg.inv(R)
    return C, Uactual    

def construct_dense_matrix(op,m, n=None, device=None):
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
    
    
    H_op represents the application of a symmetric positive definite matrix
    
    Parameters
    ----------
    basis_vecs : List of unravelled tensors that can be used in recycling. The rows are the L.I. vectors.
    strategy : Method in which the new subspace is determined
    k : Dimension of the recycled subspace to be determined
    size : 'small' or 'large'. Whether to select the k vectors of the smallest or largest coefs
    J_op : Function handle that returns action of adjoint Jacobian on a vector
    pdim : Number of parameters in order to preallocate memory for J_op products.

    Returns
    -------
    C : Tensor C = AU, satisfies C^TC=I
    U : Tensor 
    M : None or Tensor, used for alternative stopping criterion of RMINRES

    '''
    if strategy is None: # No recycling, return zero vectors
        return None, None, None, 0, 0
    
    ResPrecond = None
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

        V_J,V_H,X,S_J,S_H, t_decomp = compute_gsvd(JW, WtHW)

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

        V_J,V_H,X,S_J,S_H, t_decomp = compute_gsvd(JW, WtHW)
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


        V_J,V_H,X,S_J,S_H, t_decomp = compute_gsvd(JW, WtHW)
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
            _, evecs = scipy.linalg.eigh(H_mat.cpu(), subset_by_index=(W.shape[0]-number_of_recycle_vectors, W.shape[0]-1))
            U_tmp = torch.tensor(evecs,device=device) # Use eigenvectors of the k largest evals
        elif size=='small':
            _, evecs = scipy.linalg.eigh(H_mat.cpu(), subset_by_index=(0,number_of_recycle_vectors-1))
            U_tmp = torch.tensor(evecs,device=device) # Use eigenvectors of the k smallest evals
        elif size =='mix':
            # Use mixture of smallest and largest
            num_small = int(np.floor(number_of_recycle_vectors/2))
            _, evecs_large = scipy.linalg.eigh(H_mat.cpu(), subset_by_index=(W.shape[0]-num_small, W.shape[0]-1))
            _, evecs_small = scipy.linalg.eigh(H_mat.cpu(), subset_by_index=(0,num_small))
            
            U_tmp = torch.hstack([torch.tensor(evecs_small,device=device), torch.tensor(evecs_large, device=device)])
        t_decomp = time.time() - t0
        
    elif strategy == 'gsvd_left':
        
        H_mat = construct_dense_matrix(H_op, W.shape[0], device=device)
        J_mat = construct_dense_matrix(J_op, pdim, W.shape[0], device=device )
        t_make_dense = time.time() - t0

        V_J,V_H,X,S_J,S_H, t_decomp = compute_gsvd(J_mat, H_mat)
        
        if size=='large':
            U_tmp=  V_H[:,-number_of_recycle_vectors:]
        if size=='small':
            U_tmp=  V_H[:,:number_of_recycle_vectors]
            
        del V_J, V_H, X, S_J,S_H
            
    elif strategy =='gsvd_right':
    
    
        H_mat = construct_dense_matrix(H_op, W.shape[0], device=device)
        J_mat = construct_dense_matrix(J_op, pdim, W.shape[0], device=device )
        t_make_dense = time.time() - t0

        V_J,V_H,X,S_J,S_H, t_decomp = compute_gsvd(J_mat, H_mat)
        
        if size=='large':
            U_tmp= torch.linalg.inv(X)[:,-number_of_recycle_vectors:]
        elif size=='small':
            U_tmp=  torch.linalg.inv(X)[:,:number_of_recycle_vectors]  
        del V_J, V_H, X, S_J,S_H
        
    elif strategy =='gsvd_both':
        H_mat = construct_dense_matrix(H_op, W.shape[0], device=device)
        J_mat = construct_dense_matrix(J_op, pdim, W.shape[0], device=device )
        t_make_dense = time.time() - t0
    
        V_J,V_H,X,S_J,S_H, t_decomp = compute_gsvd(J_mat, H_mat)
        
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
    return C, U, ResPrecond, t_make_dense, t_decomp
        

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
          
    t = time.time() - t0
      
    return V_A, V_B, X, S_A, S_B, t