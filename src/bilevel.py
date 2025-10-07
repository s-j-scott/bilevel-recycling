# -*- coding: utf-8 -*-
"""
Class of bilevel - used to solve bilevel problems


Created on Fri Jul 29 13:55:27 2022

@author: sebsc
"""
#%% Load modules
import torch
from time import time

from src.optimisers import cg, bfgs, gradient_descent, MINRES
from src.recycle import determine_recycle_space
from src.create_cost_functions import create_cost_functions

#%%% Solving the lower level problem
def solve_lower_level_singleton(params, A, y, ll_fun, xshape, x0=None, analytic=None, solver='L-BFGS',verbose=1,
                                max_its=10000, grad_tol=1e-12, device=None, store_iters=False,
                                full_info = False, num_store=None):
        """ Solves lower level problem for single data pair"""
        if analytic is not None:
            return analytic( params, y), 'User supplied function used' 
        
        if device is None:
            device = params.device
        
        ll_fun.y = y
        ll_fun.params = params
        ll_fun.A = A
        
        if x0 is None:
            x0 = torch.zeros(xshape , device=device)
        
        if solver =='L-BFGS':
            return bfgs(ll_fun, x0, max_its=max_its, grad_tol=grad_tol, desc='LL ', verbose=verbose, store_iters=store_iters, full_info = full_info, 
                        num_store=num_store)
        else:
            return gradient_descent(ll_fun, x0, max_its=max_its, grad_tol=grad_tol, desc='LL ', verbose=verbose, store_iters=store_iters, full_info = full_info)

def solve_lower_levels(params, As, ys, ll_fun, xshape, x0s=None, analytic=None, solver='GD',verbose=True,
                                max_its=10000, grad_tol=1e-8, device=None, store_iters=False,
                                full_info = False, num_store=None):
    #Assume all data is given as either s-n-n tensor where s is number of samples
    if device is None: device = params.device
    t_start   = time()
    dat_num   = len(ys)
    
    if x0s is None: x0s = torch.zeros(dat_num, xshape[0], xshape[1],device=device)
    x_outs = torch.empty(dat_num,xshape[0],xshape[1],device=device)
    # info_outs = [None for _ in range(dat_num)]
    info_outs = [None]*dat_num
    
    A = As[0]
    for ind in range(dat_num):
        if len(As)>1:
            A = As[ind]
        x_outs[ind], info_outs[ind] =  solve_lower_level_singleton(params, A, ys[ind], ll_fun, x0=x0s[ind], analytic=analytic, solver=solver,verbose=verbose,
                                        max_its=max_its, grad_tol=grad_tol, device=device, store_iters=store_iters,
                                        full_info = full_info, num_store=num_store )
        t_start += info_outs[ind]['t_total'] # Do not include the solver time
    t_logging = time() - t_start # Amount time logging information associated with lower level solves
    info_outs[0]['t_logging'] = t_logging
    return x_outs, info_outs

#%% Solving the Hessian systems
def solve_hessian_systems(xs, ul, ll, w0s=None, tol=1e-6, max_its=10000, verbose=False, full_info=True, solver='MINRES', 
                          recycle_strategy=None, recycle_dim=15, recycle_basis_vecs=[None],
                          stop_crit=None):
    """Solve   H(x) * w  =  grad ul(x)
      where H is the Hessian of the lower level problem and l is the upper level cost function
      
      recycle_basis_vecs : list of length data_num where each entry is a torch tensor in which 
          each row is a vector over which the recycling subspace should be selected from.
      """
    t_start = time()
    data_num = len(xs)
    ul_grad =  ul.grad(xs)  # gradient of upper level with respect to x evaluated at reconstruction
    
    # Determine starting point
    if w0s is None: w0s = torch.zeros(xs.shape , device=xs.device)
    
    w_outs = torch.empty(w0s.shape,device=xs.device)
    # hess_infos = [None for _ in range(data_num)]
    hess_infos = [None] * data_num
    
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
                                        verbose=verbose, full_info=full_info, U=U, C=C, stop_crit=stop_crit)
            w_outs[ind] = w.view(xs[0].shape)
            hess_infos[ind]['U'] = U
        t_start += hess_infos[ind]['t_total'] # Do not include time of the solver
    t_logging = time() - t_start # Time not associated with the Hessian solve
    hess_infos[0]['t_logging'] = t_logging
    return w_outs, hess_infos

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


#%%% Bilevel cost function

class bilevel_fun(torch.nn.Module):
    
    def __init__(self, setup, device):  
        super().__init__()      
        self.setup = setup
        self.device = device
        
        ul_fun, ll_fun = create_cost_functions(setup['problem_setup']['regulariser'],device=device)
        self.ul_fun = ul_fun
        self.ll_fun = ll_fun

        self.params = torch.nn.Parameter(ll_fun.R.params)
        
        self.batch_log = {} # Internal dict containing logging information of the recent batch
        
        self._internal_timer = None
                
    def forward(self, xexacts, As, ys, idxs, warm_starts):
        """ 
        Compute upper level cost over batch.
        
        xexacts - List of images
        As      - list of forward operators
        ys      - list of observed noisy measurements
        idxs    - list of unique ids of the batch items
        """
        self._internal_time = time()
        with torch.no_grad():
            self.batch_recon_infos = {}
            for i, data_id in enumerate(idxs):
                xexact = xexacts[i]
                A = As[i]
                y = ys[i]
                x0 = warm_starts[data_id].get('recon', None)
                if x0 is not None:
                    x0 = x0[0]
                
                recon, info = self.calculate_recon(xexact, A, y, x0)
                
                # Save recent reconstructions for gradient computation
                warm_starts[data_id]['recon'] = recon.detach().clone().unsqueeze(0)
                
                self.batch_recon_infos[data_id] = info
                
            ul_eval = self.ul_fun(warm_starts , xexacts)
            
        return ul_eval
    
    def calculate_recon(self, xexact, A, y, x0):
        ' Find reconstruction for single batch item'
        return solve_lower_level_singleton(self.params, A, y[0], self.ll_fun,
                                    xshape=xexact[0].shape, 
                                    x0 = x0,
                                    solver=self.setup['solver_optns']['ll_solve']['solver'],
                                    verbose=self.setup['solver_optns']['ll_solve']['verbose'],
                                    max_its=self.setup['solver_optns']['ll_solve']['max_its'], 
                                    grad_tol=self.setup['solver_optns']['ll_solve']['tol'], 
                                    device=self.device, 
                                    store_iters=self.setup['solver_optns']['ll_solve']['store_iters'],
                                    full_info = self.setup['solver_optns']['ll_solve']['full_info'], 
                                    num_store=self.setup['solver_optns']['ll_solve']['num_store'])

    def calculate_w(self, xexact, H, g, idx, w0):
        if self.setup['solver_optns']['hess_sys']['solver'] == 'MINRES':
            solver = MINRES
        else:
            solver = cg
        return  solver(H, g, 
                        x0=w0,
                        tol=self.setup['solver_optns']['hess_sys']['tol'], 
                        max_its=self.setup['solver_optns']['hess_sys']['max_its'],
                        verbose=self.setup['solver_optns']['hess_sys']['verbose'], 
                        )


    def grad(self, xexacts, As, ys, idxs, warm_starts, step=None):  
        
        grad = torch.zeros_like(self.params)
        self.batch_log = {i:{} for i in idxs}
        self.batch_tensors = {i:{} for i in idxs}
        with torch.no_grad():            
            # Consider each element in the batch separately
            for i, data_id in enumerate(idxs):
                
                t_start = time()
                
                xexact = xexacts[i]
                A = As[i]
                y = ys[i]
                
                # --- Solve the lower level ---
                # recon, info_recon = self.calculate_recon(xexact, A, y, idxs[i])
                recon, info_recon = warm_starts[data_id]['recon'][0], self.batch_recon_infos[data_id]
                
                # --- Solve the Hessian system ---
                g = self.ul_fun.grad(recon, xexact).ravel()
                # print(g.shape)
                H = lambda z : self.ll_fun.hess(recon, z.view(recon.shape)).ravel()
                w0 = warm_starts[data_id].get('w', None)
                w, info_w = self.calculate_w(xexact, H, g, data_id, w0)
                warm_starts[data_id]['w'] = w.detach().clone()
                
                # -- Accumulate adjoint Jacobian ---
                grad -= self.ll_fun.grad_jac_adjoint(recon, w.view(recon.shape))
                
                # -- Logging 
                t_end = time() - t_start + info_recon['t_total']
                self.batch_log[data_id]['recon'] = recon.detach().clone().cpu().unsqueeze(0)
                self.batch_log[data_id]['w']     =     w.detach().clone().cpu()
                self.batch_log[data_id]['stop_it_ll']   = info_recon['stop_it']
                self.batch_log[data_id]['t_ll']         = info_recon['t_total']
                self.batch_log[data_id]['stop_it_hess'] = info_w['stop_it']
                self.batch_log[data_id]['t_hess']       = info_w['t_total']
                self.batch_log[data_id]['t_total'] = t_end
                if step is not None:
                    self.batch_log[data_id]['step'] = step
        return grad