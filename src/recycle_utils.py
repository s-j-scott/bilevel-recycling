# -*- coding: utf-8 -*-
"""
Functions used in employing and evaluating the performance of recycling in
solving a provided sequence of linear systems.
"""
import torch
from tqdm import tqdm
import numpy as np

import time

from src.recycle import determine_recycle_space, construct_dense_matrix, compute_gsvd
from src.bilevel import create_W_space, solve_lower_levels
from src.create_cost_functions import create_cost_functions
from src.optimisers import MINRES, bfgs, cg, backtracking

#%% Functions for creating cost functions and hypergradients
def create_lower_level(setup, A, y, params, device):
    _, ll_fun = create_cost_functions(reg_optns=setup['problem_setup']['regulariser'],
                                      device=device)
    A = A.to(device)
    y = y.clone().to(device)
    params = params.clone().to(device)

    ll_fun.A = A
    ll_fun.y = y
    ll_fun.params = params.to(device)
    return ll_fun

class hessian_system_generator():
    def __init__(self, setup, xexacts, As, ys, params, recons, device):
        self.setup = setup
        self.xexacts = xexacts.to(device)
        self.A = As.to(device)
        self.y = ys.to(device)
        self.params = params
        self.recons = recons
        self.device= device
        
        self.n = xexacts.shape[0] * xexacts.shape[1]
        self.pdim = params[0].nelement()
        
    def __call__(self, ind=0, param=None):
        if param is None:
            param = self.params[ind]
        xexact = self.xexacts
        recon  = self.recons[ind].to(self.device)
        ll_fun = self.make_lower_level(ind=ind, param=param)
        
        J_op    = self.make_J(ind, param, ll_fun)
        hess_op = self.make_hessian(ind, param, ll_fun)
        g       = (recon - xexact).ravel()
        return J_op, hess_op, g


    def make_lower_level(self, ind=0, param=None):
        
        if param is None:
            param = self.params[ind]
        return create_lower_level(self.setup, self.A, self.y, param, self.device)

    def make_hessian(self, ind=0, param=None,  ll_fun = None):
        if ll_fun is None:
            ll_fun = self.make_lower_level(ind=ind, param=param)
        recon = self.recons[ind].to(self.device)
        hess_op = lambda w : ll_fun.hess(recon , w.view(recon.shape)).ravel()
        return hess_op
        
    def make_J(self, ind=0, param=None, ll_fun = None):
        if ll_fun is None:
            ll_fun = self.make_lower_level(ind=ind, param=param)
        recon = self.recons[ind].to(self.device)
        J_op = lambda w : ll_fun.grad_jac_adjoint(recon, w.view(recon.shape)).ravel()
        return J_op   
        

#%% TV reconstruction
def update_best(recon_best, cost_best, recon_new, cost_new):
    if cost_best > cost_new:
        return recon_new, cost_new
    else:
        return recon_best, cost_best

def calculate_best_TV(setup, xexact, A, y, ps, verbose=False):
    # Set cost functions
    optns = setup['solver_optns']['ll_solve']
    reg_optns={'name':'HuberTV', 'gamma':setup['problem_setup']['regulariser']['gamma'], 'L':'FiniteDifference2D'}
    # reg_optns = None
    _, ll_fun_tv = create_cost_functions(reg_optns=reg_optns, device=ps[0].device)
    ll_fun_tv.y = y
    ll_fun_tv.A = A
    x0 = torch.zeros_like(xexact)
    ul_fun = lambda x : torch.norm(x - xexact)**2 / 2
    ul_costs_tv = np.empty(len(ps))
    cost_best_tv = ul_fun(x0)
    recon_best_tv = x0

    for ind in tqdm(range(len(ps)), desc='TV reconstruction'):
        p = ps[ind]
        ll_fun_tv.params = p.detach()
        xrecon_tv, info_tv = bfgs(ll_fun_tv, x0, verbose=verbose, 
                                  max_its=optns['max_its'], num_store=optns['num_store'], grad_tol=optns['tol'])
        x0 = xrecon_tv
        
        ul_costs_tv[ind] = ul_fun(xrecon_tv)
        
        recon_best_tv, cost_best_tv = update_best(recon_best_tv, cost_best_tv, xrecon_tv, ul_costs_tv[ind])
    return recon_best_tv, cost_best_tv, ul_costs_tv

#%% Functions for solving sequence of Hessian problems for different recycling strategies

class stop_criterion:
    """Custom stopping criterion that can be used in the linear system solver.
    ALlows for stopping based upon:
        - Residual norm
        - Approximation of the hypergradient error
        - True hypergradient error
        - True hyptergradient relative error
    """
    def __init__(self, device, rec_dim, mode='res', ResPrecond=None, tol=1e-3, true_hg = None, J_op=None):
        self.rec_dim = rec_dim
        self.ResPrecond = ResPrecond
        self.tol = tol
        self.mode = mode
        self.device = device
        
        self.true_hg = true_hg # True hypergradients of current systems
        self.J_op = J_op
        
        assert mode in ('res', 'hg_err_approx', 'hg_err', 'hg_rerr'), f'Stopping criterion mode {mode} not valid.'

    def __call__(self, r, w=None):
        if self.mode =='res':
            return r.norm() < self.tol
        elif self.mode == 'hg_err_approx':
            if self.ResPrecond is not None:
                return torch.norm(self.ResPrecond @ r) < self.tol   
            else: 
                return torch.norm(r) < self.tol
            
        elif self.mode == 'hg_err':
            hg_err = torch.norm(self.J_op(w.to(self.device)) - self.true_hg.to(self.device)).cpu()
            return hg_err < self.tol
        
        elif self.mode == 'hg_rerr':
            hg_err = torch.norm(self.J_op(w.to(self.device)) - self.true_hg.to(self.device)).cpu() / torch.norm(self.true_hg).cpu()
            return hg_err < self.tol
        else: 
            print('\nI dont know the stopping criterion {self.mode}')
    

def solve_hess(setup,H,g,Hold,Vold,Uold,wold,recycle_dim,
               strategy='ritz',
               size='small', 
               solver='MINRES',
               outer_rec=True, 
               J_op=None, 
               pdim=None, 
               tol=1e-3, 
               full_info=False,
               verbose=False,
               stop_crit='res',
               true_hg=None,
               device=None):
    """
     Solve the Hessian system where
     - H : Function handle that returns the action of the Hessian of current iteration
     - g : Right hand side of the linear system
     - Hold : Function handle that returns the action of the Hessian of the previous Hessian. Used for outer recycling strategy
     - Vold : Krylov solution space of previous linear system solution
     - Uold : Recycle space used for the previous linear system
     - wold : Solution of previous linear system, used for warm starting.
     - recycle_dim : Maximum dimension of the recycle space to be determined
     - strategy : Type of vectors/decomposition to be used in recycling strategy
     - size : Which part of the spectrum selected recycle vectors should be associated with
     - solver : Numerical solver to use
     - outer_rec : Whether outer or inner recycling should be performed i.e. with respect to current or previous linear system respectively
     - J_op : Function handle that returns action of adjoint Jacobian 
     - pdim : Number of learned parameters, used for J_op product.
     - stop_crit : What type of stopping criterion to be used for the linear system solve.
     - true_hg : Vector of a high quality reference hypergradient. Potentially used in the stopping criterion.
    """
    if solver == 'MINRES':
        W = create_W_space(Vold,Uold) # Make W orthogonal
        Vold['basis_vecs'] = None # Make memory actually survive
        t0 = time.time()
        if outer_rec: # Outer recycling
            Huse = H
        else: # Inner recycling
            Huse = Hold
        C, U, ResPrecond, t_make_dense, t_decomp = determine_recycle_space(Huse, W, recycle_dim = recycle_dim, strategy=strategy, size=size, J_op=J_op, pdim=pdim)
        # print(f'USING DEVICE {device}')
        stop_crit_fun = stop_criterion(device, recycle_dim, mode=stop_crit, ResPrecond=ResPrecond, tol=tol, true_hg=true_hg, J_op=J_op)
        
        t1 = time.time()  
        w, info = MINRES(H, g, C=C, U=U, x0=wold, tol=tol, max_its=setup['solver_optns']['hess_sys']['max_its'],  verbose=verbose, full_info = full_info, store_basis_vecs=True, stop_crit=stop_crit_fun)
    
        info['t_find_rec_space'] = t1-t0
        info['t_make_dense'] = t_make_dense
        info['t_decomp'] = t_decomp
    else:
        w, info = cg(H, g, x0=wold, tol=tol, max_its=setup['solver_optns']['hess_sys']['max_its'],  verbose=verbose, full_info = full_info)
        info['t_find_rec_space'] = 0
        info['t_make_dense'] = 0
        info['t_decomp'] = 0
        W,U = None,None
    return w, info, W, U

def solve_hess_seqn(hessian_maker, recycle_dim, name=None, strategy='ritz',size='small',
                    warm_start=True, tol=1e-3,  full_info=False, device=None, verbose=False,
                    stop_crit='res', true_hgs = None, solver='MINRES', outer_rec=True, img_ind=0):
    """Returns:
        - List of solutions w
        - List of association Hessian system numerical solve info
        - List of dimension of Ws
        - List of determined Ws
        - List of determined recycle spaces
        """
    device = hessian_maker.device
    setup  = hessian_maker.setup
    pdim   = hessian_maker.pdim
    t0 = time.time()
    num_systems = len(hessian_maker.params)-1 # Do not solve Hessian associated with final parameter
    
    # all_ul_costs = np.empty(num_systems)
    if true_hgs is None: true_hgs = [None for _ in range(num_systems)]
    all_ws = []
    all_infos = []
    all_W_dims = []
    if full_info:
        all_U_spaces = []
        all_W_spaces = []
    
    p_bar = tqdm(range(num_systems), desc=name)
    # Solve first Hessian without recycling
    J_op, H, g =  hessian_maker(0)
    wold = torch.zeros(hessian_maker.n, device=device)
        
    stop_crit_fun = stop_criterion(device, recycle_dim, mode=stop_crit, ResPrecond=None, tol=tol, J_op=J_op, true_hg=true_hgs[0])

    if solver == 'MINRES':
        w, info = MINRES(H, g, x0=wold, tol=tol, max_its=setup['solver_optns']['hess_sys']['max_its'],  verbose=verbose, full_info = full_info, store_basis_vecs=True, stop_crit=stop_crit_fun)
    else:
        w, info = cg(H, g, x0=wold, tol=tol, max_its=setup['solver_optns']['hess_sys']['max_its'],  verbose=verbose, full_info = full_info)
    
    all_ws.append(w.cpu().clone())
    if strategy is not None:
        info['t_find_rec_space'] = 0
        info['t_make_dense'] = 0
        info['t_decomp'] = 0
        info['t_grand_total']= time.time() -  t0
    all_infos.append(info)
    U = None
    p_bar.update()
    
    # Employ recycling for subsequent Hessian systems
    for ind in (range(num_systems-1)):
        t0 = time.time()
        Hold =  H
        # Add 1 to index because we have already solved the first Hessian system
        J_op, H, g = hessian_maker(ind+1)
        if warm_start:
            w_start = w
        else:
            w_start = wold
        w, info, W, U =  solve_hess(setup,H,g,Hold,info,U,w_start,
                                    recycle_dim,
                                    strategy=strategy,
                                    size=size, 
                                    solver=solver,
                                    J_op=J_op, 
                                    pdim=pdim,
                                    tol=tol,
                                    verbose=verbose,
                                    full_info = full_info,
                                    stop_crit=stop_crit, 
                                    true_hg=true_hgs[ind+1], 
                                    outer_rec=outer_rec,
                                    device=device)
        
        info['t_grand_total']= time.time() -  t0
        all_ws.append(w.clone().cpu())
        all_infos.append(info)
        if solver == 'MINRES': all_W_dims.append(W.shape[-1])
        if full_info and U is not None:
            all_U_spaces.append(U.cpu().clone())
            all_W_spaces.append(W.cpu().clone())
        p_bar.update()
    p_bar.close()
    
    if full_info:
        return all_ws, all_infos, np.array(all_W_dims), all_W_spaces, all_U_spaces
    else:
        return all_ws, all_infos, np.array(all_W_dims)

def calc_hypgrad_rerrs(hessian_maker, ws, true_hgs, img_ind=0):   
    # Relative error in hypergrad
    device = hessian_maker.device
    
    hg_errs = torch.empty(len(ws))
    for ind in range(len(ws)):
        J_op = hessian_maker.make_J(ind)
        hg_errs[ind] = (torch.norm(true_hgs[ind].to(device) - J_op(ws[ind].to(device))) / true_hgs[ind].norm()).cpu()
    return hg_errs


def calc_hgs(hessian_maker, ws, img_ind=0): 
        """Calculate hypergradients explicitly. Primarily used one time
        for constructing the reference high quality hypergradients."""
        out = []
        device = hessian_maker.device
        for ind in range(len(ws)):
            J_op = hessian_maker.make_J(ind)
            out.append ( J_op(ws[ind].to(device)).cpu() ) # High accuracy hypergradient
        return out


def calc_RMINRES_cost(its, hess_cost, n, s):
    return np.array(hess_cost + 4*n + 6*n*s  + its*(hess_cost + 21*n + 6*n*s - s), dtype='int64')
        # return  6*n*s  + its*(hess_cost + 21*n + 6*n*s - s)
def calc_MINRES_cost(its, hess_cost, n):
    return np.array(hess_cost + 4*n + its*(hess_cost + 16*n), dtype='int64')

def calc_hess_cost(num_filt, kernel_size, n, forward_op_cost=0):
    ''' Cost of application of the Hessian for denoising Field of Experts'''
    return n + num_filt * 2*n * (3*kernel_size**2 -1) + 2*forward_op_cost

    
def do_recycling(hessian_maker, true_hgs, name = None, strategy=None, size=None, rec_dim=1, tol=1e-3, device=None,
                       warm_start=True, full_info=False, stop_crit='res', outer_rec=True, solver='MINRES'):
    """Solve entire sequence of Hessian systems for a specified recycle strategy and return all useful information associated with the solves"""
    if full_info:
        all_w, all_info, all_W_dims, all_W_spaces, all_U_spaces = solve_hess_seqn(hessian_maker, rec_dim, device=device, name=name, outer_rec=outer_rec, size=size, strategy=strategy, tol=tol, solver=solver, warm_start=warm_start, full_info=full_info, stop_crit=stop_crit, true_hgs=true_hgs)
    else:
        all_w, all_info, all_W_dims = solve_hess_seqn(hessian_maker, rec_dim, device=device, name=name, size=size, strategy=strategy, outer_rec=outer_rec, tol=tol, warm_start=warm_start, solver=solver, full_info=full_info, stop_crit=stop_crit, true_hgs=true_hgs)
    stop_it = np.array([inf['stop_it'] for inf in all_info]) 
    hg_rerr = calc_hypgrad_rerrs(hessian_maker, all_w, true_hgs )
    
    if full_info:
        return all_w, all_info, all_W_dims, stop_it, hg_rerr, all_W_spaces, all_U_spaces
    else:
        return all_w, all_info, all_W_dims, stop_it, hg_rerr

#%% Calculating new upper level cost 
class ul_fun_eval():
    def __init__(self,hessian_maker, ul_fun, x):
        self.x = x
        self.ul_fun = ul_fun
        self.ll_fun = hessian_maker.make_lower_level()
        self.As = hessian_maker.A 
        self.ys = hessian_maker.y
        self.setup = hessian_maker.setup
        self.device = hessian_maker.device
        
    def __call__(self,param):
        optn = self.setup['solver_optns']['ll_solve']
        xnew, _ = solve_lower_levels(param, self.As, self.ys, self.ll_fun, x0s=self.x, solver=optn['solver'],verbose=False,
                                        max_its=optn['max_its'], grad_tol=optn['tol'], device=self.device, store_iters=False,
                                        full_info = False, num_store=optn['num_store'])
        
        self.x = xnew # Warm starting
        return self.ul_fun(xnew)     

def calc_ul_cost(hessian_maker, x0, p, grad_p, ll_fun, ul_fun, init_stepsize):
    """
    See what the upper level cost would be if the hypergradient associated with the
    recycling method was actually employed to determine the update (rather than the non-recycling
    hypergradient which is actually employed)
    """
    ul_evaluator = ul_fun_eval(hessian_maker, ul_fun, x0)
    step_size,_,_,_,_,ul_cost,_ = backtracking(ul_evaluator, grad_p, p, grad_p, step_size=init_stepsize, fun_eval=None)
    # print(f"Ind:{ind} | {step_size:.4f} | {df_bl_info['ul_steplengths'][ind+1]}")
    return ul_cost

def calc_all_ul_cost(hessian_maker, ws, stepsizes, ul_fun):
    pshape = hessian_maker.params[0].shape
    ll_fun = hessian_maker.make_lower_level()
    
    all_ul_costs = []    
    for ind in tqdm(range(len(ws)-1), desc='Calc UL cost'):
        p = hessian_maker.params[ind]
        x0 = hessian_maker.recons[ind]
        J_op, H, g = hessian_maker(ind)
        hypgrad = J_op(ws[ind].to(hessian_maker.device)).view(pshape)
        
        cost = calc_ul_cost(hessian_maker, x0, p, hypgrad,  ll_fun, ul_fun, stepsizes[ind]).cpu().numpy()
        
        all_ul_costs.append(cost)
    return all_ul_costs

def compute_ul_costs(hessian_maker, ul_fun, result_dict, df_bl_info, verbose=True):
    stepsizes = df_bl_info['ul_steplengths'].values
    to_merge = {}
    if verbose:
        print('About to add an "ul_eval" property to the following results:')
        print(result_dict.keys())
    for method, result in result_dict.items():
        ws = result['w']
        out = calc_all_ul_cost(hessian_maker, ws, stepsizes, ul_fun)
        to_merge[method] = out
    for method, ul_eval in to_merge.items():
        result_dict[method]['ul_eval'] = ul_eval

# %% High accuracy computation

def compute_high_accuracy(hessian_maker, tol=1e-13, verbose=False):
    all_w_high, all_info_high, _ = solve_hess_seqn(hessian_maker,
                                                   recycle_dim=0, 
                                                   strategy=None, 
                                                   name='High accuracy',
                                                   tol=tol,
                                                   verbose=verbose)
    stop_it_high = np.array([inf['stop_it'] for inf in all_info_high]) 
    all_hgs_high = calc_hgs(hessian_maker, all_w_high)
    
    info = {'w':all_w_high, 
            'info':all_info_high,
            'stop_it':stop_it_high,
            'hgs':all_hgs_high}
    return info

def prepare_calculator(hessian_maker, true_hgs, include_info=True):
    device = hessian_maker.device
    def results_calculator(name = None, rec_dim=1, strategy=None, size=None, tol=None,
                           warm_start=True, full_info=True, stop_crit='res', outer_rec=True, solver='MINRES'):
        
        if tol is None:
            tol = hessian_maker.setup['solver_optns']['hess_sys']['tol']
        out = do_recycling(hessian_maker, true_hgs, strategy=strategy, size=size, rec_dim=rec_dim, warm_start=warm_start, full_info=full_info, stop_crit=stop_crit, 
                           outer_rec=outer_rec, solver=solver, name=name, tol=tol, device=device)
        
        info = {
            'w':out[0],
            'W_dim':out[2],
            'stop_it':out[3],
            'hg_rerr':out[4]}
        if include_info:
            info['info'] = out[1]
        if full_info:
            info['W_space'] = out[5]
            info['U_space'] = out[6]
        return info
    return results_calculator

def justify_recycling(setup, results, hessian_maker, img_ind=0, save=None):
    n = setup['problem_setup']['n']**2
    num_systems = len(results['w'])
    device = hessian_maker.device

    dif_recon = torch.empty(num_systems-1)
    norm_recon = torch.empty(num_systems-1)
    dif_hess = torch.empty(num_systems-1)
    norm_hess = torch.empty(num_systems-1)
    dif_rhs = torch.empty(num_systems-1)
    norm_rhs = torch.empty(num_systems-1)
    dif_w = torch.empty(num_systems-1)
    norm_w = torch.empty(num_systems-1)
    # hess_cond = torch.empty(num_systems-1) # Condition number of Hessian

    _, H, g_old = hessian_maker(0)
    H_mat_old = construct_dense_matrix(H, n, device=device)
    w_old = results['w'][0]
    recon_old = hessian_maker.recons[0][img_ind]
    
    for i in (tqdm(range(num_systems-1), desc='Comparing Hess systems')):
        _, H, g = hessian_maker(i+1)
        H_mat = construct_dense_matrix(H, n, device=device)
        w = results['w'][i+1]
        recon = hessian_maker.recons[i][img_ind]
        
        norm_recon[i] = recon_old.norm().cpu()
        norm_hess[i] = H_mat_old.norm().cpu()
        norm_rhs[i] = g_old.norm().cpu()
        norm_w[i] = w_old.norm().cpu()
        dif_recon[i] = torch.norm(recon - recon_old).norm().cpu()
        dif_hess[i] = torch.norm(H_mat-H_mat_old).norm().cpu()
        dif_rhs[i] = torch.norm(g_old-g).norm().cpu()
        # hess_cond[i] = torch.linalg.cond(H_mat)
        dif_w[i] = (w-w_old).norm()

        g_old = g.clone()
        recon_old = recon.clone()
        w_old = w.clone()
        H_mat_old = H_mat
    

    rel_difs =  {
      'dif_hess': dif_hess,  'norm_hess': norm_hess,
      'dif_rhs': dif_rhs, 'norm_rhs':norm_rhs,
       'dif_recon': dif_recon, 'norm_recon':norm_recon,
       'dif_w':dif_w, 'norm_w':norm_w,
       # 'hess_cond':hess_cond
       }
    
    if save is not None:
        torch.save(rel_difs, save)
    
    return rel_difs

#%% Functions to investigate the HG error approximation

def full_JH_gsvd(hessian_maker, J_op, H):
    device = hessian_maker.device
    Hmat = construct_dense_matrix(H, hessian_maker.n, device=device)
    Jmat = construct_dense_matrix(J_op,  hessian_maker.pdim, hessian_maker.n, device=device)

    V_J,V_H,Xt,S_J,S_H, _ = compute_gsvd(Jmat, Hmat)
    
    return V_J,V_H,Xt,S_J,S_H

def projected_JH_gsvd(hessian_maker, J_op, H, W):
    device = hessian_maker.device
    WtHW = torch.empty_like(W)
    JW = torch.empty( hessian_maker.pdim, W.shape[-1], device=device)
    for ind in range(W.shape[-1]):
        WtHW[:,ind] = H(W[:,ind])
        JW[:,ind] = J_op(W[:,ind])
        
    WtHW = W.T @ WtHW
    V_J,V_H,X,S_J,S_H, _ = compute_gsvd(JW,WtHW) 
    
    return V_J,V_H,X,S_J,S_H

def calc_precond_residuals(ws, mat, H, g ):
    "For a list of tensors ws, calculate the results \| mat@H@w - g \| "
    device = g.device
    return [torch.norm(mat @ (H(w.to(device))-g)).cpu() for w in ws]


def make_precond_mat(hessian_maker, J_op, H, W=None, rec_dim=30 ):
    device = hessian_maker.device
    if W is not None:
        # Projected preconditioner
        V_J,V_H,X,S_J,S_H = projected_JH_gsvd(hessian_maker, J_op, H, W.to(device))
        number_of_recycle_vectors = min(rec_dim, W.shape[-1])
        ResPrecond =  V_J[:,-number_of_recycle_vectors:] @(S_J @ torch.diag(1/torch.diag(S_H)))[-number_of_recycle_vectors:,-number_of_recycle_vectors:] @ V_H[:,-number_of_recycle_vectors:].T @ W.to(device).T
    else:
        # True preconditioner
        V_J,V_H,Xt,S_J,S_H = full_JH_gsvd(hessian_maker, J_op, H)
        ResPrecond = V_J @ (S_J @ torch.diag(1/torch.diag(S_H))) @ V_H.T 
    
    return ResPrecond

def hg_err_approx(hessian_maker, hess_ind, all_W_spaces, all_info):
    """ 
    hess_ind : Int : Which hessian system to consider
    all_W_spaces : list of tensors : Each tensor is the space used in the projected GSVD
    all_info : list of dictionaries : List of all intermediate RMINRES iterates for each hessian system solve
    """
    # Assemble the relevant linear system
    J_op, H, g = hessian_maker(hess_ind+1)
    
    # True preconditioner matrix - not possible in practice
    JHinv =  make_precond_mat(hessian_maker, J_op, H)
    
    # Preconditioner matrix when using solution space of previous system - what is done in practice
    ResPrecond = make_precond_mat(hessian_maker, J_op, H, all_W_spaces[hess_ind])
    
    # Preconditioner matrix when using solution space of current system - not possible in practice
    ResPrecondCheat = make_precond_mat(hessian_maker, J_op, H, all_W_spaces[hess_ind+1])
    
    # Compute the preconditioned residuals
    JHinv_r_gritz_true = calc_precond_residuals(all_info[hess_ind+1]['xs'], JHinv, H, g)
    JHinv_approx_r_gritz2 = calc_precond_residuals(all_info[hess_ind+1]['xs'], ResPrecond, H, g)
    JHinv_approx_r_gritz_cheat = calc_precond_residuals(all_info[hess_ind+1]['xs'], ResPrecondCheat, H, g)
    
    return JHinv_r_gritz_true, JHinv_approx_r_gritz2, JHinv_approx_r_gritz_cheat
    
