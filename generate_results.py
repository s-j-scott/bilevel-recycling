# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 07:42:02 2024

@author: Sebastian J. Scott

Re-solve sequence of Hessian systems using various recycling strategies.

"""

import torch
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np

import time

from bilevel_torch import   determine_recycle_space, create_W_space, compute_gsvd, solve_lower_level_singleton
from utils import set_device, create_title_str, display_filters,  plot_timings, plot_info, plot_data
from create_cost_functions import create_forward_op, create_cost_functions, create_dataset
from optimisers import MINRES, bfgs, cg, backtracking


torch.no_grad()


def clear_tqdm():
    while len(tqdm._instances) > 0:
        tqdm._instances.pop().close()


#%% Load data set

from utils import load_info
LOAD_DIR = './data/inpainting/'
SAVE_DIR = './results/'

def number_of_params(problem_setup):
    return problem_setup['regulariser']['filter_num'] * (problem_setup['regulariser']['filter_shape']**2+1 )

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)
device = set_device(verbose=True)

setup, all_params, xexacts_np, ys_np, all_recons_np, df_bl_info = load_info(LOAD_DIR)
all_hess_solns = torch.load(LOAD_DIR+'/OUTPUT_hess_solns.pt').to(device)
xexacts, ys, all_recons  = torch.tensor(xexacts_np,device=device), torch.tensor(ys_np,device=device), torch.tensor(all_recons_np,device=device)


n = setup['problem_setup']['n']
p = number_of_params(setup['problem_setup'])


# Plotting parameters
fs = 20 # Font size
lw = 2.5# Line width



title =  create_title_str(setup,full_title=False)
bilevel_its = len(all_params) # Number of iterations to terminate bilevel problem

A = create_forward_op(setup['problem_setup']['forward_op'],sigma=setup['problem_setup']['sigma'], device=device, tensor_size=torch.Size([n,n]),mask=setup['problem_setup']['mask'])
xexacts, ys =  create_dataset(n=n, ground_truth=setup['problem_setup']['ground_truth'], A=A, data_num=setup['problem_setup']['data_num'],
                                      noiselevel=setup['problem_setup']['noiselevel'],device=device)



ul_fun, ll_fun = create_cost_functions(A,xexacts, optns=setup['problem_setup']['regulariser'],device=device)


#%% Visualise results and save said visualisations

from thesis_recycle_utils import view_tensor_image

img_ind = 0
plot_info(df_bl_info,all_params, title=title, save=LOAD_DIR+'/OUTPUT_plot_info')
display_filters(xexacts_np[img_ind], ys_np[img_ind], all_params, all_recons_np[-1][img_ind], setup, A, title=title, save=LOAD_DIR+'/OUTPUT_plot_filters_')
plot_timings(df_bl_info, title=title, save=LOAD_DIR+'/OUTPUT_plot_timings')

def display_recon(ind, clip=False, img_ind=0):
    img = all_recons[ind][img_ind]
    if not torch.is_tensor(img): img = torch.tensor(img)
    if clip:
        view_tensor_image(torch.clip(img,0,1), len(img), title='Reconstruction for UL it '+str(ind))
    else:
        view_tensor_image(img, len(img), title='Reconstruction for UL it '+str(ind))


view_tensor_image(xexacts[0],n, title='Ground truth', fontsize=fs)
if setup['problem_setup']['forward_op'] == 'inpainting': 
    mask_np = A.mask.cpu().numpy()
    y_pad = np.empty(xexacts[0].shape) * np.nan
    y_pad[mask_np] = ys[0][0].cpu()
    view_tensor_image(torch.tensor(y_pad),n, title='Measurement', fontsize=fs)
else:
    view_tensor_image(ys[0].cpu(),n, title='Measurement', fontsize=fs)


view_tensor_image(all_recons[-1][0],n, title='FoE Reconstruction', fontsize=fs)


#%% Best TV reconstruction - grid search

# compute_tv_recon = False
compute_tv_recon = True

def update_best(recon_best, cost_best, recon_new, cost_new):
    if cost_best > cost_new:
        return recon_new, cost_new
    else:
        return recon_best, cost_best

    
def calculate_best_TV(ps, gamma=0.01):
    # Set cost functions
    _, ll_fun_tv = create_cost_functions(A,xexacts, optns={'name':'HuberTV', 'gamma':gamma, 'L':'FiniteDifference2D'},device=device)
    ll_fun_tv.y = ys[0]
    x0 = torch.zeros_like(xexacts[0])
    ul_costs_tv = np.empty(len(ps))
    cost_best_tv = ul_fun(x0)
    recon_best_tv = x0

    for ind in tqdm(range(len(ps)), desc='TV reconstruction'):
        p = ps[ind]
        ll_fun_tv.params = torch.tensor(p,device=device)
        xrecon_tv, info_tv = bfgs(ll_fun_tv, x0)
        x0 = xrecon_tv
        
        ul_costs_tv[ind] = ul_fun(xrecon_tv)
        
        recon_best_tv, cost_best_tv = update_best(recon_best_tv, cost_best_tv, xrecon_tv, ul_costs_tv[ind])
    return recon_best_tv, cost_best_tv, ul_costs_tv

if compute_tv_recon:
    ps = np.logspace(-16,1,20)
    
    recon_best_tv, cost_best_tv, ul_costs_tv = calculate_best_TV(ps)
    plt.figure()
    plt.plot(np.log10(ps),ul_costs_tv)
    plt.title('Grid search for TV reg param')
    
    view_tensor_image(recon_best_tv,n, title='TV Reconstruction', fontsize=fs)
    

#%% Useful functions

# Functions for creating cost functions and hypergradients
def make_hessian_system(p, n, ll_fun, recon=None):
    ll_fun.params = p # Avoid different Hessians pointing to same memory
    ll_fun.y = ys[0]
    if recon is None:
        recon, _ = bfgs(ll_fun, x0=torch.zeros_like(xexacts[0], device=device), max_its=setup['solver_optns']['ll_solve']['max_its'], grad_tol=setup['solver_optns']['ll_solve']['tol'])
    hess_op = lambda w : ll_fun.hess(recon , w.view(n,n)).ravel()
    g = (recon - xexacts[0]).ravel()
    return hess_op, g, recon

def create_lower_level(params=None):
    _, ll_fun = create_cost_functions(A,xexacts, optns=setup['problem_setup']['regulariser'],device=device)
    if p is not None:
        ll_fun.params = params
    return ll_fun

class hyp_grad():
    def __init__(self, x, ll_fun):
        self.x = x
        self.ll_fun = ll_fun
    def __call__(self, w):
        return self.ll_fun.grad_jac_adjoint(self.x, w.view(self.x.shape)).ravel()


# Functions for solving sequence of Hessian problems for different recycling strategies

class stop_criterion:
    def __init__(self, rec_dim, mode='res', ResPrecond=None, tol=1e-3, true_hg = None, hg_op=None):
        self.rec_dim = rec_dim
        self.ResPrecond = ResPrecond
        self.tol = tol
        self.mode = mode
        
        self.true_hg = true_hg # True hypergradients of current systems
        self.hg_op = hg_op

    def __call__(self, r, w=None):
        if self.mode =='res':
            return r.norm() < self.tol
        elif self.mode == 'hg_err_approx':
            if self.ResPrecond is not None:
                return torch.norm(self.ResPrecond @ r) < self.tol   
            else: 
                return torch.norm(r) < self.tol
            
        elif self.mode == 'hg_err':
            hg_err = torch.norm(self.hg_op(w) - self.true_hg)
            return hg_err < self.tol
        
        elif self.mode == 'hg_rerr':
            hg_err = torch.norm(self.hg_op(w) - self.true_hg) / torch.norm(self.true_hg)
            return hg_err < self.tol
        else: 
            print('\nI dont know that one')
    

def solve_hess(H,g,Hold,Vold,Uold,wold,recycle_dim,strategy='ritz',size='small', solver='MINRES', outer_rec=True, hg_op=None, pdim=None, tol=1e-3, matlab_eng=None, full_info=False, stop_crit='res', true_hg=None):
    # Determine recycle space
    
    if solver == 'MINRES':
        W = create_W_space(Vold,Uold) # Make W orthogonal
        Vold['basis_vecs'] = None # Make memory actually survive
        t0 = time.time()
        if outer_rec: # Outer recycling
            Huse = H
        else: # Inner recycling
            Huse = Hold
        C, U, ResPrecond, t_make_dense, t_decomp, t_save = determine_recycle_space(Huse, W, recycle_dim = recycle_dim, strategy=strategy, size=size, J_op=hg_op, pdim=pdim)
        
        stop_crit_fun = stop_criterion(recycle_dim, mode=stop_crit, ResPrecond=ResPrecond, tol=tol, true_hg=true_hg, hg_op=hg_op)
        
        t1 = time.time()  
        w, info = MINRES(H, g, C=C, U=U, x0=wold, tol=tol, max_its=setup['solver_optns']['hess_sys']['max_its'],  verbose=False, full_info = full_info, store_basis_vecs=True, stop_crit=stop_crit_fun)
    
        info['t_find_rec_space'] = t1-t0
        info['t_make_dense'] = t_make_dense
        info['t_decomp'] = t_decomp
        info['t_save'] = t_save #  We expect t_find_rec_space = t_decomp + t_save
    else:
        w, info = cg(H, g, x0=wold, tol=tol, max_its=setup['solver_optns']['hess_sys']['max_its'],  verbose=False, full_info = full_info)
        info['t_find_rec_space'] = 0
        info['t_find_rec_space'] = 0
        info['t_decomp'] = 0
        info['t_make_dense'] = 0
        W,U = None,None
    return w, info, W, U


class ul_fun_eval():
    def __init__(self,ul_fun, ll_fun, x):
        self.ul_fun = ul_fun
        self.ll_fun = ll_fun
        self.x = x
    def __call__(self,p):
        xnew, _ = solve_lower_level_singleton(p,ys[0],self.ll_fun,x0=self.x, solver='L-BFGS', num_store=10, verbose=False)
        
        self.x = xnew
        return self.ul_fun(xnew)        

def calc_ul_cost(x0, p, grad_p, ll_fun, ul_fun, ind=0):
    """
    See what the upper level cost would be if the hypergradient associated with the
    recycling method was actually employed to determine the update (rather than the non-recycling
    hypergradient which is actually employed)
    """
    eval_ul_fun_tmp = ul_fun_eval(ul_fun, ll_fun, x0)
    _,_,_,_,_,ul_cost,_ = backtracking(eval_ul_fun_tmp, grad_p, p, grad_p, step_size=df_bl_info['ul_steplengths'][ind], fun_eval=None)
    return ul_cost

def calc_all_ul_cost(params, all_recons, all_ws):
       
    ll_fun = create_lower_level() # Calling make_hessian_system will update the parameters of the regulariser
    all_ul_costs = np.empty(len(params)-1)
    
    for ind in tqdm(range(len(all_ul_costs)), desc='Calc UL cost'):
        p = torch.tensor(params[ind] , device=device)
        H, g, x = make_hessian_system(p, n=n, ll_fun=ll_fun, recon=all_recons[ind][0])
        hyp_grad_op = hyp_grad(x, ll_fun)
        all_ul_costs[ind] = calc_ul_cost(x, p, hyp_grad_op(all_ws[ind]).view(p.shape), ll_fun, ul_fun, ind=ind).cpu().numpy()
    return all_ul_costs


def solve_hess_seqn(params, all_recons, recycle_dim, strategy='ritz',size='small',
                    warm_start=True, tol=1e-3, matlab_eng=None, full_info=False,
                    stop_crit='res', true_hgs = None, solver='MINRES', outer_rec=True):
    """Returns:
        - List of solutions w
        - List of association Hessian system numerical solve info
        - List of dimension of Ws
        - List of determined Ws
        - List of determined Us
        """
    t0 = time.time()
    num_systems = len(params)-1 # Do not solve Hessian associated with final parameter
    
    # all_ul_costs = np.empty(num_systems)
    if true_hgs is None: true_hgs = [None for _ in range(num_systems)]
    all_ws = []
    all_infos = []
    all_W_dims = []
    if full_info:
        all_U_spaces = []
        all_W_spaces = []
    
    ll_fun = create_lower_level() # Calling make_hessian_system will update the parameters of the regulariser
    
    p_bar = tqdm(range(num_systems))
    # Solve first Hessian without recycling
    p = torch.tensor(params[0] , device=device)
    pdim = p.nelement()
    wold = torch.zeros(n**2, device=device)
    H, g, x = make_hessian_system(p, n=n, ll_fun=ll_fun, recon=all_recons[0][0])
    hyp_grad_op = hyp_grad(x, ll_fun)
        
    stop_crit_fun = stop_criterion(recycle_dim, mode=stop_crit, ResPrecond=None, tol=tol, hg_op=hyp_grad_op, true_hg=true_hgs[0])
    
    if solver == 'MINRES':
        w, info = MINRES(H, g, x0=wold, tol=tol, max_its=setup['solver_optns']['hess_sys']['max_its'],  verbose=False, full_info = full_info, store_basis_vecs=True, stop_crit=stop_crit_fun)
    else:
        w, info = cg(H, g, x0=wold, tol=tol, max_its=setup['solver_optns']['hess_sys']['max_its'],  verbose=False, full_info = full_info)
    
    all_ws.append(w.clone())
    if strategy is not None:
        info['t_find_rec_space'] = 0
        info['t_make_dense'] = 0
        info['t_decomp'] = 0
        info['t_save'] = 0
        info['t_grand_total']= time.time() -  t0
    all_infos.append(info)
    U = None
    p_bar.update(1)
    
    # Employ recycling for subsequent Hessian systems
    for ind in (range(num_systems-1)):
        t0 = time.time()
        # Add 1 to index because we have already solved the first Hessian system
        Hold, _, _ = make_hessian_system(p, n=n, ll_fun=create_lower_level(), recon=all_recons[ind][0])
        # Hold = H
        p = torch.tensor(params[ind+1] , device=device)
        H, g, x = make_hessian_system(p, n=n, ll_fun=ll_fun, recon=all_recons[ind+1][0])
        hyp_grad_op = hyp_grad(x, ll_fun)
        if warm_start:
            w_start = w
        else:
            w_start = wold
        w, info, W, U =  solve_hess(H,g,Hold,info,U,w_start,recycle_dim,strategy=strategy,size=size, solver=solver, hg_op=hyp_grad_op, pdim=pdim, tol=tol,full_info = full_info, stop_crit=stop_crit, true_hg=true_hgs[ind+1], outer_rec=outer_rec)
        
        info['t_grand_total']= time.time() -  t0
        all_ws.append(w.clone())
        all_infos.append(info)
        if solver == 'MINRES': all_W_dims.append(W.shape[-1])
        if full_info:
            all_U_spaces.append(U)
            all_W_spaces.append(W)
        p_bar.update(1)
    
    if full_info:
        return all_ws, all_infos, np.array(all_W_dims), all_W_spaces, all_U_spaces
    else:
        return all_ws, all_infos, np.array(all_W_dims)

def calc_hypgrad_errs(ws, high_ws):   
    # Relative error in hypergrad
    ll_fun_tmp = create_lower_level()
    hg_errs = torch.empty(len(ws))
    
    for ind in (range(len(ws))):
        p = torch.tensor(all_params[ind] , device=device)
        ll_fun_tmp.params = p
        hypgrad_op = hyp_grad(all_recons[ind][0], ll_fun_tmp)
        
        high_hg = hypgrad_op(high_ws[ind]) # High accuracy hypergradient
        hg_errs[ind] = (torch.norm(high_hg - hypgrad_op(ws[ind])) / high_hg.norm()).cpu()
    return hg_errs

def calc_hgs(ws): 
        "Take list of ws and return list of hypergradients"
        ll_fun_tmp = create_lower_level()
        out = []
        
        for ind in (range(len(ws))):
            p = torch.tensor(all_params[ind] , device=device)
            ll_fun_tmp.params = p
            hypgrad_op = hyp_grad(all_recons[ind][0], ll_fun_tmp)
            out.append ( hypgrad_op(ws[ind]) ) # High accuracy hypergradient
        return out


def calc_RMINRES_cost(its, hess_cost, n, s):
    return np.array(hess_cost + 4*n + 6*n*s  + its*(hess_cost + 21*n + 6*n*s - s), dtype='int64')
        # return  6*n*s  + its*(hess_cost + 21*n + 6*n*s - s)
def calc_MINRES_cost(its, hess_cost, n):
    return np.array(hess_cost + 4*n + its*(hess_cost + 16*n), dtype='int64')

def calc_hess_cost(num_filt, kernel_size, n, forward_op_cost=0):
    ''' Cost of application of the Hessian for denoising Field of Experts'''
    return n + num_filt * 2*n * (3*kernel_size**2 -1) + 2*forward_op_cost

    
def compute_everything(strategy=None,size=None,rec_dim=1, tol=setup['solver_optns']['hess_sys']['tol'],matlab_eng=None,
                       warm_start=True, full_info=False, stop_crit='res', hess_cost=0, outer_rec=True, true_hgs=None, solver='MINRES'):
    """Solve entire sequence of Hessian systems for a specified recycle strategy and return all useful information associated with the solves"""
    if full_info:
        all_w, all_info, all_W_dims, all_W_spaces, all_U_spaces = solve_hess_seqn(all_params, all_recons, rec_dim, outer_rec=outer_rec, size=size, strategy=strategy, tol=tol, solver=solver, warm_start=warm_start, full_info=full_info, stop_crit=stop_crit, true_hgs=true_hgs)
    else:
        all_w, all_info, all_W_dims = solve_hess_seqn(all_params, all_recons, rec_dim, size=size, strategy=strategy, outer_rec=outer_rec, tol=tol, warm_start=warm_start, solver=solver, full_info=full_info, stop_crit=stop_crit, true_hgs=true_hgs)
    stop_it = np.array([inf['stop_it'] for inf in all_info]) 
    cost = calc_RMINRES_cost(stop_it, hess_cost , n, rec_dim)
    hg_err = calc_hypgrad_errs(all_w, all_w_high )
    if full_info:
        return all_w, all_info, all_W_dims, stop_it, cost, hg_err, all_W_spaces, all_U_spaces
    else:
        return all_w, all_info, all_W_dims, stop_it, cost, hg_err
    
    


    
def extract_times(hess_infos, item_name='t_total'):
    if isinstance(hess_infos[0], dict):
        return np.array([inf[item_name] for inf in hess_infos])
    else:
        return np.array([inf[0][item_name] for inf in hess_infos])
    
def construct_dense_matrix(op,n,m=None, verbose=False, desc=''):
    if m is None: m=n
    mat = torch.empty(n,m,device=device)
    for ind in tqdm(range(m),disable = not(verbose), desc='Dense matrix '+desc+':'):
        e = torch.zeros(m,device=device)
        e[ind] = 1.
        mat[:,ind] = op(e)
    return mat

    

def compare_performance(stop_its_none,  ul_cost_none, cost_none, hg_err_none, list_to_plot, title=None, label='None', xlabel='Linear system number', fs=22, lw=3):
    fig, ax = plt.subplot_mosaic([['it_plot','cum_stop_it','hg_err', 'ul_cost']], figsize=(20,5))
    ax['it_plot'].set_ylabel('Number of its', fontsize=fs)
    ax['it_plot'].plot(stop_its_none ,linestyle=(0,(1,1)), color='k', label=label, linewidth=lw, marker='*', markevery=0.15, markersize=10)   
    ax['cum_stop_it'].plot(np.cumsum(stop_its_none),'k',linestyle=(0,(1,1)), label=label, linewidth=lw, marker='*', markevery=0.15, markersize=10) 
    ax['cum_stop_it'].set_ylabel('Cumulative number of its', fontsize=fs)
    ax['ul_cost'].set_ylabel('Upper level cost', fontsize=fs)
    ax['ul_cost'].plot((ul_cost_none) ,linestyle=(0,(1,1)), color='k', label=label, linewidth=lw, marker='*', markevery=0.15, markersize=10)
    
    ax['hg_err'].set_ylabel('log10 HG relative error', fontsize=fs)
    ax['hg_err'].plot(torch.log10(hg_err_none) ,linestyle=(0,(1,1)), color='k', label=label, linewidth=lw, marker='*', markevery=0.15, markersize=10)
    for items in list_to_plot:
        if len(items) == 5: 
            label, stop_it, ul_cost, cost, hg_errs = items
            linestyle, marker, color = '-', None, None
        if len(items) == 7: 
            label, stop_it, ul_cost, cost, hg_errs, linestyle, color, = items
            marker, markevery = None, 0.1        
        else:
            label, stop_it,ul_cost, cost, hg_errs, linestyle, color, marker, markevery = items
        if color is None: 
            ax['it_plot'].plot(stop_it , label=label, linewidth=lw, linestyle=linestyle, marker=marker, markevery=markevery)
            ax['cum_stop_it'].plot(np.cumsum(stop_it), label=label, linewidth=lw, linestyle=linestyle, marker=marker, markevery=markevery)
            ax['hg_err'].plot(torch.log10(hg_errs) , label=label, linewidth=lw, linestyle=linestyle, marker=marker, markevery=markevery)
            ax['ul_cost'].plot(ul_cost , label=label, linewidth=lw, linestyle=linestyle, marker=marker, markevery=markevery)
        else:
            ax['it_plot'].plot(stop_it , label=label, linewidth=lw, linestyle=linestyle, marker=marker, markevery=markevery, color=color)
            ax['cum_stop_it'].plot(np.cumsum(stop_it), label=label, linewidth=lw, linestyle=linestyle, marker=marker, markevery=markevery, color=color)
            ax['hg_err'].plot(torch.log10(hg_errs) , label=label, linewidth=lw, linestyle=linestyle, marker=marker, markevery=markevery, color=color)
            ax['ul_cost'].plot((ul_cost) , label=label, linewidth=lw, linestyle=linestyle, marker=marker, markevery=markevery, color=color)

    for a in ax:
        ax[a].set_xlabel(xlabel,fontsize=fs)
        ax[a].legend(prop={'size': fs*.7})
        plt.setp(ax[a].get_xticklabels(), fontsize=fs*.8)
        plt.setp(ax[a].get_yticklabels(), fontsize=fs*.8)
    plt.suptitle(title, fontsize=fs)
    plt.tight_layout()
    plt.show()


def plot_ratios(t_total, ts, title=None, fontsize=18, linewidth=2.5, xlabel='Dimension of recycle space', xvals=None):
    labels=['Dense', 'Decomp', 'Save', 'RMINRES']
    colors=['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    markers=[None, 'o', None, 'o']
    styles=['solid', 'solid', ':', ':']
    fig,ax = plt.subplot_mosaic([
        ['overall_t'],
        ['overall_ratio']],
        figsize=(6,5.5))
    
    alpha = 0.3 # opacity of filled blocks
    ts_rat = [t/t_total for t in ts]
    if xvals is None:
        xvals = np.linspace(0,len(t_total)-1, len(t_total))
    yold = np.zeros_like(xvals)
    for i in range(len(ts)):
        ax["overall_t"].plot(xvals,ts[i], color=colors[i], label=labels[i], linewidth=linewidth, markevery=0.2, marker=markers[i], linestyle=styles[i] )
        ynew = yold + ts_rat[i]
        ax["overall_ratio"].plot(xvals, ynew, color=colors[i], label=labels[i], linewidth=linewidth )
        ax["overall_ratio"].fill_between(x=xvals,y1=yold, y2=ynew, color=colors[i], alpha=alpha)
        yold = ynew
    ax["overall_ratio"].set_ylabel('Ratio of timings', fontsize=fontsize)
    ax["overall_ratio"].legend(fontsize=fontsize*.7)
    ax["overall_t"].set_ylabel('Time (s)', fontsize=fontsize)
    ax["overall_t"].legend( fontsize=fontsize*.7)
    for a in ax:
        ax[a].set_xlabel(xlabel, fontsize=fontsize)
    plt.tight_layout()
    
    

def give_me_times(all_infos):
    t_solve  = extract_times(all_infos)
    t_dense  = extract_times(all_infos, 't_make_dense')
    t_decomp = extract_times(all_infos, 't_decomp')
    t_save   = extract_times(all_infos, 't_save')
    
    t_current = t_solve + t_dense + t_decomp + t_save
    t_optim = t_solve + t_decomp
    return t_current, t_optim, [ t_dense, t_decomp, t_save, t_solve]

#%% Determine cost of a Hessian evaluation

if setup['problem_setup']['forward_op'] == 'gaussian_blur': 
    forward_op_cost = n * np.prod(A.filter.shape[0]**2)
else:
    forward_op_cost = 0 
hess_cost = calc_hess_cost(setup['problem_setup']['regulariser']['filter_num'] , setup['problem_setup']['regulariser']['filter_shape'], n, forward_op_cost=forward_op_cost)

 #%%%%%%%%%%%%%%%%%% Compute high accuracy solutions 

rec_dim = 30 # Dimension of recycle space to be considered throughout

# High accuracy hypergradient
print('Solving with high accuracy and no recycling')
all_w_high, all_info_high, _ = solve_hess_seqn(all_params, all_recons, recycle_dim=0, strategy=None, tol=1e-13)
stop_it_high = np.array([inf['stop_it'] for inf in all_info_high]) 
all_hgs_high = calc_hgs(all_w_high)
all_hgs_high_norms = np.array([torch.norm(a).cpu() for a in all_hgs_high])
    
#%%%% Normal accuracy solutions without recycling
print('Solving with normal accuracy and no recycling')
all_w_none, all_info_none, all_W_dims_none, stop_it_none, _, all_hg_err_none, _, _ = compute_everything(None,None,1, full_info=True)
cost_none = calc_MINRES_cost(stop_it_none, hess_cost , n) # Because we used MINRES not RMINRES
ul_cost_none = df_bl_info['ul_evals'][1:].values

none_results = {'stop_it_none': stop_it_none, 'ul_cost_none':ul_cost_none, 'cost_none':cost_none, 'all_hg_err_none':all_hg_err_none}
torch.save(none_results, SAVE_DIR+'none_results')

#%% Justification for recycling
"Verify that the linear systems are similar and so utilising recycling is a reasonable idea to make."

    
def experiment_justify_recycling():

    dif_recon = torch.empty(bilevel_its-1)
    norm_recon = torch.empty(bilevel_its-1)
    dif_hess = torch.empty(bilevel_its-1)
    norm_hess = torch.empty(bilevel_its-1)
    dif_rhs = torch.empty(bilevel_its-1)
    norm_rhs = torch.empty(bilevel_its-1)
    dif_w = torch.empty(bilevel_its-1)
    norm_w = torch.empty(bilevel_its-1)
    hess_cond = torch.empty(bilevel_its-1) # Condition number of Hessian
    Jevecs_norm = torch.empty(n**2, bilevel_its-1) # Matrix with columns the norms of Jacobian applied to eigenvector
    w_evec_coefs = torch.empty(n**2, bilevel_its-1) # Coefficents of w in eigendecomp representation
    H_evals = torch.empty(n**2,bilevel_its-1) # Matrix with columns the eigenvalues of Hessian
    
    
    ll_fun_old = create_lower_level()
    p_old = torch.tensor(all_params[0] , device=device)
    Hold, gold, xold = make_hessian_system(p_old, n=n, ll_fun=ll_fun_old, recon=all_recons[0][0])
    H_mat = construct_dense_matrix(Hold, n**2)
    wold = all_hess_solns[0]
    
    for i in (tqdm(range(bilevel_its-1), desc='Comparing Hess systems')):
        ll_fun_new = create_lower_level()
        p_new = torch.tensor(all_params[i+1] , device=device)
        Hnew, gnew, xnew = make_hessian_system(p_new, n=n, ll_fun=ll_fun_new, recon=all_recons[i+1][0])
        Hnew_mat = construct_dense_matrix(Hnew, n**2)
        wnew = all_hess_solns[i+1]
        
        norm_recon[i] = xold.norm().cpu()
        norm_hess[i] = H_mat.norm().cpu()
        norm_rhs[i] = gold.norm().cpu()
        norm_w[i] = wold.norm().cpu()
        dif_recon[i] = torch.norm(xnew-xold).norm().cpu()
        dif_hess[i] = torch.norm(H_mat-Hnew_mat).norm().cpu()
        dif_rhs[i] = torch.norm(gold-gnew).norm().cpu()
        hess_cond[i] = torch.linalg.cond(H_mat)
        dif_w[i] = (wnew-wold).norm()
        
        # Explore eigen decomp of the old system
        hyp_grad_op = hyp_grad(xold, ll_fun_old)
        S_H, U_H = torch.linalg.eigh(H_mat) # Eigendecomposition
        H_evals[:,i] = S_H.clone()
        Jevecs = torch.stack([hyp_grad_op(U_H[:,i]).ravel() for i in range(len(U_H)) ], dim=1)
        Jevecs.shape
        Jevecs_norm[:,i] = torch.norm(Jevecs, dim=0)
        
        # Coefficients of w representation using eigenvectors
        w_evec_coefs[:,i] =  torch.diag(1/S_H) @ U_H.T @ gold
        
        # Prepare for next iteration
        H_mat = Hnew_mat.clone()
        gold = gnew.clone()
        xold = xnew.clone()
        wold = wnew.clone()
        ll_fun_old = ll_fun_new
    

    plot_data(torch.log10(dif_hess/norm_hess),ylabel= 'log10 $H^{(i)}$ rel. difference', xlabel='Linear system number')    
    plot_data(torch.log10(dif_rhs/norm_rhs),ylabel='log10 $g^{(i)}$ rel. difference', xlabel='Linear system number')    
    plot_data(torch.log10(dif_w/norm_w), ylabel='log10 $w^{(i)}$ rel. difference', xlabel='Linear system number')    
    plot_data(torch.log10(dif_recon/norm_recon), ylabel=r'log10 $x(\theta^{i+1})$ rel. dif.', xlabel='Linear system number')      
    plot_data(hess_cond, ylabel='Hessian condition number', xlabel='Linear system number')    


    rel_difs =  {
      'dif_hess': dif_hess,  'norm_hess': norm_hess,
      'dif_rhs': dif_rhs, 'norm_rhs':norm_rhs,
       'dif_recon': dif_recon, 'norm_recon':norm_recon,
       'dif_w':dif_w, 'norm_w':norm_w,
       'hess_cond':hess_cond
       }
    
    torch.save(rel_difs, SAVE_DIR+'rel_difs')
    
    

experiment_justify_recycling()    

#%% Comparison between Ritz (generalised) vectors and (eigen/singular)vectors}
"How does recycling utilising Ritz vectors compare to recycling using the actual eigenvectors we are wanting to approximate?"

print('Eigenvectors vs Ritz')
all_w_eig_s, all_info_eig_s, all_W_dims_eig_s, stop_it_eig_s, cost_eig_s, all_hg_err_eig_s, all_W_spaces_eig_s, all_U_spaces_eig_s = compute_everything('eig','small',rec_dim, full_info=True)
all_ul_cost_eig_s = calc_all_ul_cost(all_params, all_recons, all_w_eig_s)

all_w_ritz_s, all_info_ritz_s, all_W_dims_ritz_s, stop_it_ritz_s, cost_ritz_s, all_hg_err_ritz_s, all_W_spaces_ritz_s, all_U_spaces_ritz_s= compute_everything('ritz','small',rec_dim, full_info=True)
all_ul_cost_ritz_s = calc_all_ul_cost(all_params, all_recons, all_w_ritz_s)
print('GSVD vs GRitz')
all_w_gsvd_rl, all_info_gsvd_rl, all_W_dims_gsvd_rl, stop_it_gsvd_rl, cost_gsvd_rl, all_hg_err_gsvd_rl, all_W_spaces_gsvd_rl, all_U_spaces_gsvd_rl = compute_everything('gsvd_right','large',rec_dim, full_info=True,stop_crit='res')
all_ul_cost_gsvd_rl = calc_all_ul_cost(all_params, all_recons, all_w_gsvd_rl)
all_w_gritz_rl, all_info_gritz_rl, all_W_dims_gritz_rl, stop_it_gritz_rl, cost_gritz_rl, all_hg_err_gritz_rl, all_W_spaces_gritz_rl, all_U_spaces_gritz_rl = compute_everything('gritz_right','large',rec_dim, full_info=True,stop_crit='res')
all_ul_cost_gritz_rl = calc_all_ul_cost(all_params, all_recons, all_w_gritz_rl)


#%%% Save results

eig_v_ritz_plot = [['Eig-S', stop_it_eig_s, all_ul_cost_eig_s, cost_eig_s, all_hg_err_eig_s, ':', 'tab:cyan']
              ,['Ritz-S', stop_it_ritz_s, all_ul_cost_ritz_s, cost_ritz_s, all_hg_err_ritz_s, 'dotted', 'tab:blue','o',.4]
              ,['GSVD-L(R)', stop_it_gsvd_rl, all_ul_cost_gsvd_rl, cost_gsvd_rl, all_hg_err_gsvd_rl, 'solid', 'tab:olive', 'o',.3]
              ,['RGen-L(R)', stop_it_gritz_rl, all_ul_cost_gritz_rl, cost_gritz_rl, all_hg_err_gritz_rl, 'solid', 'tab:green','o', .3]]


torch.save(eig_v_ritz_plot, SAVE_DIR+'eig_v_ritz')

compare_performance(stop_it_none, ul_cost_none, cost_none, all_hg_err_none,
                eig_v_ritz_plot)

#%% Plot timings
t_none  = extract_times(all_info_none)
t_current_gritz_rl, t_optim_gritz_rl, t_breakdown_gritz_rl = give_me_times(all_info_gritz_rl)
t_current_gsvd_rl, t_optim_gsvd_rl, t_breakdown_gsvd_rl = give_me_times(all_info_gsvd_rl)
t_current_ritz_s, t_optim_ritz_s, t_breakdown_ritz_s = give_me_times(all_info_ritz_s)
t_current_eig_s, t_optim_eig_s, t_breakdown_eig_s = give_me_times(all_info_eig_s)

# plot_ratios(t_current_eig_s, t_breakdown_eig_s, xlabel='Linear system number')
# plot_ratios(t_current_ritz_s, t_breakdown_ritz_s, xlabel='Linear system number')
# plot_ratios(t_current_gsvd_rl, t_breakdown_gsvd_rl, xlabel='Linear system number')
# plot_ratios(t_current_gritz_rl, t_breakdown_gritz_rl, xlabel='Linear system number')



#%% MINRES vs CG
"""CG exploits positive definiteness and symmetry and is the common choice for solving sequence of Hessian systems.
We employ MINRES which only exploits symmetry. What is the impact of performance? Compare with warm vs cold start"""


# No recycling MINRES
# all_w_none, all_info_none, all_W_dims_none, stop_it_none, _, all_hg_err_none, _, _ = compute_everything(None,None,1, full_info=True)
all_w_none2, all_info_none2, all_W_dims_none2, stop_it_none2, _, all_hg_err_none2, _, _ = compute_everything(None,None,1, full_info=True, warm_start=False)
# all_w_none3, all_info_none3, all_W_dims_none3, stop_it_none3, _, all_hg_err_none3, _, _ = compute_everything(None,None,1, full_info=True, warm_start=True)
all_ul_cost_none2 = calc_all_ul_cost(all_params, all_recons, all_w_none2)
# all_ul_cost_none3 = calc_all_ul_cost(all_params, all_recons, all_w_none3)

# No recycling CG
all_w_cg, all_info_cg, all_W_dims_cg,  stop_it_cg, _, all_hg_err_cg, _, _ = compute_everything(None,None,1, full_info=True, solver='cg')
all_w_cg2, all_info_cg2, all_W_dims_cg2,  stop_it_cg2, _, all_hg_err_cg2, _, _= compute_everything(None,None,1, full_info=True, solver='cg', warm_start=False)
all_ul_cost_cg = calc_all_ul_cost(all_params, all_recons, all_w_cg)
all_ul_cost_cg2  = calc_all_ul_cost(all_params, all_recons, all_w_cg2)

#%%
# plt.figure()
# plt.plot(all_info_none[0]['r_norms'].cpu(), label='MINRES')
# plt.plot(all_info_cg[0]['r_norms'].cpu(), label='CG')
# plt.yscale('log')
# plt.legend()

#%%% Save results
cold_v_warm_plot =  [
  ['MINRES cold', stop_it_none2, all_ul_cost_none2, None, all_hg_err_none2, 'solid', 'k', 'o', .1],
  ['CG', stop_it_cg, all_ul_cost_cg, None, all_hg_err_cg, 'dashed', 'darkorange'],
  ['CG cold', stop_it_cg2, all_ul_cost_cg2, None, all_hg_err_cg2, 'solid', 'darkorange']]

torch.save(cold_v_warm_plot, SAVE_DIR+'cold_v_warm')

compare_performance(stop_it_none, ul_cost_none, cost_none, all_hg_err_none,   
               cold_v_warm_plot, label='MINRES')




#%% Information associated with small vs large (generalised singular/eigen) values
"Small or large Ritz value information better suited for this application? Same for  small or large Ritz"
" generalised value information?"

all_w_gritz_rl2, all_info_gritz_rl2, all_W_dims_gritz_rl2, stop_it_gritz_rl2, cost_gritz_rl2, all_hg_err_gritz_rl2  = compute_everything('gritz_right','large',rec_dim, stop_crit='hg_err_approx')
all_ul_cost_gritz_rl2 = calc_all_ul_cost(all_params, all_recons, all_w_gritz_rl2)



#%%% Compute the rest of them

all_w_ritz_l, all_info_ritz_l, all_W_dims_ritz_l, stop_it_ritz_l, cost_ritz_l, all_hg_err_ritz_l = compute_everything('ritz','large',rec_dim)
all_ul_cost_ritz_l = calc_all_ul_cost(all_params, all_recons, all_w_ritz_l)

all_w_ritz_b, all_info_ritz_b, all_W_dims_ritz_b,  stop_it_ritz_b, cost_ritz_b, all_hg_err_ritz_b = compute_everything('ritz','mix',rec_dim)
all_ul_cost_ritz_b = calc_all_ul_cost(all_params, all_recons, all_w_ritz_b)


print('Calcualting performance for all Harmonic Ritz vectors')
all_w_hritz_s, all_info_hritz_s, all_W_dims_hritz_s, stop_it_hritz_s, cost_hritz_s, all_hg_err_hritz_s, all_W_spaces_hritz_s, all_U_spaces_hritz_s= compute_everything('harmonic_ritz','small',rec_dim, full_info=True)
all_ul_cost_hritz_s = calc_all_ul_cost(all_params, all_recons, all_w_hritz_s)

all_w_hritz_l, all_info_hritz_l, all_W_dims_hritz_l, stop_it_hritz_l, cost_hritz_l, all_hg_err_hritz_l, all_W_lpaces_hritz_l, all_U_lpaces_hritz_l= compute_everything('harmonic_ritz','large',rec_dim, full_info=True)
all_ul_cost_hritz_l = calc_all_ul_cost(all_params, all_recons, all_w_hritz_l)

all_w_hritz_m, all_info_hritz_m, all_W_dims_hritz_m, stop_it_hritz_m, cost_hritz_m, all_hg_err_hritz_m, all_W_mpaces_hritz_m, all_U_mpaces_hritz_m= compute_everything('harmonic_ritz','mix',rec_dim, full_info=True)
all_ul_cost_hritz_m = calc_all_ul_cost(all_params, all_recons, all_w_hritz_m)



print('Calculating performance for all Ritz Generalised vectors | Residual SC')
all_w_gritz_ll, all_info_gritz_ll, all_W_dims_gritz_ll, stop_it_gritz_ll, cost_gritz_ll, all_hg_err_gritz_ll  = compute_everything('gritz_left','large',rec_dim)
all_ul_cost_gritz_ll = calc_all_ul_cost(all_params, all_recons, all_w_gritz_ll)
all_w_gritz_ls, all_info_gritz_ls, all_W_dims_gritz_ls,  stop_it_gritz_ls, cost_gritz_ls, all_hg_err_gritz_ls = compute_everything('gritz_left','small',rec_dim)
all_ul_cost_gritz_ls = calc_all_ul_cost(all_params, all_recons, all_w_gritz_ls)
all_w_gritz_lb, all_info_gritz_lb, all_W_dims_gritz_lb,  stop_it_gritz_lb, cost_gritz_lb, all_hg_err_gritz_lb = compute_everything('gritz_left','mix',rec_dim)
all_ul_cost_gritz_lb = calc_all_ul_cost(all_params, all_recons, all_w_gritz_lb)

all_w_gritz_rs, all_info_gritz_rs, all_W_dims_gritz_rs, stop_it_gritz_rs, cost_gritz_rs, all_hg_err_gritz_rs = compute_everything('gritz_right','small',rec_dim)
all_ul_cost_gritz_rs = calc_all_ul_cost(all_params, all_recons, all_w_gritz_rs)
all_w_gritz_rb, all_info_gritz_rb, all_W_dims_gritz_rb, stop_it_gritz_rb, cost_gritz_rb, all_hg_err_gritz_rb = compute_everything('gritz_right','mix',rec_dim)
all_ul_cost_gritz_rb = calc_all_ul_cost(all_params, all_recons, all_w_gritz_rb)

all_w_gritz_bs, all_info_gritz_bs, all_W_dims_gritz_bs, stop_it_gritz_bs, cost_gritz_bs, all_hg_err_gritz_bs = compute_everything('gritz_both','small',rec_dim)
all_ul_cost_gritz_bs = calc_all_ul_cost(all_params, all_recons, all_w_gritz_bs)
all_w_gritz_bl, all_info_gritz_bl, all_W_dims_gritz_bl, stop_it_gritz_bl, cost_gritz_bl, all_hg_err_gritz_bl = compute_everything('gritz_both','large',rec_dim)
all_ul_cost_gritz_bl = calc_all_ul_cost(all_params, all_recons, all_w_gritz_bl)
all_w_gritz_bb, all_info_gritz_bb, all_W_dims_gritz_bb, stop_it_gritz_bb, cost_gritz_bb, all_hg_err_gritz_bb = compute_everything('gritz_both','mix',rec_dim)
all_ul_cost_gritz_bb = calc_all_ul_cost(all_params, all_recons, all_w_gritz_bb)

print('Calculating performance for all Ritz Generalised vectors |-NSC')
all_w_gritz_ll2, all_info_gritz_ll2, all_W_dims_gritz_ll2, stop_it_gritz_ll2, cost_gritz_ll2, all_hg_err_gritz_ll2  = compute_everything('gritz_left','large',rec_dim, stop_crit='hg_err_approx')
all_ul_cost_gritz_ll2 = calc_all_ul_cost(all_params, all_recons, all_w_gritz_ll2)
all_w_gritz_ls2, all_info_gritz_ls2, all_W_dims_gritz_ls2, stop_it_gritz_ls2, cost_gritz_ls2, all_hg_err_gritz_ls2 = compute_everything('gritz_left','small',rec_dim, stop_crit='hg_err_approx')
all_ul_cost_gritz_ls2 = calc_all_ul_cost(all_params, all_recons, all_w_gritz_ls2)
all_w_gritz_lb2, all_info_gritz_lb2, all_W_dims_gritz_lb2, stop_it_gritz_lb2, cost_gritz_lb2, all_hg_err_gritz_lb2 = compute_everything('gritz_left','mix',rec_dim, stop_crit='hg_err_approx')
all_ul_cost_gritz_lb2 = calc_all_ul_cost(all_params, all_recons, all_w_gritz_lb2)

all_w_gritz_rs2, all_info_gritz_rs2, all_W_dims_gritz_rs2, stop_it_gritz_rs2, cost_gritz_rs2, all_hg_err_gritz_rs2 = compute_everything('gritz_right','small',rec_dim, stop_crit='hg_err_approx')
all_ul_cost_gritz_rs2 = calc_all_ul_cost(all_params, all_recons, all_w_gritz_rs2)
all_w_gritz_rb2, all_info_gritz_rb, all_W_dims_gritz_rb2, stop_it_gritz_rb2, cost_gritz_rb2, all_hg_err_gritz_rb2 = compute_everything('gritz_right','mix',rec_dim, stop_crit='hg_err_approx')
all_ul_cost_gritz_rb2 = calc_all_ul_cost(all_params, all_recons, all_w_gritz_rb2)

all_w_gritz_bs2, all_info_gritz_bs2, all_W_dims_gritz_bs2, stop_it_gritz_bs2, cost_gritz_bs2, all_hg_err_gritz_bs2 = compute_everything('gritz_both','small',rec_dim, stop_crit='hg_err_approx')
all_ul_cost_gritz_bs2 = calc_all_ul_cost(all_params, all_recons, all_w_gritz_bs2)
all_w_gritz_bl2, all_info_gritz_bl2, all_W_dims_gritz_bl2, stop_it_gritz_bl2, cost_gritz_bl2, all_hg_err_gritz_bl2 = compute_everything('gritz_both','large',rec_dim, stop_crit='hg_err_approx')
all_ul_cost_gritz_bl2 = calc_all_ul_cost(all_params, all_recons, all_w_gritz_bl2)

all_w_gritz_bb2, all_info_gritz_bb2, all_W_dims_gritz_bb2, stop_it_gritz_bb2, cost_gritz_bb2, all_hg_err_gritz_bb2 = compute_everything('gritz_both','mix',rec_dim, stop_crit='hg_err_approx')
all_ul_cost_gritz_bb2 = calc_all_ul_cost(all_params, all_recons, all_w_gritz_bb2)



#%%% Plotting style information
# all_plot_details = [
#     ['Ritz-S',          stop_it_ritz_s,     all_ul_cost_ritz_s, cost_ritz_s, all_hg_err_ritz_s, 'dotted', 'tab:blue','o',.4]
#    ,['RGen-L(R)',       stop_it_gritz_rl,   all_ul_cost_gritz_rl,cost_gritz_rl, all_hg_err_gritz_rl, 'solid', 'tab:green']
#    ,['RGen-L(R)-NSC',stop_it_gritz_rl2,  all_ul_cost_gritz_rl2,cost_gritz_rl2, all_hg_err_gritz_rl2, 'solid', 'tab:olive', '*', .25]
#    ] 

all_plot_details = [
     ['Ritz-S',          stop_it_ritz_s,     all_ul_cost_ritz_s,cost_ritz_s,   all_hg_err_ritz_s, 'dotted', 'tab:blue','o',.4]
    ,['Ritz-L',          stop_it_ritz_l,     all_ul_cost_ritz_l,cost_ritz_l,   all_hg_err_ritz_l, 'solid', 'tab:blue']
    ,['Ritz-M',          stop_it_ritz_b,     all_ul_cost_ritz_b,cost_ritz_b,   all_hg_err_ritz_b, 'dashed', 'tab:blue','d',.4]
    ,['HRitz-S',         stop_it_hritz_s,    all_ul_cost_hritz_s, cost_hritz_s, all_hg_err_hritz_s, 'dotted', 'tab:purple','o',.3]
    ,['HRitz-L',         stop_it_hritz_l,    all_ul_cost_hritz_l, cost_hritz_l, all_hg_err_hritz_l, 'solid', 'tab:purple']    
    ,['HRitz-M',         stop_it_hritz_m,    all_ul_cost_hritz_m, cost_hritz_m, all_hg_err_hritz_m, 'dashed', 'tab:purple','d',.4]
    ,['RGen-S(L)',       stop_it_gritz_ls,   all_ul_cost_gritz_ls,cost_gritz_ls,  all_hg_err_gritz_ls, 'dotted', 'tab:orange','o',.4]
    ,['RGen-L(L)',       stop_it_gritz_ll,   all_ul_cost_gritz_ll,cost_gritz_ll,  all_hg_err_gritz_ll, 'solid', 'tab:orange']
    ,['RGen-L(L)-NSC',stop_it_gritz_ll2,  all_ul_cost_gritz_ll2,cost_gritz_ll2, all_hg_err_gritz_ll2, 'solid', 'tab:red','*',.4]
    ,['RGen-M(L)',       stop_it_gritz_lb,   all_ul_cost_gritz_lb,cost_gritz_lb,  all_hg_err_gritz_lb, 'dashed', 'tab:orange', 'd', .4]
    ,['RGen-M(L)-NSC',stop_it_gritz_lb2,  all_ul_cost_gritz_lb2,cost_gritz_lb2, all_hg_err_gritz_lb2, 'dashed', 'tab:red', '*', .2]
    ,['RGen-S(R)',       stop_it_gritz_rs,   all_ul_cost_gritz_rs,  cost_gritz_rs,  all_hg_err_gritz_rs, 'dotted', 'tab:green', 'o', .5]
    ,['RGen-S(R)-NSC',stop_it_gritz_rs2,  all_ul_cost_gritz_rs2, cost_gritz_rs2, all_hg_err_gritz_rs2, 'dotted', 'tab:olive', '*', .25]
    ,['RGen-L(R)',       stop_it_gritz_rl,   all_ul_cost_gritz_rl,  cost_gritz_rl,  all_hg_err_gritz_rl, 'solid', 'tab:green']
    ,['RGen-L(R)-NSC',stop_it_gritz_rl2,  all_ul_cost_gritz_rl2,cost_gritz_rl2, all_hg_err_gritz_rl2, 'solid', 'tab:olive', '*', .25]
    ,['RGen-M(R)',       stop_it_gritz_rb,   all_ul_cost_gritz_rb, cost_gritz_rb,  all_hg_err_gritz_rb, 'dashed', 'tab:green', 'd', .5]
    ,['RGen-M(R)-NSC',stop_it_gritz_rb2,  all_ul_cost_gritz_rb2,cost_gritz_rb2, all_hg_err_gritz_rb2, 'dashed', 'tab:olive', '*', .25]
    ,['RGen-S(M)',       stop_it_gritz_bs,   all_ul_cost_gritz_bs,cost_gritz_bs,  all_hg_err_gritz_bs, 'dotted', 'tab:purple','o',.4]
    ,['RGen-S(M)-NSC',stop_it_gritz_bs2,  all_ul_cost_gritz_bs2,cost_gritz_bs2, all_hg_err_gritz_bs2, 'dotted', 'tab:pink','*',.2]
    ,['RGen-L(M)',       stop_it_gritz_bl,   all_ul_cost_gritz_bl,cost_gritz_bl,  all_hg_err_gritz_bl, 'solid', 'tab:purple']
    ,['RGen-L(M)-NSC',stop_it_gritz_bl2,  all_ul_cost_gritz_bl2,cost_gritz_bl2, all_hg_err_gritz_bl2, 'solid', 'tab:pink','*',0.2]
    ,['RGen-M(M)',       stop_it_gritz_bb,   all_ul_cost_gritz_bb,cost_gritz_bb,  all_hg_err_gritz_bb, (0,(3,5,1,5)), 'tab:purple','d',0.4]
    ,['RGen-M(M)-NSC',stop_it_gritz_bb2,  all_ul_cost_gritz_bb2,cost_gritz_bb2, all_hg_err_gritz_bb2, (0,(3,5,1,5)), 'tab:pink','*',0.2]
   ] 




def select_results(to_select, all_plot_details):
    return [all_plot_details[i] for i in range(len(all_plot_details)) if all_plot_details[i][0] in to_select]
      

def table_comparison(stop_it_none, cost_none, all_hg_err_none, to_display):
    print('\n Recycle method  | Total its    | Total cost')
    print('-----------------|--------------|------------')
    print('{0:16} | {1:5}        | {2:7}'.format('None', sum(stop_it_none), sum(cost_none)))
    stop_it_none_sum = sum(stop_it_none)
    for items in to_display:
        label, stop_it, cost = items[0:3]
        stop_it_sum = sum(stop_it)
        print('{0:16} | {1:5} ({2:.2f}) | {3:7}'.format(label, sum(stop_it), stop_it_sum/stop_it_none_sum, sum(cost)))
        
table_comparison(stop_it_none, cost_none, all_hg_err_none, all_plot_details)


#%%% Create plots for comparing recycling strategies

torch.save(all_plot_details, SAVE_DIR+'all_plot_details')

# Ritz 
compare_performance(stop_it_none, ul_cost_none, cost_none, all_hg_err_none,                
              select_results(['Ritz-S', 'Ritz-M', 'Ritz-L'], all_plot_details))

# Gritz left
compare_performance(stop_it_none, ul_cost_none,cost_none, all_hg_err_none,                
              select_results(['RGen-S(L)', 'RGen-M(L)', 'RGen-L(L)', 'RGen-L(L)-NSC'], all_plot_details))
# Gritz right
compare_performance(stop_it_none, ul_cost_none,cost_none, all_hg_err_none,                
              select_results(['RGen-S(R)', 'RGen-M(R)', 'RGen-L(R)', 'RGen-L(R)-NSC'], all_plot_details))
compare_performance(stop_it_none, ul_cost_none,cost_none, all_hg_err_none,                
              select_results(['RGen-S(M)', 'RGen-M(M)', 'RGen-L(M)', 'RGen-L(M)-NSC'], all_plot_details))
# Compare the best ones
compare_performance(stop_it_none,ul_cost_none, cost_none, all_hg_err_none,
               select_results(['Ritz-S' , 'RGen-L(R)', 'RGen-L(R)-NSC'] , all_plot_details))

#%% Timings of GSVD computation
t_none  = extract_times(all_info_none)
t_current_gritz_rl, t_optim_gritz_rl, t_breakdown_gritz_rl = give_me_times(all_info_gritz_rl)
t_current_gritz_rl2, t_optim_gritz_rl2, t_breakdown_gritz_rl2 = give_me_times(all_info_gritz_rl2)
t_current_ritz_s, t_optim_ritz_s, t_breakdown_ritz_s = give_me_times(all_info_ritz_s)


def plot_timings(t_none, to_be_plot,  label='None', xlabel='Linear system number', fs=20, lw=2.5):
    fig, ax = plt.subplot_mosaic([['t','cum_t']], figsize=(10,4.5))
    ax['t'].set_ylabel('Computation time (s)', fontsize=fs)
    ax['cum_t'].set_ylabel('Cumulative time (s)', fontsize=fs)
    ax['t'].plot(t_none,linestyle=(0,(1,1)), color='k', label=label, linewidth=lw, marker='*', markevery=0.15, markersize=10)   
    ax['cum_t'].plot(np.cumsum(t_none),linestyle=(0,(1,1)), color='k', label=label, linewidth=lw, marker='*', markevery=0.15, markersize=10)   
    
    for i in range(len(to_be_plot)):
        if len(to_be_plot[i]) == 4:
            label, t_rec, linestyle, color = to_be_plot[i]
            marker, markerspace = None, None
        else:
            label, t_rec, linestyle, color, marker , markerspace = to_be_plot[i]
        
        ax['t'].plot(t_rec, color=color, linestyle=linestyle, linewidth=lw, label=label, marker=marker, markevery=markerspace, markersize=10 )
        ax['cum_t'].plot(np.cumsum(t_rec), color=color, linestyle=linestyle ,linewidth=lw, label=label, marker=marker, markevery=markerspace, markersize=10 )
                
        
    for a in ax:
        ax[a].set_xlabel('Linear system number',fontsize=fs,)
        ax[a].legend(fontsize=fs*.6)
    plt.tight_layout()


timings_to_plot = [
    ['Current', t_current_ritz_s, 'dotted', 'tab:blue','o',.4 ],
    ['w/o Dense matrix or saving', t_optim_ritz_s, 'dashdot', 'blue' ]]
plot_timings(t_none, [
    ['Current', t_current_gritz_rl, 'solid', 'tab:green'],
    ['w/o Dense matrix or saving', t_optim_gritz_rl, 'dashdot', 'green' ]])


plot_timings(t_none, [
    ['Current', t_current_gritz_rl2, 'solid', 'tab:olive', '*', .25 ],
    ['w/o Dense matrix or saving', t_optim_gritz_rl2, 'dashdot', 'olive' ]])

plot_ratios(t_current_ritz_s, t_breakdown_ritz_s, xlabel='Linear system number')
plot_ratios(t_current_gritz_rl, t_breakdown_gritz_rl, xlabel='Linear system number')
plot_ratios(t_current_gritz_rl2, t_breakdown_gritz_rl2, xlabel='Linear system number')


#%% How many iterations to get a good hypergradient?
"""How many iterations would we need to do in order to achieve a specific hypergradient relative error? 
Of cours,e we cannot do this in pracice but in the numerical experiment world this is possible to determine by pre-computing
high-accuracy Hessian solutions and their associated hypergradients."""

# rec_dim=10
stop_crit_mode = 'hg_err'
stop_crit_tol = 1e-2
all_w_none_stop, all_info_none_stop, all_W_dims_none_stop,  stop_it_none_stop, _, all_hg_err_none_stop, _, _ = compute_everything(None,None,1, full_info=True, stop_crit=stop_crit_mode, tol=stop_crit_tol, true_hgs=all_hgs_high)
cost_none_stop = calc_MINRES_cost(stop_it_none_stop, hess_cost , n)
all_w_gritz_rl_stop, all_info_gritz_rl_stop, all_W_dims_gritz_rl_stop, stop_it_gritz_rl_stop, cost_gritz_rl_stop, all_hg_err_gritz_rl_stop, all_W_spaces_gritz_rl_stop, all_U_spaces_gritz_rl_stop  = compute_everything('gritz_right','large',rec_dim, full_info=True, stop_crit=stop_crit_mode, tol=stop_crit_tol, true_hgs=all_hgs_high)
all_w_ritz_s_stop, all_info_ritz_s_stop, all_W_dims_ritz_s_stop, stop_it_ritz_s_stop, cost_ritz_s_stop, all_hg_err_ritz_s_stop, all_W_spaces_ritz_s_stop, all_U_spaces_ritz_s_stop = compute_everything('ritz','small',rec_dim, full_info=True, stop_crit=stop_crit_mode, tol=stop_crit_tol, true_hgs=all_hgs_high)



all_ul_cost_none_stop = calc_all_ul_cost(all_params, all_recons, all_w_none_stop)
all_ul_cost_ritz_s_stop = calc_all_ul_cost(all_params, all_recons, all_w_ritz_s_stop)
all_ul_cost_gritz_rl_stop = calc_all_ul_cost(all_params, all_recons, all_w_gritz_rl_stop)




none_stop_results = {'stop_it_none': stop_it_none_stop, 'ul_cost_none':all_ul_cost_none_stop, 'cost_none':cost_none_stop, 'all_hg_err_none':all_hg_err_none_stop}
torch.save(none_stop_results, SAVE_DIR+'none_stop_results')



all_plot_details_hg_stop = [['Ritz-S', stop_it_ritz_s_stop, all_ul_cost_ritz_s_stop, cost_ritz_s_stop, all_hg_err_ritz_s_stop, 'dotted', 'tab:blue','o',.4]
              ,['RGen-L(R)', stop_it_gritz_rl_stop,all_ul_cost_gritz_rl_stop,  cost_gritz_rl_stop, all_hg_err_gritz_rl_stop, 'solid', 'tab:green', 'd', .5]
                ] 

torch.save(all_plot_details_hg_stop, SAVE_DIR+'all_plot_details_hg_stop')

compare_performance(stop_it_none_stop, all_ul_cost_none_stop, cost_none_stop, all_hg_err_none_stop,
                all_plot_details_hg_stop)

#%%% Print comparison of iterations saved
print('Method |Normal SC | HG err SC')
print('None   |     {:.0f} |   {:.0f}' .format(sum(stop_it_none), sum(stop_it_none_stop)))
print('Ritz-S |     {:.0f} |   {:.0f}' .format(sum(stop_it_ritz_s), sum(stop_it_ritz_s_stop)))
print('GR-L(R)|     {:.0f} |   {:.0f}' .format(sum(stop_it_ritz_s), sum(stop_it_ritz_s_stop)))

#%% Inner vs outer recycling strategy
"Inner vs outer methods for recycling. How good is using current linear system rather than waiting "
"until next linear system (one which recycle space will be utilised with) is constructed?"
"For outer recycling, could we get away with only storing only the first few Krylov vectors? "
"How would the number of iterations saved change as we increase the number of Krylov vectors that are stored?"
all_w_gritz_rl_in2, all_info_gritz_rl_in2, all_W_dims_gritz_rl_in2, stop_it_gritz_rl_in2, cost_gritz_rl_in2, all_hg_err_gritz_rl_in2 = compute_everything('gritz_right','large',rec_dim, stop_crit='hg_err_approx', outer_rec=False)
all_ul_cost_gritz_rl_in2 = calc_all_ul_cost(all_params, all_recons, all_w_gritz_rl_in2)

all_w_gritz_rl_in, all_info_gritz_rl_in, all_W_dims_gritz_rl_in, stop_it_gritz_rl_in, cost_gritz_rl_in, all_hg_err_gritz_rl_in  = compute_everything('gritz_right','large',rec_dim, stop_crit='res', outer_rec=False)
all_ul_cost_gritz_rl_in = calc_all_ul_cost(all_params, all_recons, all_w_gritz_rl_in)

all_w_ritz_s_in, all_info_ritz_s_in, all_W_dims_ritz_s_in, stop_it_ritz_s_in, cost_ritz_s_in, all_hg_err_ritz_s_in, all_W_spaces_ritz_s_in, all_U_spaces_ritz_s_in= compute_everything('ritz','small',rec_dim, full_info=True, outer_rec=False)
all_ul_cost_ritz_s_in = calc_all_ul_cost(all_params, all_recons, all_w_ritz_s_in)


#%%%

plot_details_inner_v_iner = [
    ['Ritz-S Outer',        stop_it_ritz_s,       all_ul_cost_ritz_s,      cost_ritz_s,       all_hg_err_ritz_s, 'dotted', 'tab:blue','o',.4]
   ,['Ritz-S Inner',        stop_it_ritz_s_in,    all_ul_cost_ritz_s_in,   cost_ritz_s_in,    all_hg_err_ritz_s_in, 'dashed', 'blue']
   ,['RGen-L(R) Outer',     stop_it_gritz_rl,     all_ul_cost_gritz_rl,    cost_gritz_rl,     all_hg_err_gritz_rl, 'solid', 'tab:green']
   ,['RGen-L(R) Inner',     stop_it_gritz_rl_in,  all_ul_cost_gritz_rl_in, cost_gritz_rl_in,  all_hg_err_gritz_rl_in, 'dashdot', 'green', '*', .25]
   ,['RGen-L(R)-NSC Outer', stop_it_gritz_rl2,    all_ul_cost_gritz_rl2,   cost_gritz_rl2,    all_hg_err_gritz_rl2, 'solid', 'tab:olive', '*', .25]
   ,['RGen-L(R)-NSC Inner', stop_it_gritz_rl_in2, all_ul_cost_gritz_rl_in2,cost_gritz_rl_in2, all_hg_err_gritz_rl_in2, 'dashdot', 'olive']
   ]


# def calc_w_err(all_ws):
#     return np.array([torch.norm(all_ws[i] - all_w_high[i]).cpu().numpy() / torch.norm(all_w_high[i]).cpu().numpy() for i in range(len(all_ws))])
# plt.figure()
# plt.plot(calc_w_err(all_w_ritz_s_in), label='Ritz Inner')
# plt.plot(calc_w_err(all_w_ritz_s), label='Ritz Out')
# # plt.plot(all_hg_err_gritz_rl_in, label='GRitz')
# # plt.plot(all_hg_err_gritz_rl_in2, label='GRitz2')
# plt.yscale('log')


torch.save(plot_details_inner_v_iner, SAVE_DIR+'inner_v_iner')


compare_performance(stop_it_none, ul_cost_none, cost_none, all_hg_err_none,
                plot_details_inner_v_iner)
compare_performance(stop_it_none, ul_cost_none, cost_none, all_hg_err_none,
                select_results(['Ritz-S Outer', 'Ritz-S Inner'] , plot_details_inner_v_iner))

compare_performance(stop_it_none, ul_cost_none, cost_none, all_hg_err_none,
                select_results(['RGen-L(R) Outer', 'RGen-L(R) Inner'] , plot_details_inner_v_iner))
compare_performance(stop_it_none, ul_cost_none, cost_none, all_hg_err_none,
                select_results(['RGen-L(R)-NSC Outer', 'RGen-L(R)-NSC Inner'] , plot_details_inner_v_iner))


compare_performance(stop_it_none, ul_cost_none, cost_none, all_hg_err_none,
                select_results(['Ritz-S Inner', 'RGen-L(R) Inner', 'RGen-L(R)-NSC Inner'] , plot_details_inner_v_iner))




#%% Dimension of the recycle space
"What is a good dimension for the recycle space? Since the RMINRES algorithm has computational "
"cost associated with the dimension, there will be a cost tradeoff. As increase size, what happens "
"to performance? Would expect total number of iterates to decrease to zero"


p1, p2 = 10, 11
recycle_dim = 30

JW = torch.randn(p1,p2)
WtHW = torch.randn(p2,p2)

V_J,V_H,X,S_J,S_H, t_decomp, t_save = compute_gsvd(JW, WtHW)

Xinv = torch.linalg.inv(X)

number_of_recycle_vectors = min(recycle_dim, p2)

U_tmp=  Xinv[:,-number_of_recycle_vectors:]

ResPrecond =  (S_J @ torch.diag(1/torch.diag(S_H)))[-number_of_recycle_vectors:,-number_of_recycle_vectors:] @ V_H[:,-number_of_recycle_vectors:].T
        

warm_start=True

strategy_ritz, size_ritz ='ritz','small'
strategy_gritz, size_gritz ='gritz_right','large'

recycle_dims = np.array([5*(i)+1 for i in range(20)])


all_dim_ritz_ws = [None for _ in range(len(recycle_dims))]
all_dim_ritz_infos = [None for _ in range(len(recycle_dims))]
all_dim_ritz_W_dims = [None for _ in range(len(recycle_dims))]
all_dim_ritz_stop_its = [None for _ in range(len(recycle_dims))]

all_dim_gritz_ws = [None for _ in range(len(recycle_dims))]
all_dim_gritz_infos = [None for _ in range(len(recycle_dims))]
all_dim_gritz_W_dims = [None for _ in range(len(recycle_dims))]
all_dim_gritz_stop_its = [None for _ in range(len(recycle_dims))]

all_dim_gritz2_ws = [None for _ in range(len(recycle_dims))]
all_dim_gritz2_infos = [None for _ in range(len(recycle_dims))]
all_dim_gritz2_W_dims = [None for _ in range(len(recycle_dims))]
all_dim_gritz2_stop_its = [None for _ in range(len(recycle_dims))]

for ind in range(len(recycle_dims)):
    print('INDEX : '+str(ind))
    print('Rec dim: {:3} | #{}/{}'.format(recycle_dims[ind], ind+1, len(recycle_dims)))

    all_dim_ritz_ws[ind], all_dim_ritz_infos[ind], all_dim_ritz_W_dims[ind] = solve_hess_seqn(all_params, all_recons, recycle_dims[ind], warm_start=warm_start, strategy=strategy_ritz, size=size_ritz, tol = setup['solver_optns']['hess_sys']['tol'], stop_crit='res' )
    all_dim_ritz_stop_its[ind] = np.array([inf['stop_it'] for inf in all_dim_ritz_infos[ind]]) 
    
    all_dim_gritz_ws[ind], all_dim_gritz_infos[ind], all_dim_gritz_W_dims[ind] = solve_hess_seqn(all_params, all_recons, recycle_dims[ind], warm_start=warm_start, strategy=strategy_gritz, size=size_gritz, tol = setup['solver_optns']['hess_sys']['tol'], stop_crit='res' )
    all_dim_gritz_stop_its[ind] = np.array([inf['stop_it'] for inf in all_dim_gritz_infos[ind]]) 
    
    all_dim_gritz2_ws[ind], all_dim_gritz2_infos[ind], all_dim_gritz2_W_dims[ind] = solve_hess_seqn(all_params, all_recons, recycle_dims[ind], warm_start=warm_start, strategy=strategy_gritz, size=size_gritz, tol = setup['solver_optns']['hess_sys']['tol'], stop_crit='hg_err_approx' )
    all_dim_gritz2_stop_its[ind] = np.array([inf['stop_it'] for inf in all_dim_gritz2_infos[ind]]) 

#%%% Calculate hypergradient errors
all_dim_ritz_hg_err   =  [calc_hypgrad_errs(ws, all_w_high) for ws in all_dim_ritz_ws]
all_dim_gritz_hg_err  =  [calc_hypgrad_errs(ws, all_w_high) for ws in all_dim_gritz_ws]
all_dim_gritz2_hg_err =  [calc_hypgrad_errs(ws, all_w_high) for ws in all_dim_gritz2_ws]


all_dim_ritz_max_hg_err = np.max(all_dim_ritz_hg_err, axis=1)
all_dim_gritz_max_hg_err = np.max(all_dim_gritz_hg_err, axis=1)
all_dim_gritz2_max_hg_err = np.max(all_dim_gritz2_hg_err, axis=1)


#%%%

all_dim_gritz2_max_W = np.max(all_dim_gritz2_W_dims, axis=1)
all_dim_gritz_max_W = np.max(all_dim_gritz_W_dims, axis=1)
all_dim_ritz_max_W = np.max(all_dim_ritz_W_dims, axis=1)


#%%% Plot results for dimension of recycle space experiment

def plot_dim_comp_data(none, ritz, gritz, gritz2, xvals=None, plot_none=True, title=None, ylabel=None, no_sum=False, fontsize=20, xlabel='Dimension of recycle space', linewidth=3):
    plt.figure(figsize=(6.5,5.5))
    if xvals is None:
        xvals = np.array([i+1 for i in range(len(ritz))])
    if plot_none:
        if no_sum: y = none
        else: y = sum(none)
        plt.axhline(y = y, xmin = 0.04, xmax = .96,linestyle=(0,(1,1)), color='k', label='None', linewidth=linewidth) 
    plt.plot(xvals, ritz,   label='Ritz-S',   color='tab:blue', linestyle='dotted',  linewidth=linewidth)
    plt.plot(xvals, gritz,  label='RGen-L(R)', color='tab:green', linewidth=linewidth, marker='d', markevery=.5)
    plt.plot(xvals, gritz2, label='RGen-L(R)-NSC', color='tab:olive', linewidth=linewidth, marker='d', markevery=.25)
    plt.title(title,fontsize=fontsize)
    plt.ylabel(ylabel,fontsize=fontsize)
    plt.xlabel(xlabel,fontsize=fontsize)
    plt.yticks(fontsize =fontsize*.8) 
    plt.xticks(fontsize =fontsize*.8) 
    plt.legend(fontsize=fontsize*.8)
    plt.tight_layout()
    
def calc_cumulative_time(all_infos, item_name='t_total'):
    t_seq = np.array([extract_times(hess_seq, item_name=item_name) for hess_seq in all_infos])
    return np.sum(t_seq, axis=1)

def calc_all_cumulative_times(all_infos):
    t_dense = calc_cumulative_time(all_infos, 't_make_dense') # Time for construct dense matrix
    t_decomp = calc_cumulative_time(all_infos, 't_decomp') # Time for projected matrix decomposition
    t_save = calc_cumulative_time(all_infos, 't_save') # Time for saving/loading hack for MATLAB conversion
    t_rminres = calc_cumulative_time(all_infos, 't_total') # Time for RMINRES solve

    return  t_dense, t_decomp, t_save, t_rminres

cost_minres_ritz   = calc_RMINRES_cost(np.array(all_dim_ritz_stop_its).T,   hess_cost , n, recycle_dims).T
cost_minres_gritz  = calc_RMINRES_cost(np.array(all_dim_gritz_stop_its).T,  hess_cost , n, recycle_dims).T
cost_minres_gritz2 = calc_RMINRES_cost(np.array(all_dim_gritz2_stop_its).T, hess_cost , n, recycle_dims).T

t_minres_none = extract_times(all_info_none)
t_cum_dense_ritz,   t_cum_decomp_ritz,   t_cum_save_ritz,   t_cum_rminres_ritz   = calc_all_cumulative_times(all_dim_ritz_infos)
t_cum_dense_gritz,  t_cum_decomp_gritz,  t_cum_save_gritz,  t_cum_rminres_gritz  = calc_all_cumulative_times(all_dim_gritz_infos)
t_cum_dense_gritz2, t_cum_decomp_gritz2, t_cum_save_gritz2, t_cum_rminres_gritz2 = calc_all_cumulative_times(all_dim_gritz2_infos)

t_cum_total_ritz   =  t_cum_dense_ritz  + t_cum_decomp_ritz  + t_cum_save_ritz  + t_cum_rminres_ritz
t_cum_total_gritz  =  t_cum_dense_gritz + t_cum_decomp_gritz + t_cum_save_gritz + t_cum_rminres_gritz
t_cum_total_gritz2 =  t_cum_dense_gritz2+ t_cum_decomp_gritz2+ t_cum_save_gritz2+ t_cum_rminres_gritz2




#% Comparison of computation time
plot_dim_comp_data(t_minres_none,t_cum_total_ritz,t_cum_total_gritz,t_cum_total_gritz2,  xvals=recycle_dims,
                   ylabel='Total computation time')
plot_dim_comp_data(t_minres_none,t_cum_rminres_ritz,t_cum_rminres_gritz,t_cum_rminres_gritz2, xvals=recycle_dims,
                   ylabel='Total RMINRES time')


plot_dim_comp_data(None,t_cum_decomp_ritz,t_cum_decomp_gritz,t_cum_decomp_gritz2, xvals=recycle_dims, plot_none=False,
                    ylabel='Total Decomp time')



# Factor of matrix decomposition speedup needed to save time
t_factor_ritz = (sum(t_minres_none) - t_cum_rminres_ritz ) / ( t_cum_dense_ritz +  t_cum_decomp_ritz)
t_factor_gritz = (sum(t_minres_none) - t_cum_rminres_gritz )/ (t_cum_decomp_gritz + t_cum_dense_gritz)
t_factor_gritz2 = (sum(t_minres_none) - t_cum_rminres_gritz2 )/ (t_cum_decomp_gritz2 + t_cum_dense_gritz2)
plot_dim_comp_data(t_minres_none,t_factor_ritz,t_factor_gritz,t_factor_gritz2, plot_none=False,
                   ylabel='Required speedup factor', xvals=recycle_dims)


plot_dim_comp_data(None,all_dim_ritz_max_W,all_dim_gritz_max_W,all_dim_gritz2_max_W, plot_none=False,
                   ylabel='Max dimension of W', xvals=recycle_dims)



state_to_save = {'rec_dims':recycle_dims, 'c_none':cost_none, 'c_ritz':cost_minres_ritz, 'c_gritz':cost_minres_gritz, 'c_gritz2':cost_minres_gritz2
                 ,'stop_ritz':all_dim_ritz_stop_its, 'stop_gritz':all_dim_gritz_stop_its, 'stop_gritz2':all_dim_gritz2_stop_its
                 ,'hg_ritz':all_dim_ritz_hg_err, 'hg_gritz':all_dim_gritz_hg_err, 'hg_gritz2':all_dim_gritz2_hg_err
                 }
torch.save(state_to_save, SAVE_DIR+'all_dims_results')


#% Comparison of FLOPS
plot_dim_comp_data(cost_none,np.sum(cost_minres_ritz, axis=1),np.sum(cost_minres_gritz, axis=1),np.sum(cost_minres_gritz2, axis=1),
                   ylabel='FLOPs of RMINRES', xvals=recycle_dims)

#% Comparison of number of iterations
plot_dim_comp_data(stop_it_none,np.sum(all_dim_ritz_stop_its, axis=1),np.sum(all_dim_gritz_stop_its, axis=1),np.sum(all_dim_gritz2_stop_its, axis=1),
                   ylabel='Total RMINRES its', xvals=recycle_dims)
#% Mean HG relative err
plot_dim_comp_data(np.log10(all_hg_err_none.mean()),np.log10(np.mean(all_dim_ritz_hg_err, axis=1)),np.log10(np.mean(all_dim_gritz_hg_err, axis=1)),np.log10(np.mean(all_dim_gritz2_hg_err, axis=1)), no_sum=True, 
                   ylabel='log10 Mean HG relative error', xvals=recycle_dims)
plot_dim_comp_data(np.log10(all_hg_err_none.max()),np.log10(np.max(all_dim_ritz_hg_err, axis=1)),np.log10(np.max(all_dim_gritz_hg_err, axis=1)),np.log10(np.max(all_dim_gritz2_hg_err, axis=1)), no_sum=True, 
                   ylabel='log10 Max HG relative error', xvals=recycle_dims)


plot_ratios(t_cum_total_ritz,   [t_cum_dense_ritz,   t_cum_decomp_ritz,   t_cum_save_ritz,   t_cum_rminres_ritz], xvals=recycle_dims)
plot_ratios(t_cum_total_gritz,  [t_cum_dense_gritz,  t_cum_decomp_gritz,  t_cum_save_gritz,  t_cum_rminres_gritz], xvals=recycle_dims)
plot_ratios(t_cum_total_gritz2, [t_cum_dense_gritz2, t_cum_decomp_gritz2, t_cum_save_gritz2, t_cum_rminres_gritz2], xvals=recycle_dims)
#%%Approximation of the hypergradient error
""" How good is our approximation of the hypergradient error using the projected GSVD?
We approximate \| J^{i} H^{i}^{-1} r_k \| with \| J^{i}W(W^T H^{i} W)^{-1} WT  r_k \| 
where W=W^{i}. We compare these quantities and, to determine how much is the error introduced
by doing a projection vs doing a projection onto a subspace whose solution space is techinically
not that associated to H^i (but rather H^{i-1}), for W=W^{i+1}.
"""

all_w_ritz_s, all_info_ritz_s, all_W_dims_ritz_s, stop_it_ritz_s, cost_ritz_s, all_hg_err_ritz_s, all_W_spaces_ritz_s, all_U_spaces_ritz_s = compute_everything('ritz','small',rec_dim, full_info=True)
all_w_gritz_rl, all_info_gritz_rl, all_W_dims_gritz_rl, stop_it_gritz_rl, cost_gritz_rl, all_hg_err_gritz_rl, all_W_spaces_gritz_rl, all_U_spaces_gritz_rl  = compute_everything('gritz_right','large',rec_dim, full_info=True)
all_w_gritz_rl2, all_info_gritz_rl2, all_W_dims_gritz_rl2, stop_it_gritz_rl2, cost_gritz_rl2, all_hg_err_gritz_rl2, all_W_spaces_gritz_rl2, all_U_spaces_gritz_rl2  = compute_everything('gritz_right','large',rec_dim, full_info=True, stop_crit='hg_err_approx')

#%%% Useful functions

def create_JH_gsvd(hess_ind):
    ll_fun_single = create_lower_level()
    H, g, x = make_hessian_system(torch.tensor(all_params[hess_ind] , device=device), n=n, ll_fun=ll_fun_single, recon=all_recons[hess_ind][0])
    Hmat = construct_dense_matrix(H, n**2)
    
    hyp_grad_op = hyp_grad(x, ll_fun_single)
    Jmat = construct_dense_matrix(hyp_grad_op,  p, n**2)

    V_J,V_H,Xt,S_J,S_H, _, _ = compute_gsvd(Jmat, Hmat)
    
    return hyp_grad_op, H,g, V_J,V_H,Xt,S_J,S_H


def make_projected_gsvd(W,H,hyp_grad_op):
    WtHW = torch.empty_like(W) 
    JW = torch.empty( p, W.shape[-1], device=W.device)
    for ind in range(W.shape[-1]):
        WtHW[:,ind] = H(W[:,ind])
        JW[:,ind] = hyp_grad_op(W[:,ind])
        
    WtHW = W.T @ WtHW 
    V_J,V_H,X,S_J,S_H, _, _ = compute_gsvd(JW,WtHW) 
    
    # gc.collect()
    return V_J,V_H,X,S_J,S_H

def make_precond_mat(W,V_J,V_H,X,S_J,S_H, rec_dim=15 ):
    number_of_recycle_vectors = min(rec_dim, W.shape[-1])
    ResPrecond =  V_J[:,-number_of_recycle_vectors:] @(S_J @ torch.diag(1/torch.diag(S_H)))[-number_of_recycle_vectors:,-number_of_recycle_vectors:] @ V_H[:,-number_of_recycle_vectors:].T @ W.T
    return ResPrecond

def mult_ws_by_mat(ws, mat, H, g ):
    return np.array([torch.norm(mat @ (H(w)-g)).cpu().numpy() for w in ws])


def plot_comparison_of_hg_err_approx(actual, approx, cheat_approx, label1='Projected', nolog=False,label2='Projected cheat', title='', xlabel='RMINRES iteration', ylabel='', fontsize=20):
    plt.figure(figsize=(5.5,4.5))
    if nolog:
        plt1, plt2, plt3 = actual, approx, cheat_approx
    else:
        plt1, plt2, plt3 = np.log10(actual), np.log10(approx), np.log10(cheat_approx)
    
    xplt = np.arange(len(plt1))+1
    plt.plot(xplt,plt1, ':', label='True', linewidth=3, color='black')
    plt.plot(xplt,plt2, label=label1, linewidth=3, color='tab:olive', marker='d', markevery=.25)
    plt.plot(xplt,plt3, ':', label=label2, linewidth=3, color='tab:olive', marker='d', markevery=.25)
    plt.legend(fontsize=fontsize*.75)
    plt.xlabel(xlabel,fontsize=fontsize*.8)
    plt.ylabel(ylabel,fontsize=fontsize*.8)
    plt.title(title,fontsize=fontsize)
    plt.tight_layout()
    # plt.xticks(np.arange(len(plt1)), np.arange(1, len(plt1)+1))
    

#%% See how application of the approx JHinv actually compare in evaluation of the norm

def calc_res_norms(H,ws):
    return np.array([H(w).norm().cpu().numpy() for w in ws])



def experiment_JHinv_comp(hess_ind):

    hyp_grad_op, H, g, V_J0,V_H0,X0t,S_J0,S_H0 = create_JH_gsvd(hess_ind+1)
    V_Jcheat,V_Hcheat,Xcheat,S_Jcheat,S_Hcheat = make_projected_gsvd(all_W_spaces_gritz_rl2[hess_ind+1],H,hyp_grad_op)
    V_J2,V_H2,X2,S_J2,S_H2 = make_projected_gsvd(all_W_spaces_gritz_rl2[hess_ind],H,hyp_grad_op)
    
    JHinv = V_J0 @ (S_J0 @ torch.diag(1/torch.diag(S_H0))) @ V_H0.T 
    ResPrecondcheat = make_precond_mat(all_W_spaces_gritz_rl2[hess_ind+1], V_Jcheat ,V_Hcheat ,Xcheat ,S_Jcheat ,S_Hcheat )
    ResPrecond2 = make_precond_mat(all_W_spaces_gritz_rl2[hess_ind], V_J2,V_H2,X2,S_J2,S_H2)
    JHinv_r_gritz_true = mult_ws_by_mat(all_info_gritz_rl2[hess_ind+1]['xs'], JHinv, H, g)
    JHinv_approx_r_gritz_cheat = mult_ws_by_mat(all_info_gritz_rl2[hess_ind+1]['xs'], ResPrecondcheat, H, g)
    JHinv_approx_r_gritz2 = mult_ws_by_mat(all_info_gritz_rl2[hess_ind+1]['xs'], ResPrecond2, H, g)
    
    state_dict = {'true':JHinv_r_gritz_true, 'approx':JHinv_approx_r_gritz2, 'cheat':JHinv_approx_r_gritz_cheat}
    torch.save(state_dict, SAVE_DIR+'hg_err_comp_system'+str(hess_ind))
    
    plot_comparison_of_hg_err_approx(JHinv_r_gritz_true,JHinv_approx_r_gritz2,JHinv_approx_r_gritz_cheat, 
                                     label1='$W^{('+str(hess_ind)+')}$ approx',  label2='$W^{('+str(hess_ind+1)+')}$ approx',
                                     title='Hessian system '+str(hess_ind), ylabel='log10 HG error')

experiment_JHinv_comp(94)
experiment_JHinv_comp(59)
experiment_JHinv_comp(39)
experiment_JHinv_comp(14)

#%% Compute final difference between true and approximate HG error for all Hessian systems

num_recycle_systems = len(all_W_spaces_gritz_rl2)-1
all_JHinv_r_gritz2 = np.empty(num_recycle_systems)
all_JHinv_approx_r_gritz  = np.empty(num_recycle_systems) # Cheating projection
all_JHinv_approx_r_gritz2  = np.empty(num_recycle_systems) # {Projection used in practice
all_JHinv_norm = np.empty(num_recycle_systems)
all_JHinv_approx_norm = np.empty(num_recycle_systems) # Cheating projection
all_JHinv_approx_norm2 = np.empty(num_recycle_systems) # Projection used in practice


all_approx_err_gritz  = np.empty(num_recycle_systems) # Cheat
all_approx_err_gritz2  = np.empty(num_recycle_systems) #  Used in practice


for hess_ind in tqdm(range(num_recycle_systems), desc='Accuracy of JHinv approx'):
    hyp_grad_op, H, g, V_J0,V_H0,X0t,S_J0,S_H0 = create_JH_gsvd(hess_ind+1)
    V_J,V_H,X,S_J,S_H = make_projected_gsvd(all_W_spaces_gritz_rl2[hess_ind+1],H,hyp_grad_op)
    V_J2,V_H2,X2,S_J2,S_H2 = make_projected_gsvd(all_W_spaces_gritz_rl2[hess_ind],H,hyp_grad_op)
    
    JHinv = V_J0 @ (S_J0 @ torch.diag(1/torch.diag(S_H0))) @ V_H0.T 
    ResPrecond = make_precond_mat(all_W_spaces_gritz_rl2[hess_ind+1], V_J,V_H,X,S_J,S_H)
    ResPrecond2 = make_precond_mat(all_W_spaces_gritz_rl2[hess_ind], V_J2,V_H2,X2,S_J2,S_H2)
    
    all_JHinv_norm[hess_ind] = torch.linalg.norm(JHinv, ord=2)
    all_JHinv_approx_norm[hess_ind] = torch.linalg.norm(ResPrecond, ord=2)
    all_JHinv_approx_norm2[hess_ind] = torch.linalg.norm(ResPrecond2, ord=2)
    
    gritz_rl2_tmp = mult_ws_by_mat([all_w_gritz_rl2[hess_ind+1]], JHinv, H, g)
    
    # Consider entire sequence of Hessian system
    JHinv_r_gritz2 = mult_ws_by_mat(all_info_gritz_rl2[hess_ind+1]['xs'], JHinv, H, g)
    JHinv_approx_r_gritz = mult_ws_by_mat(all_info_gritz_rl2[hess_ind+1]['xs'], ResPrecond, H, g)
    JHinv_approx_r_gritz2 = mult_ws_by_mat(all_info_gritz_rl2[hess_ind+1]['xs'], ResPrecond2, H, g)
    all_approx_err_gritz[hess_ind] = np.linalg.norm(JHinv_r_gritz2 - JHinv_approx_r_gritz)
    all_approx_err_gritz2[hess_ind] = np.linalg.norm(JHinv_r_gritz2 - JHinv_approx_r_gritz2)
    
    # Only multiple by final solution of Hessian system
    all_JHinv_r_gritz2[hess_ind] = mult_ws_by_mat([all_w_gritz_rl2[hess_ind+1]], JHinv, H, g)[0]
    all_JHinv_approx_r_gritz[hess_ind] = mult_ws_by_mat([all_w_gritz_rl2[hess_ind+1]], ResPrecond, H, g)[0]
    all_JHinv_approx_r_gritz2[hess_ind] = mult_ws_by_mat([all_w_gritz_rl2[hess_ind+1]], ResPrecond2, H, g)[0]
    
    
    
state_dict = {'true':all_JHinv_r_gritz2, 'approx':all_JHinv_approx_r_gritz2, 'cheat':all_JHinv_approx_r_gritz,
              'truenorm':all_JHinv_norm, 'approxnorm':all_JHinv_approx_norm2 , 'cheatnorm':all_JHinv_approx_norm }
torch.save(state_dict, SAVE_DIR+'all_hg_err_comp')


plot_comparison_of_hg_err_approx((all_JHinv_r_gritz2),(all_JHinv_approx_r_gritz2),(all_JHinv_approx_r_gritz), 
                                 label1='$W^{(i)}$ approx',  label2='$W^{(i+1)}$ approx',
                                 ylabel='log10 HG error', xlabel='Linear system number')

plot_comparison_of_hg_err_approx(all_JHinv_norm,all_JHinv_approx_norm2,(all_JHinv_approx_norm), 
                                 label1='$W^{(i)}$ approx',  label2='$W^{(i+1)}$ approx',
                                 ylabel='Matrix 2-norm of preconditioner', xlabel='Linear system number', nolog=True)




