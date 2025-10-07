# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 07:42:02 2024

@author: Sebastian J. Scott

Re-solve sequence of Hessian systems using various recycling strategies.

"""

import torch
from tqdm import tqdm
import numpy as np

from src.utils import set_device, set_seed, select_results
from src.create_cost_functions import create_cost_functions

from src import plotting as pltt
import src.recycle_utils as rutils

import json 

import src.utils as utils
from src.load_dataset import create_dataset, create_dataloader

torch.no_grad()
torch.set_default_dtype(torch.float64)

#%% Load data

LOAD_DIR    = './results/gaussian_blur/bilevel_data/'
SAVE_DIR    = './results/gaussian_blur/recycle_data/'
FIGURES_DIR = './results/gaussian_blur/figures/'

do_experiments = {
    'calculate_test_recon':
        # True
        False
    ,'justify_recycling':
        # True
        False
    ,'full_decomp_comparison':
        # True
        False
    ,'cg_vs_minres':
        # True
        False
    ,'compare_all_strategies':
        True
        # False
    ,'true_hg_stop_crit':
        # True
        False
    ,'inner_vs_outer':
        # True
        False
    ,'explore_recycle_space_dimension':
        # True
        False
    ,'hg_approx_accuracy':
        # True
        False
    }

do_info_plots = False

rec_dim = 30 # Dimension of recycle space to be considered throughout

#%% Automated selection of quantities

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)
device = set_device(verbose=False)

with open(LOAD_DIR+'setup.json') as f:
    setup = json.load(f)

all_params = torch.load(LOAD_DIR+'history_params.pt')
all_logs = torch.load(LOAD_DIR+'all_logs.pt')

if do_info_plots:
    summary = utils.sum_state_keys(setup, all_logs)
    pltt.plot_timings(summary)
    
    pltt.display_filters( all_params[0], square=False,
                  save=FIGURES_DIR+'plot_filters_init.png')
    
    pltt.display_filters( all_params[-1], square=False,
                  save=FIGURES_DIR+'plot_filters_learned.png')

#%% Generate test dataset

set_seed(0) # Reproducability
train_dataset = create_dataset(setup['problem_setup'])
train_dataloader = create_dataloader(train_dataset, batch_size = 1)

test_setup = setup['problem_setup'].copy()
if test_setup['forward_op'] == 'identity':
    test_setup['forward_op'] = 'gaussian_blur'
test_setup['train']= False
test_setup['data_num'] = 50
test_setup['sigma'] = 3.
test_setup['noiselevel'] = .2

torch.manual_seed(0)
test_dataset = create_dataset(test_setup)
test_dataloader = create_dataloader(test_dataset, batch_size = 1)


#%%% Compute and save reconstructions of test dataset
if do_experiments['calculate_test_recon']:
    for batch in test_dataloader: 
        batch = utils.to_device(batch, device)
        y_test = batch['forward_image'][0][0]
        xexact_test = batch['image'][0][0]
        A_test = batch['forward_op'][0]
        # break
    
    ul_fun, ll_fun = create_cost_functions(setup['problem_setup']['regulariser'],device=device)
    ll_solver = utils.get_lower_level_solver(setup, ll_fun, xexact_test.shape, device)
    
    recon_test = ll_solver(all_params[-1], A_test, y_test)
    psnr_test = utils.psnr(xexact_test, recon_test)
    recon0_test = ll_solver(all_params[0], A_test, y_test)
    psnr0_test = utils.psnr(xexact_test, recon0_test)
    
    psnry_test = utils.psnr(xexact_test, y_test)
    
    pltt.display_recons(xexact_test,[y_test, recon_test, recon0_test],
                   subtitles=[f'Measurement\n(PSNR: {psnry_test:.2f})', 
                              f'FoE Recon\n(PSNR: {psnr_test:.2f})', 
                              f'DCT Recon\n(PSNR: {psnr0_test:.2f})'
                              ],
                   save=FIGURES_DIR+'plot_test_reconstructions.png')


#%% Get train set
for batch in train_dataloader: 
    batch = utils.to_device(batch, device)
    y = batch['forward_image'][0][0]
    xexact = batch['image'][0][0]
    A = batch['forward_op'][0]
    idx = batch['idx'][0]
    state = all_logs[idx]
    

    seqn_recons = [ x[0] for x in state['recon'][:setup['solver_optns']['ul_solve']['max_its']]]
    seqn_params = [all_params[i] for i in state['step'][:setup['solver_optns']['ul_solve']['max_its']]]
    
    # Function that returns J_op, Hessian, g, and recon for a given parameter/Hessian system index
    hessian_maker = rutils.hessian_system_generator(setup, xexact, A, y, seqn_params, seqn_recons, device)
    
    if idx == 1: break

#%% Compute high and normal accuracy solutions

baseline = {}
baseline['high'] =  rutils.compute_high_accuracy(hessian_maker)

calculate_results = rutils.prepare_calculator(hessian_maker, baseline['high']['hgs'])
baseline['none'] = calculate_results(name='MINRES warm', full_info=False)

torch.save(baseline,SAVE_DIR+'results_baseline.pt')

#%% Justification for recycling
"Verify that the linear systems are similar and so utilising recycling is a reasonable idea to make."
if do_experiments['justify_recycling']:
    results_justify = rutils.justify_recycling(setup, baseline['none'], hessian_maker,
                                    save=SAVE_DIR+'results_justify.pt')    

    pltt.plot_data(torch.log10(results_justify['dif_hess']/results_justify['norm_hess']),
                   ylabel= 'log10 $H^{(i)}$ rel. difference', 
                   xlabel='Linear system number',
                   save = FIGURES_DIR+'justify_hess.png')    
    pltt.plot_data(torch.log10(results_justify['dif_rhs']/results_justify['norm_rhs']),
                   ylabel='log10 $g^{(i)}$ rel. difference', 
                   xlabel='Linear system number',
                   save = FIGURES_DIR+'justify_rhs.png')    
    pltt.plot_data(torch.log10(results_justify['dif_w']/results_justify['norm_w']),
                   ylabel='log10 $w^{(i)}$ rel. difference',
                   xlabel='Linear system number',
                   save = 'justify_w.png')    
    pltt.plot_data(torch.log10(results_justify['dif_recon']/results_justify['norm_recon']),
                   ylabel=r'log10 $x(\theta^{i+1})$ rel. dif.', 
                   xlabel='Linear system number',
                   save = FIGURES_DIR+'justify_recon.png')      

#%% Comparison between Ritz (generalised) vectors and (eigen/singular)vectors}
"How does recycling utilising Ritz vectors compare to recycling using the actual eigenvectors we are wanting to approximate?"

results = {}

results['Ritz-S']    = calculate_results(name='Ritz-S',    strategy='ritz',        size='small', rec_dim=rec_dim, full_info=True)
results['RGen-L(R)'] = calculate_results(name='RGen-L(R)', strategy='gritz_right', size='large', rec_dim=rec_dim, full_info=True)

if do_experiments['full_decomp_comparison']:
    results['Eig-S']     = calculate_results(name='Eig-S',     strategy='eig',         size='small', rec_dim=rec_dim)
    results['GSVD-L(R)'] = calculate_results(name='GSVD-L(R)', strategy='gsvd_right',  size='large', rec_dim=rec_dim)

    pltt.compare_performance(baseline['none'], select_results(['Eig-S', 'Ritz-S', 'GSVD-L(R)', 'RGen-L(R)'], results),
                        save=FIGURES_DIR+'full_decomp_vs_projected.png')

#%% MINRES vs CG
"""CG exploits positive definiteness and symmetry and is the common choice for solving sequence of Hessian systems.
We employ MINRES which only exploits symmetry. What is the impact of performance? Compare with warm vs cold start"""

if do_experiments['cg_vs_minres']:
    results['MINRES cold'] = calculate_results(name='MINRES cold', full_info=False, warm_start=False)
    results['CG warm'] = calculate_results(name='CG warm', full_info=False, solver='cg')
    results['CG cold'] = calculate_results(name='CG cold', full_info=False, solver='cg', warm_start=False)
    
    pltt.compare_performance(baseline['none'], select_results(['MINRES cold', 'CG warm',  'CG cold'], results), baseline_label='MINRES warm',
                        save=FIGURES_DIR+'cold_vs_warm.png')

#%% Information associated with small vs large (generalised singular/eigen) values
"Small or large Ritz value information better suited for this application? Same for  small or large Ritz"
" generalised value information?"

results['RGen-L(R)-NSC'] = calculate_results(name='RGen-L(R)-NSC', strategy='gritz_right', size='large', rec_dim=rec_dim, full_info=True, stop_crit='hg_err_approx')

if do_experiments['compare_all_strategies']:
    results_config = [
        # Format: (name, strategy, size, full_info=False, stop_crit=None)
        # Ritz
        ('Ritz-L', 'ritz', 'large'),
        ('Ritz-M', 'ritz',   'mix'),
        
        # Harmonic Ritz
        ('HRitz-S', 'harmonic_ritz', 'small'),
        ('HRitz-L', 'harmonic_ritz', 'large'),
        ('HRitz-M', 'harmonic_ritz',   'mix'),
    
        # Generalized Ritz
        ('RGen-S(L)', 'gritz_left',  'small'),
        ('RGen-L(L)', 'gritz_left',  'large'),
        ('RGen-M(L)', 'gritz_left',    'mix'),
        ('RGen-S(R)', 'gritz_right', 'small'),
        ('RGen-L(R)', 'gritz_right', 'large'),
        ('RGen-M(R)', 'gritz_right',   'mix'),
        ('RGen-S(M)', 'gritz_both',  'small'),
        ('RGen-L(M)', 'gritz_both',  'large'),
        ('RGen-M(M)', 'gritz_both',    'mix'),
    
        # New stopping criterion
        ('RGen-L(L)-NSC', 'gritz_left',  'large', False, 'hg_err_approx'),
        ('RGen-S(L)-NSC', 'gritz_left',  'small', False, 'hg_err_approx'),
        ('RGen-M(L)-NSC', 'gritz_left',    'mix', False, 'hg_err_approx'),
        ('RGen-S(R)-NSC', 'gritz_right', 'small', False, 'hg_err_approx'),
        ('RGen-L(R)-NSC', 'gritz_right',   'mix', False, 'hg_err_approx'),
        ('RGen-M(R)-NSC', 'gritz_right',   'mix', False, 'hg_err_approx'),
        ('RGen-S(M)-NSC', 'gritz_both',  'small', False, 'hg_err_approx'),
        ('RGen-L(M)-NSC', 'gritz_both',  'large', False, 'hg_err_approx'),
        ('RGen-M(M)-NSC', 'gritz_both',    'mix', False, 'hg_err_approx'),
    ]
    
    for cfg in results_config:
        # Unpack config with defaults
        name, strategy, size, *optional = cfg
        full_info = optional[0] if len(optional) > 0 else False
        stop_crit = optional[1] if len(optional) > 1 else 'res'
    
        results[name] = calculate_results(
            name=name,
            strategy=strategy,
            size=size,
            rec_dim=rec_dim,
            full_info=full_info,
            stop_crit=stop_crit
        )
    
    
    pltt.table_comparison(baseline['none'], results)
    pltt.compare_performance(baseline['none'], select_results(['Ritz-S' , 'RGen-L(R)', 'RGen-L(R)-NSC'], results),
                        save=FIGURES_DIR+'compare_best.png')
    pltt.compare_performance(baseline['none'], select_results(['Ritz-S', 'RGen-L(L)', 'RGen-L(L)-NSC'], results),
                        save=FIGURES_DIR+'compare_best.png')
    pltt.compare_performance(baseline['none'], select_results(['Ritz-S' , 'RGen-L(M)', 'RGen-L(M)-NSC'], results),
                        save=FIGURES_DIR+'compare_best.png')
    #%%
    # Save and display results
    
    # Ritz 
    pltt.compare_performance(baseline['none'], select_results(['Ritz-S', 'Ritz-M', 'Ritz-L'], results),
                        save=FIGURES_DIR+'compare_ritz.png')
    # HRitz 
    pltt.compare_performance(baseline['none'], select_results(['HRitz-S', 'HRitz-M', 'HRitz-L'], results),
                        save=FIGURES_DIR+'compare_hritz.png')
    # Gritz left
    pltt.compare_performance(baseline['none'], select_results(['RGen-S(L)', 'RGen-M(L)', 'RGen-L(L)', 'RGen-L(L)-NSC'], results),
                        save=FIGURES_DIR+'compare_gritz_left.png')
    # Gritz right
    pltt.compare_performance(baseline['none'], select_results(['RGen-S(R)', 'RGen-M(R)', 'RGen-L(R)', 'RGen-L(R)-NSC'], results),
                        save=FIGURES_DIR+'compare_gritz_right.png')
    # Gritz mix
    pltt.compare_performance(baseline['none'], select_results(['RGen-S(M)', 'RGen-M(M)', 'RGen-L(M)', 'RGen-L(M)-NSC'], results),
                        save=FIGURES_DIR+'compare_gritz_mix.png')


torch.save(results, SAVE_DIR+'results_recycle.pt')

pltt.table_comparison(baseline['none'], results)
# Compare the best ones
pltt.compare_performance(baseline['none'], select_results(['Ritz-S' , 'RGen-L(L)', 'RGen-L(R)-NSC'], results),
                    save=FIGURES_DIR+'compare_best.png')

#%% How many iterations to get a good hypergradient?
"""How many iterations would we need to do in order to achieve a specific hypergradient relative error? 
Of course we cannot do this in pracice but in the numerical experiment world this is possible to determine by pre-computing
high-accuracy Hessian solutions and their associated hypergradients."""

if do_experiments['true_hg_stop_crit']:
    stop_crit_tol = 1e-2
    stop_crit = 'hg_err'
    results_hg_sc = {}
    results_hg_sc['None']      = calculate_results(name='MINRES HG SC', full_info=False, tol=stop_crit_tol, stop_crit=stop_crit)
    results_hg_sc['Ritz-S']    = calculate_results(name='Ritz-S HG SC', strategy='ritz', size='small', tol=stop_crit_tol, stop_crit=stop_crit)
    results_hg_sc['RGen-L(R)'] = calculate_results(name='RGen-L(R) HG SC', strategy='gritz_right', size='large', tol=stop_crit_tol, stop_crit=stop_crit)
    
    # Save and display results
    torch.save(results_hg_sc, SAVE_DIR+'results_hg_stop.pt')
    pltt.compare_performance(results_hg_sc['None'],  select_results(['Ritz-S','RGen-L(R)'], results_hg_sc),
                             save = FIGURES_DIR+'true_hg_stop_crit.png')
    
    print('\nMethod |   Normal SC  |   HG err SC')
    print(  '-------|--------------|--------------')
    print('Ritz-S | {:} | {:.3f} |  {:} | {:.3f}' .format(baseline['none']['stop_it'].sum(),torch.log10(baseline['none']['hg_rerr'].mean()),results_hg_sc['None']['stop_it'].sum(), torch.log10(results_hg_sc['None']['hg_rerr'].mean())))
    print('Ritz-S | {:} | {:.3f} |  {:} | {:.3f}' .format(results['Ritz-S']['stop_it'].sum(),torch.log10(results['Ritz-S']['hg_rerr'].mean()),results_hg_sc['Ritz-S']['stop_it'].sum(), torch.log10(results_hg_sc['Ritz-S']['hg_rerr'].mean())))
    print('GR-L(R)| {:} | {:.3f} |  {:} | {:.3f}' .format(results['RGen-L(R)']['stop_it'].sum(),torch.log10(results['RGen-L(R)']['hg_rerr'].mean()),results_hg_sc['RGen-L(R)']['stop_it'].sum(), torch.log10(results_hg_sc['RGen-L(R)']['hg_rerr'].mean())))

#%% Inner vs outer recycling strategy
"Inner vs outer methods for recycling. How good is using current linear system rather than waiting "
"until next linear system (one which recycle space will be utilised with) is constructed?"
"For outer recycling, could we get away with only storing only the first few Krylov vectors? "
"How would the number of iterations saved change as we increase the number of Krylov vectors that are stored?"


if do_experiments['inner_vs_outer']:
    results_outer = {'Ritz-S Outer':results['Ritz-S'], 
                     'RGen-L(R) Outer':results['RGen-L(R)'],
                     'RGen-L(R)-NSC Outer':results['RGen-L(R)-NSC']}
    
    results_outer['Ritz-S Inner']        = calculate_results(name='Ritz-S Inner', strategy='ritz', size='small', rec_dim=rec_dim, outer_rec=False)
    results_outer['RGen-L(R) Inner']     = calculate_results(name='RGen-L(R) Inner', strategy='gritz_right', size='large', rec_dim=rec_dim, outer_rec=False)
    results_outer['RGen-L(R)-NSC Inner'] = calculate_results(name='RGen-L(R)-NSC Inner', strategy='gritz_right', size='large', rec_dim=rec_dim, outer_rec=False, stop_crit='hg_err_approx')
    
    #% Save and display results 
    torch.save(results_outer, SAVE_DIR+'results_inner_outer.pt')
    
    pltt.compare_performance(results_hg_sc['None'],  select_results(['Ritz-S Inner','Ritz-S Outer'], results_outer),
                             save = FIGURES_DIR+'inner_outer_ritz.png')
    pltt.compare_performance(results_hg_sc['None'],  select_results(['RGen-L(R) Inner','RGen-L(R) Outer'], results_outer),
                             save = FIGURES_DIR+'inner_outer_gritz.png')
    pltt.compare_performance(results_hg_sc['None'],  select_results(['RGen-L(R)-NSC Inner','RGen-L(R)-NSC Outer'], results_outer),
                             save = FIGURES_DIR+'inner_outer_gritznsc.png')
    
    

#%% Dimension of the recycle space
"What is a good dimension for the recycle space? Since the RMINRES algorithm has computational "
"cost associated with the dimension, there will be a cost tradeoff. As increase size, what happens "
"to performance? Would expect total number of iterates to decrease to zero"

if do_experiments['explore_recycle_space_dimension']:
    recycle_dims = np.array([5*(i)+1 for i in range(20)])
    
    results_dim_raw = {'Ritz-S':[], 'RGen-L(R)':[], 'RGen-L(R)-NSC':[]}
    
    for ind, rec_dim in enumerate(recycle_dims):
        print('\nRec dim: {:3} | #{}/{}\n'.format(recycle_dims[ind], ind+1, len(recycle_dims)))
    
        results_dim_raw['Ritz-S'].append(calculate_results(name='Ritz-S', strategy='ritz', size='small', rec_dim=rec_dim))
        results_dim_raw['RGen-L(R)'].append(calculate_results(name='RGen-L(R)', strategy='gritz_right', size='large', rec_dim=rec_dim))
        results_dim_raw['RGen-L(R)-NSC'].append(calculate_results(name='RGen-L(R)-NSC', strategy='gritz_right', size='large', rec_dim=rec_dim, stop_crit='hg_err_approx'))
    
    #% Process computed quantities to per-recycle space dimension metrics
    forward_op_cost = 0 
    hess_cost = rutils.calc_hess_cost(setup['problem_setup']['regulariser']['filter_num'] , setup['problem_setup']['regulariser']['filter_shape'], setup['problem_setup']['n']**2, forward_op_cost=forward_op_cost)
    
    results_dim = {}
    for name, item in results_dim_raw.items():
        stop_its = torch.tensor([result['stop_it'].sum() for result in item])
        results_dim[name] = {'stop_its':stop_its ,
                             'flops':torch.empty(len(item)),
                             'hg_rerr_mean':torch.empty(len(item)),
                             'hg_rerr_max':torch.empty(len(item))} 
        for ind, result in enumerate(item):
            stop_it = result['stop_it']
            s = np.array([u.shape[1] for u in result['U_space']])
            flops = (rutils.calc_RMINRES_cost(stop_it[1:], hess_cost, setup['problem_setup']['n']**2, s).sum() + rutils.calc_MINRES_cost(stop_it[0], hess_cost, setup['problem_setup']['n']**2))
            results_dim[name]['flops'][ind] = flops
            results_dim[name]['hg_rerr_mean'][ind] = result['hg_rerr'].mean()
            results_dim[name]['hg_rerr_max'][ind] = result['hg_rerr'].max()
    
    results_dim_baseline = {
        'stop_its': baseline['none']['stop_it'].sum() * torch.ones(len(item)) 
        ,'hg_rerr_mean':baseline['none']['hg_rerr'].mean()*torch.ones(len(item))
        ,'hg_rerr_max':baseline['none']['hg_rerr'].max()*torch.ones(len(item))
        ,'flops':rutils.calc_MINRES_cost(baseline['none']['stop_it'], hess_cost , setup['problem_setup']['n']**2).sum() * torch.ones(len(item)) 
        }
    
    # Plot results for dimension of recycle space experiment
    torch.save(results_dim_raw, SAVE_DIR+'results_dim_raw.pt')
    torch.save(results_dim, SAVE_DIR+'results_dim.pt')
    torch.save(results_dim_baseline , SAVE_DIR+'results_dim_baseline.pt')
    torch.save(recycle_dims , SAVE_DIR+'results_recycle_dims.pt')
    
    pltt.compare_performance(results_dim_baseline, results_dim, xvalues=recycle_dims,
                        quantity_keys = ['stop_its', 'flops', 'log10_hg_rerr_mean','log10_hg_rerr_max'],
                        ylabels = ['Total (R)MINRES its', '(R)MINRES FLOPs', 'log10 Mean HG RERR', 'log10 Max HG RERR'],
                        save=FIGURES_DIR+'results_dim.png')

#%%#%%Approximation of the hypergradient error
""" How good is our approximation of the hypergradient error using the projected GSVD?
We approximate \| J^{i} H^{i}^{-1} r_k \| with \| J^{i}W(W^T H^{i} W)^{-1} WT  r_k \| 
where W=W^{i}. We compare these quantities and, to determine how much is the error introduced
by doing a projection vs doing a projection onto a subspace whose solution space is technically
not that associated to H^i (but rather H^{i-1}), for W=W^{i+1}.
"""
tmp_dict = select_results(['Ritz-S','RGen-L(R)','RGen-L(R)-NSC'], results)

#%% See how application of the approx JHinv actually compare in evaluation of the norm

if do_experiments['hg_approx_accuracy']:
    # method  = 'Ritz-S'
    # method = 'RGen-L(R)'
    method = 'RGen-L(R)-NSC'
    
    all_W_spaces = tmp_dict[method]['W_space']
    all_info = tmp_dict[method]['info']
    
    results_hg_approx = {'true':[], 'approx':[], 'cheat':[]}
    for hess_ind in tqdm(range(len(all_W_spaces)-1), desc='HG approx accuracy'):
        JHinv_r, JHinv_approx_r, JHinv_cheat_r = rutils.hg_err_approx(hessian_maker, hess_ind, all_W_spaces, all_info)
        results_hg_approx['true'].append(JHinv_r)
        results_hg_approx['approx'].append(JHinv_approx_r)
        results_hg_approx['cheat'].append(JHinv_cheat_r)
    
    torch.save(results_hg_approx, SAVE_DIR+'results_hg_approx.pt')
    
    #% Plot a few specific RMINRES calls
    considered_systems = [14,39,59]
    considered_systems = [3]
    for ind in considered_systems:
        pltt.plot_comparison_of_hg_err_approx(results_hg_approx['true'][ind], 
                                         results_hg_approx['approx'][ind], 
                                         results_hg_approx['cheat'][ind],
                                         save=FIGURES_DIR+f'hg_approx_system{ind}.png')
    
    pltt.plot_comparison_of_hg_err_approx([res[-1] for res in results_hg_approx['true']], 
                                     [res[-1] for res in results_hg_approx['approx']], 
                                     [res[-1] for res in results_hg_approx['cheat']],
                                     save=FIGURES_DIR+'hg_approx_all_systems.png')