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

from src.utils import set_device, create_title_str, load_data, set_seed, select_results
from src.create_cost_functions import create_problem

from src import plotting as pltt
import src.recycle_utils as rutils

import json 

import src.utils as utils
from src.load_dataset import create_dataset, create_dataloader

torch.no_grad()



#%% Load data

LOAD_DIR    = './results/gaussian_blur/bilevel_data/'
SAVE_DIR    = './results/gaussian_blur/recycle_data/'
FIGURES_DIR = './results/gaussian_blur/figures/'


rec_dim = 30 # Dimension of recycle space to be considered throughout

#%% Automated selection of quantities

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)
device = set_device(verbose=False)

with open(LOAD_DIR+'setup.json') as f:
    setup = json.load(f)
setup['solver_optns']['hess_sys']['max_its'] = 4096


all_params = torch.load(LOAD_DIR+'history_params.pt')
all_logs = torch.load(LOAD_DIR+'all_logs.pt')

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
from src.create_cost_functions import create_cost_functions
for batch in test_dataloader: 
    batch = utils.to_device(batch, device)
    y_test = batch['forward_image'][0][0]
    xexact_test = batch['image'][0][0]
    A_test = batch['forward_op'][0]
    break

ul_fun, ll_fun = create_cost_functions(setup['problem_setup']['regulariser'],device=device)
ll_solver = utils.get_lower_level_solver(setup, ll_fun, xexact_test.shape, device)

recon_test = ll_solver(all_params[-1], A_test, y_test)
psnr_test = utils.psnr(xexact_test, recon_test)
recon0_test = ll_solver(all_params[0], A_test, y_test)
psnr0_test = utils.psnr(xexact_test, recon0_test)

ps = 10**torch.linspace(-3,1,10, device=device)
recon_tv, ul_tv, ul_vals_tv = rutils.calculate_best_TV(setup,xexact_test, A_test, y_test, ps, verbose=True)
psnrtv_test = utils.psnr(xexact_test, recon_tv)
psnry_test = utils.psnr(xexact_test, y_test)

pltt.display_recons(xexact_test,[y_test, recon_test, recon_tv],
               subtitles=[f'Measurement\n(PSNR: {psnry_test:.2f})', 
                          f'FoE Recon\n(PSNR: {psnr_test:.2f})', 
                          # f'DCT Recon\n(PSNR: {psnr0_test:.2f})'
                          f'TV Recon\n(PSNR: {psnrtv_test:.2f})'
                          ],
               save=FIGURES_DIR+'plot_test_reconstructions.png')


#%% Iterate over train dataset
for i, batch in enumerate(train_dataloader):
    tqdm.write(f'### Item {i} / {len(train_dataloader)}')
    # batch = utils.to_device(batch, device)
    xexact = batch['image'][0][0]
    y = batch['forward_image'][0][0]
    A = batch['forward_op'][0]
    idx = batch['idx'][0]
    A = A.to(device)

    # Consider specific Hessian system
    state = all_logs[idx]
    
    seqn_recons = [ x[0] for x in state['recon'][:setup['solver_optns']['ul_solve']['max_its']]]
    seqn_params = [all_params[i] for i in state['step'][:setup['solver_optns']['ul_solve']['max_its']]]
    
    # Function that returns J_op, Hessian, g, and recon for a given parameter/Hessian system index
    hessian_maker = rutils.hessian_system_generator(setup, xexact, A, y, seqn_params, seqn_recons, device)
    
    #% Compute high and normal accuracy solutions
    baseline = {}
    baseline['high'] =  rutils.compute_high_accuracy(hessian_maker)
    
    calculate_results = rutils.prepare_calculator(hessian_maker, baseline['high']['hgs'], include_info=False)
    baseline['none'] = calculate_results(name='MINRES warm', full_info=False)
    
    results = {}
    results['Ritz-S']    = calculate_results(name='Ritz-S',    strategy='ritz',        size='small', rec_dim=rec_dim, full_info=True)
    results['RGen-L(R)'] = calculate_results(name='RGen-L(R)', strategy='gritz_right', size='large', rec_dim=rec_dim, full_info=True)
    results['RGen-L(R)-NSC'] = calculate_results(name='RGen-L(R)-NSC', strategy='gritz_right', size='large', rec_dim=rec_dim, full_info=True, stop_crit='hg_err_approx')
    results['RGen-L(L)'] = calculate_results(name='RGen-L(L)', strategy='gritz_left', size='large', rec_dim=rec_dim, full_info=True)
    results['RGen-L(L)-NSC'] = calculate_results(name='RGen-L(L)-NSC', strategy='gritz_left', size='large', rec_dim=rec_dim, full_info=True, stop_crit='hg_err_approx')
    
    torch.save(baseline,SAVE_DIR+f'{idx}results_baseline.pt')
    torch.save(results, SAVE_DIR+f'{idx}results_recycle.pt')
    pltt.table_comparison(baseline['none'], results)
    
    