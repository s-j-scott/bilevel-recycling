# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 07:42:02 2024

@author: Sebastian J. Scott

Re-solve sequence of Hessian systems using various recycling strategies.

"""

import torch
import numpy as np

from src.utils import set_device, select_results
from src import plotting as pltt

torch.no_grad()

#%% Load data

LOAD_DIR = './results/inpainting/recycle_data/'
BILEVEL_DIR = './results/inpainting/bilevel_data/'
FIGURES_DIR = './results/inpainting/figures/'
    
torch.set_default_dtype(torch.float64)
device = set_device(verbose=True)

state = torch.load(LOAD_DIR+'none_results', weights_only=False)
stop_it_none, ul_cost_none, cost_none, all_hg_err_none = state['stop_it_none'], state['ul_cost_none'], state['cost_none'], state['all_hg_err_none']

baseline={'none':{'stop_it':state['stop_it_none'],
                    'hg_rerr':state['all_hg_err_none'],
                    'ul_cost':state['ul_cost_none'],
                                  }}
quantity_keys=['stop_it', 'cumsum_stop_it','log10_hg_rerr', 'ul_cost']
ylabels=['Number of its', 'Cumulative #its', 'log10 HG rel. error', 'Upper level cost']

#%% Motivation of 

diff_info = torch.load(LOAD_DIR+'rel_difs')

pltt.plot_data(torch.log10(diff_info['dif_hess']/diff_info['norm_hess']),
               ylabel= 'log10 $H^{(i)}$ relative diff', 
               xlabel='Linear system number')    
pltt.plot_data(torch.log10(diff_info['dif_rhs']/diff_info['norm_rhs']),
               ylabel='log10 $g^{(i)}$ relative diff', 
               xlabel='Linear system number')    
pltt.plot_data(torch.log10(diff_info['dif_w']/diff_info['norm_w']), 
               ylabel='log10 $w^{(i)}$ relative diff',
               xlabel='Linear system number')    
pltt.plot_data(torch.log10(diff_info['dif_recon']/diff_info['norm_recon']), 
               ylabel=r'log10 $x(\theta^{i+1})$ relative diff', 
               xlabel='Linear system number')

#%% Timings plot
import pandas
bl_info = pandas.read_csv(BILEVEL_DIR+'OUTPUT_bl_info.csv')
pltt.plot_timings(
    {'t_total':bl_info['ul_itn_times'], 't_hess':bl_info['t_hess'],'t_ll':bl_info['t_total_lls']},
    save=FIGURES_DIR+'timings')

#%% Eig vs eigs
def dictify(legacy):
    legacy_dic = {}
    for col in legacy:
        legacy_dic[col[0]] = {'hg_rerr':col[4] , 'stop_it':col[1], 'ul_cost':col[2]}
    return legacy_dic

def legacy_get(name):
    data = torch.load(LOAD_DIR+name, weights_only=False)
    data_dic = dictify(data)
    return data_dic

def easy_plot(name):
    dic = legacy_get(name)
    pltt.compare_performance(baseline['none'], dic, quantity_keys=quantity_keys, ylabels=ylabels, 
                             save = FIGURES_DIR+name+'.png')

easy_plot('eig_v_ritz')

#%% Cold vs warm
cold_v_warm = legacy_get('cold_v_warm')
cold_v_warm['CG warm'] = cold_v_warm.pop('CG')
pltt.compare_performance(baseline['none'], cold_v_warm, baseline_label='MINRES', 
                         quantity_keys=quantity_keys, ylabels=ylabels,
                         save=FIGURES_DIR+'cold_v_warm.png')
#%% Small v large

results = legacy_get('all_plot_details')

# Ritz 
pltt.compare_performance(baseline['none'], select_results(['Ritz-S', 'Ritz-M', 'Ritz-L'], results),
                    save=FIGURES_DIR+'compare_ritz.png', quantity_keys=quantity_keys, ylabels=ylabels)
# HRitz 
pltt.compare_performance(baseline['none'], select_results(['HRitz-S', 'HRitz-M', 'HRitz-L'], results),
                    save=FIGURES_DIR+'compare_hritz.png', quantity_keys=quantity_keys, ylabels=ylabels)
# Gritz left
pltt.compare_performance(baseline['none'], select_results(['RGen-S(L)', 'RGen-M(L)', 'RGen-L(L)', 'RGen-L(L)-NSC'], results),
                    save=FIGURES_DIR+'compare_gritz_left.png', quantity_keys=quantity_keys, ylabels=ylabels)
# Gritz right
pltt.compare_performance(baseline['none'], select_results(['RGen-S(R)', 'RGen-M(R)', 'RGen-L(R)', 'RGen-L(R)-NSC'], results),
                    save=FIGURES_DIR+'compare_gritz_right.png', quantity_keys=quantity_keys, ylabels=ylabels)
# Gritz mix
pltt.compare_performance(baseline['none'], select_results(['RGen-S(M)', 'RGen-M(M)', 'RGen-L(M)', 'RGen-L(M)-NSC'], results),
                    save=FIGURES_DIR+'compare_gritz_mix.png', quantity_keys=quantity_keys, ylabels=ylabels)

# Best
pltt.compare_performance(baseline['none'], select_results(['Ritz-S' , 'RGen-L(R)', 'RGen-L(R)-NSC'], results),
                    save=FIGURES_DIR+'compare_best.png', quantity_keys=quantity_keys, ylabels=ylabels)
#%% Iterations to reach hypergradient
easy_plot('all_plot_details_hg_stop')

#%%  Inner vs outer

results_io = legacy_get('inner_v_outer')

pltt.compare_performance(baseline['none'], select_results(['Ritz-S Outer', 'Ritz-S Inner'], results_io),
                    save=FIGURES_DIR+'io_compare_ritz.png', quantity_keys=quantity_keys, ylabels=ylabels)

pltt.compare_performance(baseline['none'], select_results(['RGen-L(R) Outer', 'RGen-L(R) Inner'], results_io),
                    save=FIGURES_DIR+'io_compare_gritz.png', quantity_keys=quantity_keys, ylabels=ylabels)

pltt.compare_performance(baseline['none'], select_results(['RGen-L(R)-NSC Outer', 'RGen-L(R)-NSC Inner'], results_io),
                    save=FIGURES_DIR+'io_compare_gritznew.png', quantity_keys=quantity_keys, ylabels=ylabels)

#%% HG error approx comapre

for ind in [14,39,59,94]:
    state_hg_tmp = torch.load(LOAD_DIR+'hg_err_comp_system'+str(ind), weights_only=False)
    pltt.plot_comparison_of_hg_err_approx(state_hg_tmp['true'], 
                                     state_hg_tmp['approx'], 
                                     state_hg_tmp['cheat'],
                                    label1=f'$W^{({ind})}$ approx',
                                    label2=f'$W^{({ind+1})}$ approx',
                                    title=f'Hessian system {ind}',
                                     save=FIGURES_DIR+f'hg_approx_all_systems{ind}.png')

state_hg = torch.load(LOAD_DIR+'all_hg_err_comp', weights_only=False)
pltt.plot_comparison_of_hg_err_approx(state_hg['true'], 
                                 state_hg['approx'], 
                                 state_hg['cheat'],
                                 save=FIGURES_DIR+'hg_approx_all_systems.png')


#%% Dimension of the recycle space

state = torch.load( LOAD_DIR+'all_dims_results', weights_only=False)


results_dim_baseline = {
        'stop_its': baseline['none']['stop_it'].sum() * torch.ones(len(state['rec_dims'])) 
        ,'hg_rerr_mean':baseline['none']['hg_rerr'].mean()*torch.ones(len(state['rec_dims']))
        ,'hg_rerr_max':baseline['none']['hg_rerr'].max()*torch.ones(len(state['rec_dims']))
        ,'flops': state['c_none'].sum() * torch.ones(len(state['rec_dims']))
        }

results_dim = {
    'Ritz-S': {'flops':state['c_ritz'].sum(axis=1), 
               'hg_rerr_mean':torch.stack(state['hg_ritz']).mean(axis=1),
               'hg_rerr_max':torch.stack(state['hg_ritz']).max(axis=1)[0],
               'stop_its':torch.tensor(np.stack(state['stop_ritz']).sum(axis=1))},
    
    'RGen-L(R)': {'flops':state['c_gritz'].sum(axis=1), 
                  'hg_rerr_mean':torch.stack(state['hg_gritz']).mean(axis=1),
                  'hg_rerr_max':torch.stack(state['hg_gritz']).max(axis=1)[0], 
                  'stop_its':torch.tensor(np.stack(state['stop_gritz']).sum(axis=1))},
    
    'RGen-L(R)-NSC': {'flops':state['c_gritz2'].sum(axis=1),
                      'hg_rerr_max':torch.stack(state['hg_gritz2']).max(axis=1)[0], 
                      'hg_rerr_mean':torch.stack(state['hg_gritz2']).mean(axis=1), 
                      'stop_its':torch.tensor(np.stack(state['stop_gritz2']).sum(axis=1))},
    }

pltt.compare_performance(results_dim_baseline, results_dim, xvalues=state['rec_dims'],
                    quantity_keys = ['stop_its', 'flops', 'log10_hg_rerr_mean','log10_hg_rerr_max'],
                    xlabel='Recycle space dimension',
                    ylabels = ['Total (R)MINRES its', '(R)MINRES FLOPs', 'log10 Mean HG RERR', 'log10 Max HG RERR'],
                    save=FIGURES_DIR+'results_dim.png')

