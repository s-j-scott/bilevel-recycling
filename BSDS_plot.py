# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 07:42:02 2024

@author: Sebastian J. Scott

Accumulate results across multiple Hessian system solve runs

"""

import torch
from tqdm import tqdm

from src.utils import set_device, set_seed, select_results
from src import plotting as pltt
import src.utils as utils

import json

                
torch.no_grad()

#%% Load data

BILEVEL_DIR = './results/gaussian_blur/bilevel_data/'
LOAD_DIR    = './results/gaussian_blur/recycle_data/'
FIGURES_DIR = './results/gaussian_blur/figures/'
    
torch.set_default_dtype(torch.float64)
device = set_device(verbose=True)

#%% Declare useful variables

torch.set_default_dtype(torch.float64)
set_seed(0)
device = set_device(verbose=False)

with open(BILEVEL_DIR+'setup.json') as f:
    setup = json.load(f)

all_params = torch.load(BILEVEL_DIR+'history_params.pt')
all_logs = torch.load(BILEVEL_DIR+'all_logs.pt')

#%% Summary statistics

summary = utils.sum_state_keys(setup, all_logs)
pltt.plot_timings(summary)

pltt.display_filters( all_params[0], square=False,
              save=FIGURES_DIR+'plot_filters_init.png')

pltt.display_filters( all_params[-1], square=False,
              save=FIGURES_DIR+'plot_filters_learned.png')

#%% Single training data sequence


def get_dict(data_id):
    baseline = torch.load(LOAD_DIR+f'{data_id}results_baseline.pt', weights_only=False)
    results = torch.load(LOAD_DIR+f'{data_id}results_recycle.pt',   weights_only=False)
    return baseline, results

baseline, results = get_dict(26)
pltt.table_comparison(baseline['none'], results)
pltt.compare_performance(baseline['none'], select_results(['Ritz-S' , 'RGen-L(R)', 'RGen-L(R)-NSC'], results),
                    save=FIGURES_DIR+'compare_best.png')

#%% Accumulated results
accum_baseline = {}
accum_results = {}

def add_to_dic(accum, results, data_num):
    for method, result in results.items():
        if method == 'high':
            pass
        else:            
            try:
                accum[method]['stop_it'] += result['stop_it']
                accum[method]['hg_rerr'] += result['hg_rerr'] / data_num
                
            except:
                accum[method] = {'stop_it':result['stop_it'] , 'hg_rerr':result['hg_rerr']}

data_num = setup['problem_setup']['num_crops_per_image'] * setup['problem_setup']['data_num']
for i in tqdm(range(data_num), desc='Accumulating results'):
    try:
        baseline_, results_ = get_dict(i)
        add_to_dic(accum_baseline, baseline_, data_num)
        add_to_dic(accum_results, results_, data_num)
    except:
        pass

pltt.table_comparison(accum_baseline['none'],accum_results)
pltt.compare_performance(accum_baseline['none'], select_results(['Ritz-S' , 'RGen-L(R)', 'RGen-L(R)-NSC'], accum_results),
                    ylabels=['Number of its', 'Cumulative No. of its', 'log10 mean HG rel. err.'],
                    save=FIGURES_DIR+'compare_best_accum.png')