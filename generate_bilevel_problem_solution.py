# -*- coding: utf-8 -*-
"""

Learn optimal filters of a fields of experts regularizer for an MNSIT inpainting
    problem via bilevel learning.
    
  
@author: Sebastian J. Scott

"""

import torch

from bilevel_torch import bilevel_gradient_descent, populate_options
from utils import set_device, save_bl_info,  save_dataset
from create_cost_functions import create_forward_op, create_cost_functions, create_dataset

import json
import os
import matplotlib.pyplot as plt
import numpy as np

torch.no_grad()

#%% Create and save details of the problem

# Specify inverse problem and cost function
problem_setup = {
            'forward_op': 'inpainting', 'sigma':None, 'mask':.7,
            'n':28, # For MNIST must be 28
          'noiselevel':0.3,
          'data_num':1,
            'ground_truth':'MNIST',
        
            'regulariser':{
                'name':'FieldOfExperts',
                'filter_shape':5,
                  'filter_num':3,
                    'expert':'l2',
              'gamma':.1,
              'eps':1e-6}
          }

# # Specify options for numerical solver
solver_optns = {'ul_solve':
                  {'max_its': 1000,
                  'tol': 1e-3,
                    'solver':'GD'
                  },
          'll_solve':
                  {'verbose':False, 
                  'warm_start':True, 
                  'max_its':5000,
                    'tol': 1e-3,
                  'solver':'L-BFGS'
                  ,'store_iters':True
                  ,'num_store':10
                  },
        'hess_sys':
                {'warm_start':True,
                'tol': 1e-2,
                'max_its':5000,
                'solver':'MINRES',
                'recycle_strategy':None
                    }}

solver_optns = populate_options(solver_optns)
setup = {'problem_setup':problem_setup, 'solver_optns':solver_optns}



# # Directory to save results in 
RESULTS_DIR = './data/' + problem_setup['forward_op'] +'/'
# RESULTS_DIR = './results/thesis/placeholder/'

n = setup['problem_setup']['n']
# Plotting parameters
fs = 20 # Font size
lw = 2.5# Line width

#%% Generate data for the inverse problem
torch.set_default_dtype(torch.float64)
torch.manual_seed(0)
device = set_device(verbose=True)

A = create_forward_op(problem_setup['forward_op'],sigma=problem_setup['sigma'], device=device, tensor_size=torch.Size([n,n]),mask=problem_setup['mask'])
xexacts, ys =  create_dataset(n=n, ground_truth=problem_setup['ground_truth'], A=A, data_num=problem_setup['data_num'],
                                      noiselevel=problem_setup['noiselevel'],device=device)



#%% Solve the bilevel learning problem without recycling
ul_fun, ll_fun = create_cost_functions(A,xexacts, optns=setup['problem_setup']['regulariser'],device=device)
bl_soln, bl_recons, bl_info = bilevel_gradient_descent(ul_fun, ll_fun, ys, optns=setup['solver_optns'])


#%% Save dataset
all_params, all_recons, df_bl_info = save_bl_info(bl_recons, bl_info, path=RESULTS_DIR)
xexacts_np, ys_np = save_dataset(xexacts,ys, path=RESULTS_DIR)

# # For Krylov chapter, also save solutions of Hessian systems
all_hess_solns = torch.stack([ bl_info['hess_infos'][i][0]['xs'][-1] for i in range(len(bl_info['hess_infos']))])
torch.save(all_hess_solns, RESULTS_DIR+'/OUTPUT_hess_solns.pt')
if not os.path.exists(RESULTS_DIR):
  os.mkdir(RESULTS_DIR)
with open(RESULTS_DIR+"OUTPUT_setup.json", 'w') as f: 
    json.dump(setup, f, indent=6)

from utils import plot_info

plot_info(df_bl_info, all_params)

def plot_timings(df_bl_info, title=None, save=None, fontsize=15):

    fig,ax = plt.subplot_mosaic([
        ['t_total', 't_total_lls', 't_hess', 't_misc'  ],
        ['overall_ratio','overall_ratio','overall_ratio', 'overall_ratio']],
        figsize=(10,6))
    
    col_ll = 'tab:blue'
    col_hess = 'tab:orange'
    col_misc = 'tab:green'
    alpha = 0.3 # opacity of filled blocks
    
    t_total = df_bl_info['ul_itn_times']
    
    t_total_lls = df_bl_info['t_total_lls']
    t_hess = df_bl_info['t_hess']
    t_misc = t_total-t_total_lls-t_hess
    s_total = sum(t_total)
    s_hess = sum(t_hess)
    s_misc = sum(t_misc)
    s_total_lls = sum(t_total_lls)
    
    t_total_lls_rat = ( t_total_lls)/t_total
    t_hess_rat = ( t_hess )/t_total
    t_misc_rat = ( t_misc )/t_total
    ################################################################
    ax["t_total"].plot((t_total) ,'k')
    ax["t_total"].set_title('Total iteration time\n{:.2f}s  ({:.2f}%)'.format(s_total, 100), fontsize=fontsize)
    
    ax["t_total_lls"].plot((t_total_lls), color=col_ll)
    ax["t_total_lls"].set_title('Lower level solve time\n{:.2f}s  ({:.2f}%)'.format(s_total_lls, s_total_lls/s_total*100), fontsize=fontsize)
    
    ax["t_hess"].plot((t_hess), color=col_hess)
    ax["t_hess"].set_title('Hessian solve time\n{:.2f}s  ({:.2f}%)'.format(s_hess, s_hess/s_total*100), fontsize=fontsize)

    
    ax["t_misc"].plot((t_misc), color=col_misc)
    ax["t_misc"].set_title('Other\n{:.2f}s  ({:.2f}%)'.format(s_misc, s_misc/s_total*100), fontsize=fontsize)
  
    x = np.linspace(0,len(t_misc)-1, len(t_misc))
    
    ax["overall_ratio"].plot(t_total_lls_rat, color=col_ll, label='Lower level solve' )
    ax["overall_ratio"].fill_between(x=x,y1=t_total_lls_rat, color=col_ll, alpha=alpha)
    ax["overall_ratio"].plot(t_total_lls_rat+t_hess_rat, color=col_hess, label='Hessian solve' )
    ax["overall_ratio"].fill_between(x=x,y1=t_total_lls_rat+t_hess_rat, y2=t_total_lls_rat, color=col_hess, alpha=alpha)
    ax["overall_ratio"].plot(t_total_lls_rat+t_hess_rat + t_misc_rat , color=col_misc, label='Misc')
    ax["overall_ratio"].fill_between(x=x,y1= t_total_lls_rat+t_hess_rat + t_misc_rat, y2=t_total_lls_rat+t_hess_rat, color=col_misc, alpha=alpha)
    ax["overall_ratio"].set_ylabel('Ratio of timings', fontsize=fontsize)
    ax["overall_ratio"].legend(fontsize=fontsize*.7)
    ax["overall_ratio"].set_ylim(-.05,1.05)

    for a in ax:
        ax[a].set_xlabel('Linear system number', fontsize=fontsize*.8)

    plt.suptitle(title)
    plt.tight_layout()
    if isinstance(save,str): plt.savefig(save)

plot_timings(df_bl_info, save=RESULTS_DIR+'/OUTPUT_plot_timings')
