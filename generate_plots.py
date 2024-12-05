# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 21:24:44 2024

@author: Sebastian J. Scott

Create plots of the recycling results.

"""

import torch
import numpy as np
import matplotlib.pyplot as plt

from thesis_recycle_utils import plot_data

#%%
LOAD_DIR = './results/'



state = torch.load(LOAD_DIR+'none_results')
stop_it_none, ul_cost_none, cost_none, all_hg_err_none = state['stop_it_none'], state['ul_cost_none'], state['cost_none'], state['all_hg_err_none']

#%%

def select_results(to_select, all_plot_details):
    if to_select is None:
        return all_plot_details
    else:
        return [all_plot_details[i] for i in range(len(all_plot_details)) if all_plot_details[i][0] in to_select]


def pretty_plot(val_to_plot, plot_settings, label, fig=None, ax=None, fs=23, lw=2.5):
    if fig is None:
        fig, ax = plt.subplots()
        
    if len(plot_settings) == 2: # Only colour and linestyle specified
        linestyle, color = plot_settings
        marker, markevery = None, 0.1        
    else:
        linestyle, color, marker, markevery = plot_settings

    ax.plot(val_to_plot , label=label, linewidth=lw, linestyle=linestyle, marker=marker, markevery=markevery, color=color, markersize=10)
    plt.xticks(fontsize=fs*.8)
    plt.yticks(fontsize=fs*.8)
    plt.legend(fontsize=fs*.7)


def make_plot(default_result, recycle_results, ind_to_use=1, ylabel='', xlabel='Linear system number', 
              default_name='None', fs=25, lw=3, fig=None, ax=None):
    if fig is None:
        fig, ax = plt.subplots(figsize=(5.5,5))
    ax.set_xlabel(xlabel, fontsize=fs)
    ax.set_ylabel(ylabel, fontsize=fs)
    
    pretty_plot(default_result, ((0,(1,1)), 'k', '*', 0.15), label=default_name, fig=fig,ax=ax, fs=fs, lw=lw)
    for ind, info in enumerate(recycle_results):
        
        if ind_to_use == 3:
            val_to_plot = np.cumsum(info[1])
        elif ind_to_use == 4:
            val_to_plot = np.log10(info[4])
        else:
            val_to_plot = info[ind_to_use]
        
        pretty_plot(val_to_plot, plot_settings=info[5:], label=info[0], fig=fig,ax=ax, fs=fs, lw=lw)
    plt.tight_layout()


def display_recycle_compare(quantity, methods_to_plot, all_plot_details, fs=23, lw=3, default_name='None'):
    if quantity == 'its':
        ind_to_use = 1 
        ylabel, default_result = 'Number of its', stop_it_none
    elif quantity == 'cits':
        ind_to_use = 3
        ylabel, default_result = 'Cumulative number of its',  np.cumsum(stop_it_none)
    elif quantity == 'hg':
        ind_to_use = 4
        ylabel, default_result = 'log10 HG relative error',  np.log10(all_hg_err_none)
    elif quantity == 'ul': 
        ind_to_use = 2
        ylabel, default_result = 'Upper level cost',  ul_cost_none
    
    vals_to_plot = select_results(methods_to_plot, all_plot_details)
    make_plot(default_result,  vals_to_plot, ind_to_use=ind_to_use, ylabel=ylabel, fs=fs, lw=lw, default_name=default_name )



# order: name, stop, ul, flops, hg_err, [plot options]
def make_all_rec_comp_plots(all_plot_details, methods_to_plot=None, default_name='None'):
    display_recycle_compare('its',methods_to_plot, all_plot_details, default_name=default_name)
    display_recycle_compare('cits',methods_to_plot, all_plot_details, default_name=default_name)
    display_recycle_compare('hg',methods_to_plot, all_plot_details, default_name=default_name)
    display_recycle_compare('ul',methods_to_plot, all_plot_details, default_name=default_name)
  
#%% Motivation of recycling
diff_info = torch.load(LOAD_DIR+'rel_difs')
plot_data(torch.log10(diff_info['dif_hess']/diff_info['norm_hess']),ylabel= 'log10 $H^{(i)}$ rel. difference', xlabel='Linear system number')    
plot_data(torch.log10(diff_info['dif_rhs']/diff_info['norm_rhs']),ylabel='log10 $g^{(i)}$ rel. difference', xlabel='Linear system number')    
plot_data(torch.log10(diff_info['dif_w']/diff_info['norm_w']), ylabel='log10 $w^{(i)}$ rel. difference', xlabel='Linear system number')    
plot_data(torch.log10(diff_info['dif_recon']/diff_info['norm_recon']), ylabel=r'log10 $x(\theta^{i+1})$ rel. difference', xlabel='Linear system number')      
plot_data(diff_info['hess_cond'], ylabel='Hessian condition number', xlabel='Linear system number')    


  
#%% Ritz vs Eigs 
 
eig_v_ritz_plot = torch.load(LOAD_DIR+'eig_v_ritz')
make_all_rec_comp_plots(eig_v_ritz_plot)


#%% Cold vs warms

cold_v_warm_plot = torch.load(LOAD_DIR+'cold_v_warm')
make_all_rec_comp_plots(cold_v_warm_plot, default_name='MINRES')

#%% Small vs Large

all_plot_details = torch.load(LOAD_DIR+'all_plot_details')
make_all_rec_comp_plots(all_plot_details, ['RGen-S(L)', 'RGen-M(L)', 'RGen-L(L)', 'RGen-L(L)-NSC'])
make_all_rec_comp_plots(all_plot_details,['RGen-S(R)', 'RGen-M(R)', 'RGen-L(R)', 'RGen-L(R)-NSC'])
make_all_rec_comp_plots(all_plot_details,['RGen-S(M)', 'RGen-M(M)', 'RGen-L(M)', 'RGen-L(M)-NSC'])
make_all_rec_comp_plots(all_plot_details,['Ritz-S' , 'Ritz-M', 'Ritz-L'])
make_all_rec_comp_plots(all_plot_details,['HRitz-S' , 'HRitz-M', 'HRitz-L'])
make_all_rec_comp_plots(all_plot_details,['Ritz-S' , 'RGen-L(R)', 'RGen-L(R)-NSC'])


#%% Iterations to reach hypergradient


all_plot_details_hg_stop = torch.load(LOAD_DIR+'all_plot_details_hg_stop')
state_stop = torch.load(LOAD_DIR+'none_results')
stop_it_none_stop, ul_cost_none_stop, cost_none_stop, all_hg_err_none_stop = state_stop['stop_it_none'], state_stop['ul_cost_none'], state_stop['cost_none'], state_stop['all_hg_err_none']
make_all_rec_comp_plots(all_plot_details_hg_stop,['Ritz-S' , 'RGen-L(R)'])

#%% Inner vs outer
plot_details_inner_v_outer = torch.load(LOAD_DIR+'inner_v_outer')
make_all_rec_comp_plots(plot_details_inner_v_outer,['Ritz-S Outer', 'Ritz-S Inner'])
make_all_rec_comp_plots(plot_details_inner_v_outer,['RGen-L(R) Outer', 'RGen-L(R) Inner'])
make_all_rec_comp_plots(plot_details_inner_v_outer,['RGen-L(R)-NSC Outer', 'RGen-L(R)-NSC Inner'])


#%% Hg error approx compare

def plot_comparison_of_hg_err_approx(actual, approx, cheat_approx, label1='Projected', nolog=False,label2='Projected cheat', title='', xlabel='RMINRES iteration', ylabel='', fontsize=23):
    plt.figure(figsize=(5.5,5))
    if nolog:
        plt1, plt2, plt3 = actual, approx, cheat_approx
    else:
        plt1, plt2, plt3 = np.log10(actual), np.log10(approx), np.log10(cheat_approx)
    
    xplt = np.arange(len(plt1))+1
    plt.plot(xplt,plt1, ':', label='True', linewidth=3, color='black')
    plt.plot(xplt,plt2, label=label1, linewidth=3, color='tab:olive', marker='d', markevery=.25)
    plt.plot(xplt,plt3, ':', label=label2, linewidth=3, color='tab:olive', marker='d', markevery=.25)
    plt.xticks(fontsize=fontsize*.8)
    plt.yticks(fontsize=fontsize*.8)
    plt.legend(fontsize=fontsize*.7)
    plt.xlabel(xlabel,fontsize=fontsize)
    plt.ylabel(ylabel,fontsize=fontsize)
    plt.title(title,fontsize=fontsize)
    plt.tight_layout()


def plot_JHinv_comp(hess_ind):
    state_hg_err = torch.load(LOAD_DIR+'hg_err_comp_system'+str(hess_ind))
    JHinv_r_gritz_true,JHinv_approx_r_gritz2,JHinv_approx_r_gritz_cheat = state_hg_err['true'],state_hg_err['approx'],state_hg_err['cheat']
    
    plot_comparison_of_hg_err_approx(JHinv_r_gritz_true,JHinv_approx_r_gritz2,JHinv_approx_r_gritz_cheat, 
                                     label1='$W^{('+str(hess_ind)+')}$ approx',  label2='$W^{('+str(hess_ind+1)+')}$ approx',
                                     title='Hessian system '+str(hess_ind), ylabel='log10 HG error')
plot_JHinv_comp(14)
plot_JHinv_comp(39)
plot_JHinv_comp(59)
plot_JHinv_comp(94)

#%%
state_all_hg_err = torch.load(LOAD_DIR+'all_hg_err_comp')

all_JHinv_r_gritz_true,all_JHinv_approx_r_gritz2,all_JHinv_approx_r_gritz_cheat = state_all_hg_err['true'],state_all_hg_err['approx'],state_all_hg_err['cheat']
all_JHinv_norm_true, all_JHinv_norm_approx, all_JHinv_norm_cheat =  state_all_hg_err['truenorm'],state_all_hg_err['approxnorm'],state_all_hg_err['cheatnorm']

plot_comparison_of_hg_err_approx(all_JHinv_r_gritz_true, all_JHinv_approx_r_gritz2, all_JHinv_approx_r_gritz_cheat, 
                                 label1='$W^{(i)}$ approx',  label2='$W^{(i+1)}$ approx',
                                 ylabel='log10 HG error', xlabel='Linear system number')

plot_comparison_of_hg_err_approx(all_JHinv_norm_true,all_JHinv_norm_approx,all_JHinv_norm_cheat, 
                                  label1='$W^{(i)}$ approx',  label2='$W^{(i+1)}$ approx',
                                  ylabel='Matrix 2-norm of preconditioner', xlabel='Linear system number', nolog=True)

#%% Recycling dimension


def plot_dim_comp_data(none, ritz, gritz, gritz2, xvals=None, plot_none=True, title=None, ylabel=None, no_sum=False, fontsize=23, xlabel='Dimension of recycle space', linewidth=3):
    plt.figure(figsize=(5.5,5))
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
    plt.legend(fontsize=fontsize*.7)
    plt.tight_layout()
    
    
state = torch.load( LOAD_DIR+'all_dims_results')
recycle_dims, cost_minres_ritz, cost_minres_gritz, cost_minres_gritz2 = state['rec_dims'], state['c_ritz'],state['c_gritz'],state['c_gritz2']
all_dim_ritz_stop_its, all_dim_gritz_stop_its, all_dim_gritz2_stop_its = state['stop_ritz'],state['stop_gritz'],state['stop_gritz2']
all_dim_ritz_hg_err, all_dim_gritz_hg_err, all_dim_gritz2_hg_err = state['hg_ritz'],state['hg_gritz'], state['hg_gritz2']


#% Comparison of FLOPS
plot_dim_comp_data(np.log10(np.sum(cost_none)),np.log10(np.sum(cost_minres_ritz, axis=1)),np.log10(np.sum(cost_minres_gritz, axis=1)),np.log10(np.sum(cost_minres_gritz2, axis=1)),
                   ylabel='log10 FLOPs of RMINRES', xvals=recycle_dims, no_sum=True)

#% Comparison of number of iterations
plot_dim_comp_data(stop_it_none,np.sum(all_dim_ritz_stop_its, axis=1),np.sum(all_dim_gritz_stop_its, axis=1),np.sum(all_dim_gritz2_stop_its, axis=1),
                   ylabel='Total RMINRES its', xvals=recycle_dims)
#% Mean HG relative err
plot_dim_comp_data(np.log10(all_hg_err_none.mean()),np.log10(np.mean(all_dim_ritz_hg_err, axis=1)),np.log10(np.mean(all_dim_gritz_hg_err, axis=1)),np.log10(np.mean(all_dim_gritz2_hg_err, axis=1)), no_sum=True, 
                   ylabel='log10 Mean HG rel. error', xvals=recycle_dims)
plot_dim_comp_data(np.log10(all_hg_err_none.max()),np.log10(np.max(all_dim_ritz_hg_err, axis=1)),np.log10(np.max(all_dim_gritz_hg_err, axis=1)),np.log10(np.max(all_dim_gritz2_hg_err, axis=1)), no_sum=True, 
                   ylabel='log10 Max HG rel. error', xvals=recycle_dims)
