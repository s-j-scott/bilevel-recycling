# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 22:18:18 2023

@author: Sebastian J. Scott


"""

import torch
import numpy as np
import pandas as pd
import json

import matplotlib.pyplot as plt
from matplotlib import colors

def set_device(verbose = True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose:
        print('Using device:', device)
        print()
        #Additional Info when using cuda
        if device.type == 'cuda':
            print(torch.cuda.get_device_name(0))
            print('Memory Usage:')
            print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**2,3), 'MB')
            print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**2,3), 'MB')
            print()
    return device

def save_dataset(xexacts,ys, save=True, path=''):
    xexacts_np = xexacts.detach().cpu().numpy()
    ys_np = ys.detach().cpu().numpy()
    if save:
        np.savez(path+"OUTPUT_data_groundtruths", *xexacts_np)
        np.savez(path+"OUTPUT_data_measurements", *ys_np)
    return xexacts_np, ys_np


def load_dataset(path):
    container = np.load(path+"OUTPUT_data_groundtruths.npz")
    xexacts = np.array([container[key] for key in container])
    container = np.load(path+"OUTPUT_data_measurements.npz")
    ys = np.array([container[key] for key in container])
    return xexacts, ys

def load_dataset_as_tensor(path,device='cpu'):
    xexacts, ys = load_dataset(path)
    return torch.tensor(xexacts,device=device), torch.tensor(ys,device=device)


def save_bl_info(bl_recons , bl_info, save=True, path=None):
    if path is None: path = ''
    params = np.array([t.detach().cpu().numpy() for t in bl_info['ul_info']['all_iters']])
    recons = np.array([t.detach().cpu().numpy() for t in bl_recons])
    # if isinstance(bl_recons, list):
    # else:
    #     recons = bl_recons.cpu()
    
    ul_evals = bl_info['ul_info']["fun_evals"] # Evaluation of upper level
    ul_gnorms = bl_info['ul_info']["grad_norms"] # UL gradient norm
    ul_steplengths = bl_info['ul_info']["step_lengths"] # UL determiend step length
    ul_backtrack_its = bl_info['ul_info']['backtrack_its'] #Number of backtracking iterations
    ul_itn_times = bl_info['ul_info']["t_iteration_total"] # UL time of each iteration
    ul_grad_times = bl_info['ul_info']["t_grad_computation"] # UL time of gradient computation
    t_ul_steplength = bl_info['ul_info']["t_step_length"] # UL time of backtracking linesearch
    ll_solves_per_ul_it = bl_info['ul_info']["ll_solves_per_ul_it"] # UL time of backtracking linesearch
    # Time spent solving lower level problems for that UL iteration
    if len(t_ul_steplength) < len(ul_gnorms):
        t_ul_steplength = np.append(t_ul_steplength,None)
        ul_steplengths = np.append(ul_steplengths, None)
        ul_backtrack_its = np.append(ul_backtrack_its, None)
    rolling_sum = 0
    # data_num = len(bl_recons)
    t_total_lls = np.empty(len(ll_solves_per_ul_it))
    for ind in range(len(t_total_lls)):
        # Each item of tmp_ll_infos is a list with ll_info for each item in dataset
        tmp_ll_infos = bl_info['ll_call_infos'][rolling_sum:(rolling_sum+ll_solves_per_ul_it[ind])]
        # tmp_sum = 0
        # for data_ind in range(len(bl_recons)):
            # tmp_sum += sum([tmp_info[data_ind]['t_total'] for tmp_info in tmp_ll_infos])
        # t_total_lls[ind] = sum([ll_single['t_total'] for ll_single in  ])
        # t_total_lls[ind] = tmp_sum
        t_total_lls[ind] = sum([sum([item['t_total'] for item in ll_info]) for ll_info in tmp_ll_infos])
        rolling_sum += ll_solves_per_ul_it[ind]

    # hess_res = np.array([info['r_norms'].cpu().numpy() for info in bl_info['hess_infos']]) # Residual of Hessian system solution
    
    hess_res = np.array([sum([item['r_norms'][-1].cpu() for item in h_info]) for h_info in bl_info['hess_infos']]) # Number of iterations to solve Hessian system 
    hess_its = np.array([sum([item['stop_it'] for item in h_info]) for h_info in bl_info['hess_infos']]) # Number of iterations to solve Hessian system  
    t_hess = np.array([sum([item['t_total'] for item in h_info]) for h_info in bl_info['hess_infos']]) # Time of Hessian system solve
    hess_flag = np.array([sum([item['convg_flag'] for item in h_info]) for h_info in bl_info['hess_infos']]) # Number of iterations to solve Hessian system  
    
    ll_flag = np.array([sum([item['convg_flag'] for item in ll_info]) for ll_info in bl_info['ll_infos']]) # Time of Hessian system solve
    ll_gnorms = np.array([sum([item['grad_norms'].cpu() for item in ll_info]) for ll_info in bl_info['ll_infos']]) # Time of Hessian system solve
    
    # ll_gnorms = np.array([info['grad_norms'].cpu() for info in bl_info['ll_infos']]) # LL gradient norm
    ll_its =  np.array([sum([item['stop_it'] for item in ll_info]) for ll_info in bl_info['ll_infos']]) # LL stopping iteration
    t_ll_solve= np.array([sum([item['t_total'] for item in ll_info]) for ll_info in bl_info['ll_infos']])  ## LL time to solve associated reconstruction
    
    
    state = {"ul_evals":ul_evals, "ul_gnorms":ul_gnorms, "ul_steplengths":ul_steplengths,
                       "ul_steplengths":ul_steplengths,
                       "ul_backtrack_its" :ul_backtrack_its,
                       "ul_itn_times":ul_itn_times ,
                       "t_ul_grad":ul_grad_times,
                       "t_ul_steplength":t_ul_steplength, 
                       "ll_solves_per_ul_it":ll_solves_per_ul_it,
                       "t_total_lls":t_total_lls,
                       "hess_res":hess_res,
                        "hess_flag":hess_flag,
                       "hess_its":hess_its,
                       "t_hess":t_hess,
                       "ll_gnorms":ll_gnorms ,
                       "ll_its" :ll_its,
                       "ll_flag":ll_flag,
                       "t_ll_solve":t_ll_solve
                       }
    # for key in state.keys():
    #     print(key + f' | length {len(state[key])}')
    
    df = pd.DataFrame(state)
    
    if save:
        np.savez(path+"OUTPUT_params",*params)
        np.savez(path+"OUTPUT_recons",*recons)
        df.to_csv(path+"OUTPUT_bl_info.csv", index=False)
    return params, recons, df


def load_info(path):
    df_bl_info = pd.read_csv(path+"OUTPUT_bl_info.csv")
    container = np.load(path+"OUTPUT_params.npz")
    params = np.array([container[k] for k in container])
    container = np.load(path+"OUTPUT_recons.npz")
    recons = np.array([container[k] for k in container])
    container = np.load(path+"OUTPUT_data_groundtruths.npz")
    xexacts = np.array([container[key] for key in container])
    container = np.load(path+"OUTPUT_data_measurements.npz")
    ys = np.array([container[key] for key in container])

    with open(path+"OUTPUT_setup.json","r") as f:
        setup = json.load(f)
        
    return setup, params, xexacts, ys, recons, df_bl_info 
    
#%% Convert all tensors in a dictionary into ndarrays
def untensor_dict(dictionary):
    # Iterate through all sub-dictionaries of dictionary and replaces, if possible, a tensor with a numpy array
    if isinstance(dictionary,dict):
        for key in dictionary:
            dictionary[key] = untensor_dict(dictionary[key])
    elif isinstance(dictionary, list):
        for ind in range(len(dictionary)):
            dictionary[ind] = untensor_dict(dictionary[ind])
    elif torch.is_tensor(dictionary):
        dictionary = dictionary.detach().cpu().numpy()
    return dictionary

#%% Visualisation of results

def display_image(array, title=None,save=None):
    plt.figure()
    plt.ion()
    plt.imshow(array)
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.ioff()
    if isinstance(save,str): plt.savefig(save)
    
def display_images(xexact,y,xrecon, title=None,save=None, fs=15, cmap='Greys'):
    fig,ax = plt.subplot_mosaic([['xexact','y','recon']],figsize=(10,4))
    
    
    im=ax['xexact'].imshow(xexact,cmap=cmap)
    ax['xexact'].set_title(r'Ground truth',fontsize=fs)
    axis_fun(fig,ax['xexact'],im)
    
    im=ax['y'].imshow(y,cmap=cmap)
    ax['y'].set_title(r'Measurement',fontsize=fs)
    axis_fun(fig,ax['y'],im)
    
    im=ax['recon'].imshow(xrecon,cmap=cmap)
    ax['recon'].set_title(r'Reconstruction',fontsize=fs)
    axis_fun(fig,ax['recon'],im)
    
    plt.suptitle(title)
    plt.tight_layout()
    if isinstance(save,str): plt.savefig(save)
    
    
def insert_row_of_images(ax, ax_ind, img_ind, images, title=None, clip=False, vmin=0,vmax=1, cmap='Viridis'):
    '''Misc function useful for the display_recons_of_dataset function '''
    try:    image_display = images[img_ind].copy() 
    except: image_display = images[img_ind].clone()
    if clip: 
        image_display = np.clip(image_display, vmin,vmax)
        if len(ax.shape)==1:
            im = ax[ax_ind].imshow(image_display, vmin=vmin, vmax=vmax)
        else:
            im = ax[ax_ind,img_ind].imshow(image_display, vmin=vmin, vmax=vmax)
    else:
        if len(ax.shape)==1:
            im = ax[ax_ind].imshow(image_display)
        else:
            im = ax[ax_ind,img_ind].imshow(image_display)
            #
    if len(ax.shape)==1:
        plt.colorbar(im, ax=ax[ax_ind])
        ax[ax_ind].set_title(title)
    else:
        plt.colorbar(im, ax=ax[ax_ind,img_ind])
        ax[ax_ind,img_ind].set_title(title)

def display_recons_of_dataset(xexacts,ys,recons, max_display=5, save=None, clip=False):
    ''''Take selection of entries in the training data and display the 
    ground truth, noisy measurement, and reconstruction. '''
    if len(xexacts) < max_display: num_display = len(xexacts)
    else: num_display = max_display
    fig, ax = plt.subplots(3, num_display, figsize=(10,7))
    for ind in range(num_display):
            insert_row_of_images(ax,0,ind, xexacts, title='Ground truth', clip=clip)
            insert_row_of_images(ax,1,ind, ys, title='Measurement', clip=clip)
            insert_row_of_images(ax,2,ind, recons, title='Reconstruction', clip=clip)
    plt.tight_layout()    
    
    if isinstance(save,str): plt.savefig(save)

    
    
    

def axis_fun(fig,ax,im):
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def display_filters(xexact, y, params, recon, setup, A=None, fs=15, title=None, device=None, save=None, cmap='viridis'):
    
    filter_shape = setup['problem_setup']['regulariser']['filter_shape']
    filter_num = setup['problem_setup']['regulariser']['filter_num']
    
    inpaint=False
    if setup['problem_setup']['forward_op'] == 'inpainting': 
        inpaint=True
        mask_np = A.mask.cpu().numpy()
        y_pad = np.empty(xexact.shape) * np.nan
        y_pad[mask_np] = y[0]
    for filter_ind in range(filter_num):
        # fig,ax = plt.subplot_mosaic([['xexact','y','recon','filter_traj','filter_traj'],
        #                              ['x_conv','y_conv','filter','filter_traj','filter_traj']],figsize=(18,8))
        
        fig,ax = plt.subplot_mosaic([['xexact','xexact','y','y','recon','recon','filter_traj','filter_traj','filter_traj','filter_traj'],
                                     ['xexact','xexact','y','y','recon','recon','filter_traj','filter_traj','filter_traj','filter_traj'],
                                     ['x_conv','x_conv','y_conv','y_conv','filter','filter','filter_traj','filter_traj','filter_traj','filter_traj'],
                                     ['x_conv','x_conv','y_conv','y_conv','filter','filter','reg_traj','reg_traj','reg_traj','reg_traj']],figsize=(18,8))
        
        im=ax['xexact'].imshow(xexact,cmap=cmap)
        ax['xexact'].set_title(r'Ground truth $x^\star$',fontsize=fs)
        axis_fun(fig,ax['xexact'],im)
        
        if inpaint: im=ax['y'].imshow(y_pad,cmap=cmap)
        else: im=ax['y'].imshow(y,cmap=cmap)
        ax['y'].set_title(r'Measurement $y$',fontsize=fs)
        axis_fun(fig,ax['y'],im)
        
        im=ax['recon'].imshow(recon,cmap=cmap)
        ax['recon'].set_title(r'Reconstruction',fontsize=fs)
        axis_fun(fig,ax['recon'],im)
        
        foe_filter = params[-1][filter_ind][:-1].reshape(filter_shape,filter_shape)
        # foe_filter = params[-1][filter_ind].reshape(filter_shape,filter_shape)
        conv_op = torch.nn.Conv2d(1,1,kernel_size=filter_shape,padding='same', bias=False, device=device)
        conv_op.weight = torch.nn.Parameter(torch.tensor(foe_filter, device=device).unsqueeze(0).unsqueeze(0))
        x_conv = conv_op(torch.tensor(xexact).unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0).clone().detach().cpu().numpy()
        y_conv = conv_op(torch.tensor(y).unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0).clone().detach().cpu().numpy()
                
        val = np.max(np.abs(foe_filter))
        norm = colors.TwoSlopeNorm(vmin=-val, vcenter=0, vmax=val)
        im = ax['filter'].imshow(foe_filter,cmap='bwr',norm=norm)
        ax['filter'].set_title('Learned filter\nSum:{:.4f}'.format(foe_filter.sum()),fontsize=fs)
        axis_fun(fig,ax['filter'],im)
        
        val = np.max(np.abs(x_conv))
        norm = colors.TwoSlopeNorm(vmin=-val, vcenter=0, vmax=val)
        im=ax['x_conv'].imshow(x_conv,cmap='bwr',norm=norm)
        ax['x_conv'].set_title(r'Filter applied to $x^\star$',fontsize=fs)
        axis_fun(fig,ax['x_conv'],im)
        
        val = np.max(np.abs(y_conv))
        norm = colors.TwoSlopeNorm(vmin=-val, vcenter=0, vmax=val)
        im=ax['y_conv'].imshow(y_conv,cmap='bwr',norm=norm)
        ax['y_conv'].set_title(r'Filter applied to $y$',fontsize=fs)
        axis_fun(fig,ax['y_conv'],im)
        
        ax['filter_traj'].plot([p[filter_ind][:-1].ravel() for p in params])
        ax['filter_traj'].set_title('Evolution of filter',fontsize=fs)
        ax['filter_traj'].set_xlabel('GD iteration',fontsize=fs-3)
        
        
        ax['reg_traj'].plot([p[filter_ind][-1].ravel() for p in params])
        ax['reg_traj'].set_title('Evolution of reg param',fontsize=fs)
        ax['reg_traj'].set_xlabel('GD iteration',fontsize=fs-3)
        
        plt.suptitle(title,fontsize=fs)
        plt.tight_layout()
        if isinstance(save,str): plt.savefig(save+str(filter_ind+1))
    
    

def plot_timings(df_bl_info, title=None, save=None):

    fig,ax = plt.subplot_mosaic([
        ['t_total', 't_total_lls', 't_hess', 't_misc'  ],
        ['overall_t','overall_t','overall_t', 'overall_t'],
        ['t_total_rat', 't_total_lls_rat', 't_hess_rat', 't_misc_rat'  ],
        ['overall_ratio','overall_ratio','overall_ratio', 'overall_ratio']],
        figsize=(16,10))
    
    col_ll = 'tab:blue'
    col_hess = 'tab:orange'
    col_misc = 'tab:green'
    alpha = 0.3 # opacity of filled blocks
    
    t_total = df_bl_info['ul_itn_times']
    
    t_total_lls = df_bl_info['t_total_lls']
    t_hess = df_bl_info['t_hess']
    t_misc = t_total-t_total_lls-t_hess
    s_total = sum(t_total)
    s_total_lls = sum(t_total_lls)
    s_hess = sum(t_hess)
    s_misc = sum(t_misc)
    
    t_total_rat = t_total/t_total
    t_total_lls_rat = ( t_total_lls)/t_total
    t_hess_rat = ( t_hess )/t_total
    t_misc_rat = ( t_misc )/t_total
    ################################################################
    ax["t_total"].plot(np.log10(t_total))
    ax["t_total"].set_title('Log of overall iteration time\nTotal time:{:.2f}  ({:.2f}%)'.format(s_total, 100))
    ax["t_total_rat"].plot(t_total_rat)
    ax["t_total_rat"].set_title('Ratio overall iteration time\nMax:{:.3f} Min:{:.3f} Mean:{:.3f}'.format(t_total_rat.max(), t_total_rat.min(), t_total_rat.mean()))
    ax["t_total_rat"].set_ylim(-.05,1.05)
    
    ax["t_total_lls"].plot(np.log10(t_total_lls), color=col_ll)
    ax["t_total_lls"].set_title('Log of total LL solves time\nTotal time:{:.2f}  ({:.2f}%)'.format(s_total_lls, s_total_lls/s_total*100))
    ax["t_total_lls_rat"].plot(t_total_lls_rat, color=col_ll)
    ax["t_total_lls_rat"].set_title('Ratio of total LL solves time\nMax:{:.3f} Min:{:.3f} Mean:{:.3f}'.format(t_total_lls_rat.max(), t_total_lls_rat.min(),t_total_lls_rat.mean()))
    ax["t_total_lls_rat"].set_ylim(-.05,1.05)
    
    ax["t_hess"].plot(np.log10(t_hess), color=col_hess)
    ax["t_hess"].set_title('Log of Hessian solve time\nTotal time:{:.2f}  ({:.2f}%)'.format(s_hess, s_hess/s_total*100))
    ax["t_hess_rat"].plot(t_hess_rat, color=col_hess)
    ax["t_hess_rat"].set_title('Ratio of Hessian solve time\nMax:{:.3f} Min:{:.3f} Mean:{:.3f}'.format(t_hess_rat.max(), t_hess_rat.min(), t_hess_rat.mean()))   
    ax["t_hess_rat"].set_ylim(-.05,1.05)
    
    ax["t_misc"].plot(np.log10(t_misc), color=col_misc)
    ax["t_misc"].set_title('Log of time for misc computations\nTotal time:{:.2f}  ({:.2f}%)'.format(s_misc, s_misc/s_total*100))     
    ax["t_misc_rat"].plot(t_misc_rat, color=col_misc)
    ax["t_misc_rat"].set_title('Ratio of time for misc computations\nMax:{:.3f} Min:{:.3f} Mean:{:.3f}'.format(t_misc_rat.max(), t_misc_rat.min(), t_misc_rat.mean()))  
    ax["t_misc_rat"].set_ylim(-.05,1.05)     
    
    x = np.linspace(0,len(t_misc)-1, len(t_misc))
    ax["overall_t"].plot(t_total_lls, color=col_ll, label='LLs' )
    ax["overall_t"].fill_between(x=x,y1=t_total_lls, color=col_ll, alpha=alpha)
    ax["overall_t"].plot(t_total_lls+t_hess, color=col_hess, label='Hess' )
    ax["overall_t"].fill_between(x=x,y1=t_total_lls+t_hess, y2=t_total_lls, color=col_hess, alpha=alpha)
    ax["overall_t"].plot(t_total_lls+t_hess + t_misc , color=col_misc, label='Misc')
    ax["overall_t"].fill_between(x=x,y1= t_total_lls+t_hess + t_misc, y2=t_total_lls+t_hess, color=col_misc, alpha=alpha)
    ax["overall_t"].set_title('Breakdown of overall timing')
    ax["overall_t"].legend()
    # ax["overall_t"].set_yscale('log')
    
    ax["overall_ratio"].plot(t_total_lls_rat, color=col_ll, label='LLs' )
    ax["overall_ratio"].fill_between(x=x,y1=t_total_lls_rat, color=col_ll, alpha=alpha)
    ax["overall_ratio"].plot(t_total_lls_rat+t_hess_rat, color=col_hess, label='Hess' )
    ax["overall_ratio"].fill_between(x=x,y1=t_total_lls_rat+t_hess_rat, y2=t_total_lls_rat, color=col_hess, alpha=alpha)
    ax["overall_ratio"].plot(t_total_lls_rat+t_hess_rat + t_misc_rat , color=col_misc, label='Misc')
    ax["overall_ratio"].fill_between(x=x,y1= t_total_lls_rat+t_hess_rat + t_misc_rat, y2=t_total_lls_rat+t_hess_rat, color=col_misc, alpha=alpha)
    ax["overall_ratio"].set_title('Breakdown of overall ratio')
    ax["overall_ratio"].legend()
    ax["overall_ratio"].set_ylim(-.05,1.05)

    
    plt.suptitle(title)
    plt.tight_layout()
    if isinstance(save,str): plt.savefig(save)
    
def plot_info(df_bl_info, params, title=None, save=None):
    fig,ax = plt.subplot_mosaic([
        ['ul_eval','ul_grad'            ,'hess_res','step_length'  ,'ll_grad'   ],
        ['t_total','t_grad_computation' ,'t_hess'  ,'t_step_length','t_ll_solve'],
        ['params' ,'params'             ,'hess_its','backtrack_its','ll_its'    ]],
        figsize=(18,8))
    
    
    ax["ul_eval"].plot(np.log10(df_bl_info['ul_evals']))
    ax["ul_eval"].set_title('Log of evaluation of UL')
    
    ax["ul_grad"].plot(np.log10(df_bl_info['ul_gnorms']))
    ax["ul_grad"].set_title('log of UL grad norm')
    
    ax["hess_res"].plot(np.log10(df_bl_info['hess_res']))
    ax["hess_res"].set_title('Log of residual norm of Hessian solve')
    
    
    ax["step_length"].plot(np.log10(df_bl_info['ul_steplengths']))
    ax["step_length"].set_title('Log of UL step length')
    
    ax["ll_grad"].plot(np.log10(df_bl_info['ll_gnorms']))
    ax["ll_grad"].set_title('Log of LL grad norm at recon')
    ################################################################
    ax["t_total"].plot(np.log10(df_bl_info['ul_itn_times']))
    ax["t_total"].set_title('Log of iteration time (s)')
    
    ax["t_grad_computation"].plot(np.log10(df_bl_info['t_ul_grad']))
    ax["t_grad_computation"].set_title('Log UL grad computation time')
    
    ax["t_hess"].plot(np.log10(df_bl_info['t_hess']))
    ax["t_hess"].set_title('Log of Hessian solve time')   
    
    ax["t_step_length"].plot(np.log10(df_bl_info['t_ul_steplength']))
    ax["t_step_length"].set_title('Log of UL backtracking time')
    
    ax["t_ll_solve"].plot(np.log10(df_bl_info['t_ll_solve']))
    ax["t_ll_solve"].set_title('Log of LL solve time')
    ################################################################
    ax["params"].plot([p.ravel() for p in params])
    ax["params"].set_title('History of parameters')
    
    ax["hess_its"].plot(df_bl_info['hess_its'])
    ax["hess_its"].set_title('Number of iterations to solve Hess system\nTotal: {}'.format(df_bl_info['hess_its'].sum()))    

    ax["backtrack_its"].plot(df_bl_info['ul_backtrack_its'])
    ax["backtrack_its"].set_title('Number of backtracking iterations')
    
    ax["ll_its"].plot(df_bl_info['ll_its'])
    ax["ll_its"].set_title('Number of iteration of LL solve')
    
    plt.suptitle(title)
    plt.tight_layout()
    if isinstance(save,str): plt.savefig(save)


def create_title_str(setup, full_title=True):
    title = 'Data={}, A={}, n={}\nReg:{}, expert:{}, #filter:{}, width:{}'.format(
        setup['problem_setup']['ground_truth'],
        setup['problem_setup']['forward_op'],
        setup['problem_setup']['n'],
        setup['problem_setup']['regulariser']['name'],
        setup['problem_setup']['regulariser']['expert'],
        setup['problem_setup']['regulariser']['filter_num'],
        setup['problem_setup']['regulariser']['filter_shape'])
    solver_detail = '\nUL tol:{:.2e}, maxit:{}\nLL tol:{:.2e}, maxit:{}\nHess tol:{:.2e}, maxit:{}, recycle:{}, rec_dim:{}'.format(
        setup['solver_optns']['ul_solve']['tol'],
        setup['solver_optns']['ul_solve']['max_its'],
        setup['solver_optns']['ll_solve']['tol'],
        setup['solver_optns']['ll_solve']['max_its'],
        setup['solver_optns']['hess_sys']['tol'],
        setup['solver_optns']['hess_sys']['max_its'],
        setup['solver_optns']['hess_sys']['recycle_strategy'],
        setup['solver_optns']['hess_sys']['recycle_dim'])
    if full_title:
        title += solver_detail
    return title

    
def view_tensor_image(tensor, m, n=None, title='', fontsize=15 ):
    if n is None: n = m
    plt.figure(figsize=(4,3.5))
    im = plt.imshow(tensor.view(m,n).cpu())
    plt.title(title, fontsize=fontsize)
    plt.colorbar(im,fraction=0.046, pad=0.04)
    plt.axis('off')
    plt.tight_layout()
    
    
    
def plot_data(vals, title=None, fontsize=23, xlabel=None, ylabel=None, linewidth=3):
    plt.figure(figsize=(5.5,5))
    plt.plot(vals,linewidth=linewidth)
    plt.title(title,fontsize=fontsize)
    plt.xlabel(xlabel,fontsize=fontsize)
    plt.ylabel(ylabel,fontsize=fontsize)
    plt.yticks(fontsize = fontsize*.8) 
    plt.xticks(fontsize = fontsize*.8) 
    plt.tight_layout()
    

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
    
# def compare_performance(stop_its_none, cost_none, hg_err_none, list_its, list_costs, list_hg_errs, list_labels,title=None, fs=20, lw=1.5):
#     fig, ax = plt.subplot_mosaic([['it_plot','it_ratio','flop_plot','flop_ratio', 'hg_err']], figsize=(18,6))
#     ax['it_plot'].set_title('Its', fontsize=fs)
#     ax['it_plot'].plot(stop_its_none ,':', color='k', label='None', linewidth=lw)    
#     ax['hg_err'].set_title('log10 HG Error', fontsize=fs)
#     ax['hg_err'].plot(torch.log10(hg_err_none) ,':', color='k', label='None', linewidth=lw)
#     ax['it_ratio'].set_title('Ratio of its', fontsize=fs)
#     ax['flop_plot'].plot(np.log10(cost_none),color='k' , linestyle=':', label='None', linewidth=lw)
#     ax['flop_plot'].set_title('log10 FLOPs', fontsize=fs)
#     ax['flop_ratio'].set_title('Ratio of FLOPs', fontsize=fs)
#     for i in range(len(list_its)):
#         ax['it_plot'].plot(list_its[i] , label=list_labels[i], linewidth=lw)
#         ax['it_ratio'].plot(list_its[i]/stop_its_none ,  label=list_labels[i], linewidth=lw)
#         ax['flop_plot'].plot(np.log10(list_costs[i]) ,  label=list_labels[i], linewidth=lw)
#         ax['flop_ratio'].plot(list_costs[i]/cost_none ,  label=list_labels[i], linewidth=lw)
#         ax['hg_err'].plot(torch.log10(list_hg_errs[i]) , label=list_labels[i], linewidth=lw)
#     for a in ax:
#         ax[a].set_xlabel('UL iteration',fontsize=fs*.7)
#         ax[a].legend(fontsize=fs*.7)
#     plt.suptitle(title)
#     plt.tight_layout()