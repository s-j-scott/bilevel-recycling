# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 22:55:24 2025

@author: sebsc
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
import math

def plot_data(vals, title=None, fontsize=23, xlabel=None, ylabel=None, linewidth=3, save=None):
    plt.figure(figsize=(5.5,5))
    plt.plot(vals,linewidth=linewidth)
    plt.title(title,fontsize=fontsize)
    plt.xlabel(xlabel,fontsize=fontsize)
    plt.ylabel(ylabel,fontsize=fontsize)
    plt.yticks(fontsize = fontsize*.8) 
    plt.xticks(fontsize = fontsize*.8) 
    plt.tight_layout()
    
    if isinstance(save, str): plt.savefig(save)
    plt.show()
    

def plot_timings(state, title=None, save=None, fontsize=15):

    fig,ax = plt.subplot_mosaic([
        ['t_total', 't_total_lls', 't_hess', 't_misc'  ],
        ['overall_ratio','overall_ratio','overall_ratio', 'overall_ratio']],
        figsize=(10,3.5), height_ratios=[2,5])
    
    col_ll = 'tab:blue'
    col_hess = 'tab:orange'
    col_misc = 'tab:green'
    alpha = 0.3 # opacity of filled blocks
    
    t_total = state['t_total']
    
    t_total_lls = state['t_ll']
    t_hess = state['t_hess']
    t_misc = t_total-t_total_lls-t_hess
    s_total = sum(t_total)
    s_hess = sum(t_hess)
    s_misc = sum(t_misc)
    s_total_lls = sum(t_total_lls)
    
    t_total_lls_rat = ( t_total_lls)/t_total
    t_hess_rat = ( t_hess )/t_total
    t_misc_rat = ( t_misc )/t_total
    
    print(f'\nTotal time : {s_total:.3f} | {100.00}%')
    print(f'Lower level: {s_total_lls:.3f} | {s_total_lls/s_total*100:.2f}%')
    print(f'Hessian    : {s_hess:.3f} | {s_hess/s_total*100:.2f}%')
    print(f'Misc       : {s_misc:.3f} | {s_misc/s_total*100:.2f}%\n')
    
    ################################################################
    ax["t_total"].plot((t_total) ,'k')
    ax["t_total"].set_title('Total iteration time\n{:.2f}s  ({:.2f}%)'.format(s_total, 100), fontsize=fontsize)
    ax["t_total"].set_ylim(0, 1.2*t_total.max())
    
    ax["t_total_lls"].plot((t_total_lls), color=col_ll)
    ax["t_total_lls"].set_title('Lower level solve time\n{:.2f}s  ({:.2f}%)'.format(s_total_lls, s_total_lls/s_total*100), fontsize=fontsize)
    ax["t_total_lls"].set_ylim(0, 1.2*t_total_lls.max())
    
    ax["t_hess"].plot((t_hess), color=col_hess)
    ax["t_hess"].set_title('Hessian solve time\n{:.2f}s  ({:.2f}%)'.format(s_hess, s_hess/s_total*100), fontsize=fontsize)
    ax["t_hess"].set_ylim(0, 1.2*t_hess.max())
    
    ax["t_misc"].plot((t_misc), color=col_misc)
    ax["t_misc"].set_title('Other\n{:.2f}s  ({:.2f}%)'.format(s_misc, s_misc/s_total*100), fontsize=fontsize)
    ax["t_misc"].set_ylim(0, 1.2*t_misc.max())
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
    plt.show()


def axis_fun(fig,ax,im):
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

def display_filters(param, title='', scaled=False, square=True, fontsize=25, suptitvals=None, suptitvals2=None, save=None):
    filter_num, tmp_size = param.shape
    filter_size = int((tmp_size-1)**.5)
    if square:
        cols = math.ceil(math.sqrt(filter_num))
        rows = math.ceil(filter_num / cols)
    else:
        rows = 3
        cols = filter_num // rows
    
    if scaled:
        all_params = torch.empty(filter_num, filter_size, filter_size)
        for i, param_tmp in enumerate(param):
            param_tmp = param[i,:-1].view(filter_size,filter_size).cpu()
            reg_param = 10**param[i,-1].cpu()
            all_params[i] = (param_tmp * reg_param)
        vmin = all_params.min()
        vmax = all_params.max()
        # print(f'vmin{vmin:.3f} vmax{vmax:.3f}')
    
    if suptitvals is not None:
        sorted_indices = sorted(range(filter_num), key=lambda i: suptitvals[i], reverse=True)
    else:
        sorted_indices = list(range(filter_num))
        
    fig, ax = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
    ax = ax.flatten()
    for i, idx in enumerate(sorted_indices):
        if square: i+= 1
        if scaled:
            ax[i].imshow(all_params[idx], vmin=vmin, vmax=vmax)
        else:
            param_tmp = param[idx]
            param_tmp = param[idx,:-1].view(filter_size,filter_size).cpu()
            ax[i].imshow(param_tmp)
        if suptitvals is not None:
            to_disp = f'{suptitvals[idx]:.3f}'
            if suptitvals2 is not None:
                to_disp += f'\n{suptitvals2[idx]:.3f}'
            ax[i].set_title(to_disp, fontsize=fontsize)
        ax[i].axis('off')
        
    # for j in range(i + 1, len(ax)):
    if square: ax[0].axis('off')
    plt.suptitle(title, fontsize=fontsize)
    plt.tight_layout()
    # plt.subplots_adjust(wspace=0.05, hspace=0.05)
    
    if isinstance(save,str): plt.savefig(save)
    plt.show()

def display_recons(
    ground_truth, tensors,
    subtitles=None, 
    fontsize=15, 
    save=None, 
    cmap='viridis'
    ):
    N = len(tensors)
    cols, rows = N+1, 1
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = axes.flatten()
    
    im = axes[0].imshow(ground_truth.detach().cpu(), cmap=cmap)
    axes[0].axis("off")
    axes[0].set_title('Ground truth', fontsize=fontsize-2)
    
    vmax = ground_truth.max().cpu()
    vmin = ground_truth.min().cpu()
    
    for i, ax in enumerate(axes[1:]):
        if i < N+1:
            tensor = tensors[i].detach().cpu().clip(vmin,vmax)
            ax.imshow(tensor, cmap=cmap)
            
            # Pick subtitle if provided, else fallback to numbered title
            if subtitles is not None:
                subtitle = subtitles[i]
                ax.set_title(subtitle, fontsize=fontsize-2)
            ax.axis("off")
        else:
            ax.axis("off")
    
    # plt.suptitle(title, fontsize=fontsize, y=1.02)
    plt.tight_layout()
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.04)
    
    if isinstance(save, str):
        plt.savefig(save, bbox_inches="tight")
    plt.show()

def view_tensor_images(
    tensors, 
    n=None, 
    title='', 
    subtitles=None, 
    fontsize=15, 
    save=None, 
    cmap='viridis'
):
    """
    Display multiple tensor images given as a list.
    
    tensors: list of torch.Tensors
        Each tensor should be 2D (H, W).
    n: tuple/list or int, optional
        Grid shape (rows, cols).
        If None, a near-square grid is chosen automatically.
    subtitles: list of str, optional
        Custom subtitles for each image.
    """
    
    N = len(tensors)
    if N == 0:
        raise ValueError("Empty tensor list provided.")
    
    # Choose grid size automatically if not provided
    if n is None:
        cols = int(math.ceil(math.sqrt(N)))
        rows = int(math.ceil(N / cols))
    elif isinstance(n, (list, tuple)):
        rows, cols = n
    else:
        rows, cols = n, n
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i < N:
            tensor = tensors[i].detach().cpu()
            im = ax.imshow(tensor, cmap=cmap)
            
            # Pick subtitle if provided, else fallback to numbered title
            subtitle = subtitles[i] if (subtitles is not None and i < len(subtitles)) else f"{title} {i}"
            ax.set_title(subtitle, fontsize=fontsize-2)
            ax.axis("off")
        else:
            ax.axis("off")
    
    plt.suptitle(title, fontsize=fontsize, y=1.02)
    plt.tight_layout()
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.04)
    
    if isinstance(save, str):
        plt.savefig(save, bbox_inches="tight")
    plt.show()
    
def view_tensor_image(tensor, n=None, title='', fontsize=15 , save=None):
    if isinstance(n, list):
        m,n = n
    elif isinstance(n, float):
        m=n
    else:
        [m,n] = list(tensor.shape)
    plt.figure(figsize=(4,3.5))
    im = plt.imshow(tensor.view(m,n).cpu())
    plt.title(title, fontsize=fontsize)
    plt.colorbar(im,fraction=0.046, pad=0.04)
    plt.axis('off')
    plt.tight_layout()
    
    if isinstance(save,str): plt.savefig(save)
    plt.show()


    
def get_color_scheme():
    def theme(linestyle='-', color='k', marker=None, markevery=0.1):
        return [color, linestyle, marker, markevery]
    theme_s = ['#E69F00', '-', None, 0] #Orange
    theme_m = ['#D55E00', '--', 'd', 30]
    theme_l = ['#56B4E9', ':',  'o', 30] #SkyBlue
    
    theme_alt = ['#0072b2', 'dashdot', '>', 30]
    
    def get_theme(name):
        if name[-4:] == '-NSC':
            return theme_alt
        elif '-S' in name:
            return theme_s
        elif '-M' in name:
            return theme_m
        elif '-L' in name:
            return theme_l
        
    color_dict = {  # Display settings with variant-based coloring
        # -- Full dimension --
        'Eig-S':                 theme('-', '#d55e00', 'd', 30),
        'GSVD-L(R)':             theme(':', '#0072B2', '>', 30),
        # -- Krylov Baselines --
        'MINRES cold':           theme('solid', 'k', 'o', 20),
        'CG warm':               theme('dashed', '#d55e00'),
        'CG cold':               theme('solid', '#d55e00'),
        # -- Outer/Inner Experiments --
        'Ritz-S Outer':             theme_s,
        'Ritz-S Inner':             ['#0072B2', '--', 'o', 30],
        'RGen-L(R) Outer':          theme_l,
        'RGen-L(R) Inner':          ['#d55e00', '-', None, 0],
        'RGen-L(R)-NSC Outer':      theme_alt,
        'RGen-L(R)-NSC Inner':      ['#cc79a7', '-', None, 0],
    }
    
    for name in [
        # -- Ritz --
        'Ritz-S','Ritz-L','Ritz-M',
        # -- Harmonic Ritz --
        'HRitz-S','HRitz-L','HRitz-M',
        # -- RGen Left --
        'RGen-S(L)','RGen-L(L)', 'RGen-L(L)-NSC','RGen-M(L)','RGen-M(L)-NSC',       
        # -- RGen Right --
        'RGen-S(R)','RGen-S(R)-NSC','RGen-L(R)','RGen-L(R)-NSC','RGen-M(R)','RGen-M(R)-NSC',        
        # -- RGen Mix --
        'RGen-S(M)','RGen-S(M)-NSC','RGen-L(M)','RGen-L(M)-NSC','RGen-M(M)','RGen-M(M)-NSC',
        ]: 
        color_dict[name] = get_theme(name)
    return color_dict

color_dict = get_color_scheme()

def process_result(key, result):
    if key[:7] == 'cumsum_':
        y = np.cumsum(result[key[7:]])
    elif key[:6] == 'log10_':
        y = torch.log10(result[key[6:]])
    else:
        y = result[key]
    return y

def compare_performance(
    res_baseline,
    dict_to_plot,
    quantity_keys=['stop_it', 'cumsum_stop_it','log10_hg_rerr'],
    ylabels=['Number of its', 'Cumulative No. of its', 'log10 HG relative error'],
    title=None,
    xvalues=None,
    baseline_label='None',
    xlabel='Linear system number',
    fs=22,
    lw=4,
    shared_legend = True,
    save=False
):
    """
    Compare performance across multiple methods with customizable metrics.

    Parameters:
        res_baseline: dict
            Baseline result dictionary.
        dict_to_plot: dict[str, dict]
            Other methods' result dictionaries, keyed by method name.
        color_dict: dict[str, tuple]
            Maps method name to (color, linestyle, marker, markevery).
        quantity_keys: list[str]
            Keys in the result dicts to plot (e.g., 'stop_it', 'hg_rerr').
        ylabels: list[str]
            Y-axis labels for each subplot (must match quantity_keys).
        title: str
            Optional super title.
        baseline_label: str
            Label for baseline line in legend.
        xlabel: str
            X-axis label (shared).
        fs: int
            Font size.
        lw: int
            Line width.
    """
    assert len(quantity_keys) == len(ylabels), "Each quantity must have a corresponding y-label."
    

    num_plots = len(quantity_keys)
    markersize = 13

    fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots,4.5), sharex=False)

    if num_plots == 1:
        axes = [axes]  # Make iterable

    for i, (key, ylabel) in enumerate(zip(quantity_keys, ylabels)):
        ax = axes[i]

        # Handle special cases like 'log10'
        y_baseline = process_result(key, res_baseline)
        if xvalues is None:
            xvalues = np.arange(len(y_baseline))

        ax.plot(
            xvalues,
            y_baseline,
            linestyle=(0, (1, 1)),
            color='k',
            label=baseline_label,
            linewidth=lw,
            marker='*',
            markevery=30,
            markersize=markersize
        )

        for label, results in dict_to_plot.items():
            color, linestyle, marker, markevery = color_dict[label]

            y = process_result(key, results)

            ax.plot(
                xvalues,
                y,
                label=label,
                linewidth=lw,
                linestyle=linestyle,
                marker=marker,
                markevery=markevery,
                color=color,
                markersize=markersize
            )

        ax.set_ylabel(ylabel, fontsize=fs)
        ax.set_xlabel(xlabel, fontsize=fs)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.tick_params(axis='both', labelsize=fs * 0.8)

    plt.tight_layout()
    if shared_legend:
        # Shared legend at the top
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles, labels,
            loc='upper center',
            ncol=len(labels),
            fontsize=fs * 0.85,
            frameon=False,
            bbox_to_anchor=(0.5, 1.05)
        )
        fig.subplots_adjust(top=0.92)
    else:
        for ax in axes:
            ax.legend(fontsize=fs*.7)
    if title:
        plt.suptitle(title, fontsize=fs)
    
    if isinstance(save, str):
        plt.savefig(save)
    plt.show()
    
def table_comparison(res_none, res_to_print):
    print('\n Recycle method  |  #Its    (Save%)  |  Avg log HG Err')
    print('-----------------|-------------------|-----------------')
    none_sum = sum(res_none['stop_it'])
    print('{0:16} | {1:5}  ( 100.00%) | {2:7.3f} '.format('None', none_sum , torch.log10(res_none['hg_rerr'].mean())))
    for name, result in res_to_print.items():
        tmp_sum = sum(result['stop_it'])
        print('{0:16} | {1:5}  ({2:7.2f}%) | {3:7.3f} '.format(name, tmp_sum, tmp_sum/none_sum*100, torch.log10(result['hg_rerr'].mean())))

    
def plot_comparison_of_hg_err_approx(actual, approx, cheat_approx, label1='$W^{(i)}$ approx', nolog=False,label2='$W^{(i+1)}$ approx', title='', xlabel='RMINRES iteration', ylabel='log10 error of\nHG error approx', fontsize=25, save=None):
    plt.figure(figsize=(5.5,4.5))
    if nolog:
        plt1, plt2, plt3 = actual, approx, cheat_approx
    else:
        plt1, plt2, plt3 = np.log10(actual), np.log10(approx), np.log10(cheat_approx)
    
    xplt = np.arange(len(plt1))+1
    plt.plot(xplt,plt1, ':', label='True', linewidth=3, color='black', marker='*', markevery=.1)
    plt.plot(xplt,plt2, label=label1, linewidth=3, color='tab:olive', marker='d', markevery=.15)
    plt.plot(xplt,plt3, '--', label=label2, linewidth=3, color='tab:olive')
    plt.legend(fontsize=fontsize*.6)
    plt.xlabel(xlabel,fontsize=fontsize)
    plt.ylabel(ylabel,fontsize=fontsize)
    plt.title(title,fontsize=fontsize)
    plt.tight_layout()
    if save is not None:
        plt.savefig(save)
    plt.show()

