# -*- coding: utf-8 -*-
"""
Created on Fri Jul 11 07:47:19 2025

@author: sebsc
"""

import torch
import numpy as np
import pandas as pd
import random
import os
import json
from tqdm import tqdm
import sys

import torch.distributed as dist

from src.bilevel import  solve_lower_level_singleton

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


def to_device(batch, device):
    """
    Recursively move all tensors in a dictionary of lists (or nested structures)
    to the specified device.
    
    Args:
        batch (dict): Dictionary where values can be lists of tensors, tensors, or nested dicts.
        device (torch.device or str): Target device (e.g. 'cuda' or 'cpu')
    
    Returns:
        dict: New dictionary with all tensors moved to device.
    """
    def move_item(item):
        if isinstance(item, list):
            return [move_item(i) for i in item]
        elif isinstance(item, dict):
            return {k: move_item(v) for k, v in item.items()}
        elif isinstance(item, int):
            return item
        else:
            return item.to(device) 

    return {k: move_item(v) for k, v in batch.items()}


def tqdm_compliant_print(msg, distributed=True):
    """ Allow for pretty prints while debugging (on CPU) and avoid
    duplicated prints of tqdm progress bar when on GPU
    """
    if distributed:
        print(msg, file=sys.stderr)
    else:        
        tqdm.write(msg, file=sys.stdout)

def set_seed(seed, deterministic = True):
    """Set all relevant seeds for full reproducibility in 
    PyTorch + NumPy + random."""
    
    # Python built-in random
    random.seed(seed)
    
    # NumPy random
    np.random.seed(seed)
    
    # PyTorch CPU
    torch.manual_seed(seed)
    
    # PyTorch CUDA
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU

    # Environment hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

    # CuDNN deterministic behavior
    if deterministic:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)  # PyTorch >= 1.8
    else:
        os.environ.pop('CUBLAS_WORKSPACE_CONFIG', None)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(False)
        
#%% Distributed learning

def ddp_setup(rank, world_size, backend='nccl'):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    if world_size > 1:    
        # Initialize the process group
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
        local_rank = rank - gpus_per_node * (rank // gpus_per_node)
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        # print(f"host: {gethostname()}, rank: {rank}, local_rank: {local_rank}")
        
        tensor = torch.zeros(1, device=device)
        dist.all_reduce(tensor)  # force NCCL init
        if rank == 0:
            print(f"[Distributed] Initialized {world_size} processes with backend '{backend}'.")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("[Distributed] Running in single-process (non-distributed) mode.")
    
    return device


def cleanup_distributed():
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
        
def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0
        
        

def calc_total(info_list, key='t_total'):
    # Need to iterate over each training data
    return sum([info[key] for info in info_list])


def select_results(to_select, all_plot_details):
    return {name:all_plot_details[name] for name in to_select}

#%% Reconstructing training dataset
def sum_state_keys(setup, all_logs):
    max_its = setup['solver_optns']['ul_solve']['max_its']
    to_summarise = ['stop_it_ll', 't_ll', 'stop_it_hess', 't_hess', 't_total']
    summary = {key:torch.zeros(max_its) for key in to_summarise}
    for data_id, logs in all_logs.items():
        for key, log in logs.items():
            if key in to_summarise:
                summary[key] += torch.tensor(log[:max_its])
    return summary

def foe_summands(x, params, ll_fun, device):
    filters = params[:,:-1].to(device)
    reg_params = params[:,-1].to(device)
    num_filter, tmp_length = filters.shape
    filter_size = int((tmp_length)**.5)
    cross_kernel = filters.view(num_filter,1,filter_size, filter_size)
    conv_kernel = torch.flip(cross_kernel, [2,3])
    
    expert = ll_fun.R.expert
    conv_op = torch.nn.Conv2d(1,num_filter, kernel_size=filter_size,padding='same', bias=False, device=device)
    conv_op.weight = torch.nn.Parameter(conv_kernel, requires_grad=False)
    
    x_conv = conv_op(x.to(device).unsqueeze(0).unsqueeze(0)).detach()
    return  (expert(x_conv) * 10**reg_params ).cpu()


def get_lower_level_solver(setup, ll_fun, shape, device):
    def construct_recon(params, A, y):
        recon, info = solve_lower_level_singleton(params.to(device), A, y.to(device), ll_fun,
                                            x0=torch.zeros(shape).to(device),
                                            xshape = shape,
                                            grad_tol = setup['solver_optns']['ll_solve']['tol'],
                                            max_its = setup['solver_optns']['ll_solve']['max_its'],
                                            num_store=setup['solver_optns']['ll_solve']['num_store']
                                            ,verbose=False)
        return recon
    return construct_recon

def reconstruct_dataset(param, ll_solver, xexacts, As, ys, save=None, desc=None):
    recons = []
    for ind in tqdm(range(len(ys)), desc=desc):
        recons.append( ll_solver(param, As[ind], ys[ind]) )
    psnrs = calculate_psnrs(recons, xexacts)
    if isinstance(save,str):
        torch.save([recons,psnrs], save)
    return recons, psnrs

def psnr(x, xstar, max_pixel=1):
    sqrtmse = torch.norm(xstar-x).cpu()/xstar.numel()**.5
    psnr = 20 * torch.log10(max_pixel / sqrtmse)
    return psnr
def calculate_psnrs(As,Bs):
    out = torch.empty(len(As))
    for i in range(len(As)):
        out[i] = psnr(As[i], Bs[i])
    return out