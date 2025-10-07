# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 11:04:55 2024

@author: sebsc

Generate data and cost functions
"""

from src.functionals import upper_level, lower_level, FieldOfExperts, Huber,  HuberTV

from src.configurations import populate_reg_options
from src.load_dataset import create_dataloader, create_dataset

def create_regulariser(regulariser='FieldOfExperts', filter_shape=5, filter_num=1, expert='l2',device=None, gamma=0.01, 
                       L = 'FiniteDifference2D', params='Normal', learn_filters=True, learn_reg_params=True):
    if regulariser == 'Huber':
        return Huber(device=device, gamma=gamma)
    elif regulariser == 'HuberTV':
       return HuberTV(device=device, gamma=gamma)
    elif regulariser == 'FieldOfExperts':
        return  FieldOfExperts(device=device, params=params, filter_shape=filter_shape, 
                               expert=expert,
                               filter_num=filter_num,
                               learn_reg_params=learn_reg_params,
                               learn_filters=learn_filters)
    
    else:
        raise Exception('Name of regulariser not recognised')


def create_cost_functions(reg_optns,device=None):
    
    reg_optns = populate_reg_options(reg_optns)
    
    R = create_regulariser(regulariser=reg_optns['name'], 
                           filter_shape=reg_optns['filter_shape'],
                           filter_num=reg_optns['filter_num'],
                           expert=reg_optns['expert'],
                           device=device, gamma=reg_optns['gamma'],
                           L=reg_optns['L'],
                           params=reg_optns['params'],
                           learn_filters = reg_optns['learn_filters'],
                           learn_reg_params=reg_optns['learn_reg_params'])
    
    ll_fun = lower_level(R, eps=reg_optns['eps'])
    ul_fun = upper_level()
    
    return ul_fun, ll_fun


def create_problem(optns, device, batch_size=8, num_workers=0):
    dataset = create_dataset(optns)
    dataloader = create_dataloader( batch_size, num_workers)
    ul_fun, ll_fun = create_cost_functions(reg_optns=optns['regulariser'],device=device)
    return ul_fun, ll_fun, dataloader


