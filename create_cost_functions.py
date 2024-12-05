# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 11:04:55 2024

@author: sebsc

Generate data and cost functions
"""

import torch
import torchvision

from linear_ops import IdentityOperator, GaussianBlur, FiniteDifference2D, Inpainting
from functionals import upper_level, lower_level, FieldOfExperts, Huber, TikhonovPhillips, Tikhonov, HuberTV

# from PIL import Image as image

from torchvision.transforms import v2



#%% Create / load dataset
def create_forward_op(forward_op='identity', sigma=1., device=None, mask=0.7, tensor_size=None):
    if forward_op=='identity':
        return IdentityOperator()
    
    elif forward_op =='gaussian_blur':
        return GaussianBlur(sigma=sigma, device=device)
    
    elif forward_op =='inpainting':
        return Inpainting(mask=mask, tensor_size=tensor_size, device=device)
    
    else: raise ValueError('Chocie of forward operator "{}" not recognised.'.format(forward_op))
    

def create_dataset(n=128,ground_truth='geometric', data_num=1, noiselevel=0.1, A=None, device=None):
    # Currently we create and store the entire dataset at all times
    
    if A is None: A = create_forward_op('identity')
    if isinstance(noiselevel, float): noiselevel = torch.ones(data_num, device=device)*noiselevel
    else: assert len(noiselevel) == data_num
    
    xexacts = torch.empty(data_num, n, n, device=device)
    
    if ground_truth=='geometric':
        tmp_xexact = torch.zeros(n,n,device=device)
        tmp_xexact[int(n/2):] += .5
        tmp_xexact.T[int(n/2):] += .5
        xexacts[:] = tmp_xexact
    elif ground_truth =='MNIST':
        transforms = torchvision.transforms.Compose(
        [v2.ToImage() , v2.ToDtype(torch.float32), v2.Resize(n) ])
        mnist_full = torchvision.datasets.MNIST(root='data', train=True, download=True, transform=transforms)
        for img_ind in range(data_num):
            xexacts[img_ind] = mnist_full[img_ind][0][0].to(device)/255
    elif ground_truth =='FashionMNIST':
        transforms = torchvision.transforms.Compose(
        [v2.ToImage() , v2.ToDtype(torch.float32), v2.Resize(n) ])
        mnist_full = torchvision.datasets.FashionMNIST(root='data', train=True, download=True, transform=transforms)
        for img_ind in range(data_num):
            xexacts[img_ind] = mnist_full[img_ind][0][0].to(device)/255
    elif ground_truth =='ruby':
        transforms = torchvision.transforms.Compose(
        [v2.PILToTensor(), v2.ToDtype(torch.float32), v2.Resize(n), v2.Grayscale() ])
        dataset = torchvision.datasets.ImageFolder(r"./data/DogPictures/",transform=transforms)
        for img_ind in range(data_num):
            # There is only one image in the folder
            xexacts[img_ind] = dataset[0][0][0].to(device)/255
    
    y_shape = A(xexacts[0]).shape
    ys = torch.empty(data_num, y_shape[0], y_shape[1], device=device) #!!! abuse denoising setting
    for ind in range(data_num):
        noise = torch.randn(y_shape,device=device)
        ys[ind] = A(xexacts[ind]) + noise*noiselevel[ind]*A(xexacts[ind]).norm()/noise.norm()
        
    return xexacts, ys

# from torch.utils.data import Dataset, DataLoader

# class BilevelDataset(Dataset):
#     def __init__(self, xexacts, ys):
#         self.xexacts = xexacts
#         self.ys = ys
#         self.img_dim = xexacts[0].shape
#     def __len__(self):
#         return len(self.xexacts)
#     def __getitem__(self, idx):
#         return self.xexacts[idx], self.ys[idx]

# n = 64
# transforms = torchvision.transforms.Compose(
# [v2.PILToTensor(), v2.ToDtype(torch.float32), v2.Resize(n), v2.Grayscale() ])
# dataset = torchvision.datasets.ImageFolder(r"./data/DogPictures/",transform=transforms)
# print(dataset[0][0].shape)

# print(torch.tensor(dataset[0][0]))

def create_regulariser(regulariser='FieldOfExperts', filter_shape=5, filter_num=1, expert='l2',device=None, gamma=0.01, L = 'FiniteDifference2D'):
    if regulariser == 'Tikhonov':
         return Tikhonov(device=device)
    elif regulariser == 'Huber':
        return Huber(device=device, gamma=gamma)
    elif regulariser == 'HuberTV':
       return HuberTV(device=device, gamma=gamma)
    elif regulariser == 'FieldOfExperts':
        return  FieldOfExperts(device=device, filter_shape=filter_shape, expert=expert,filter_num=filter_num)
    elif regulariser == 'TikhonovPhillips':
        if L == 'Identity':
            Lop = IdentityOperator()
        elif L == 'FiniteDifference2D':
            Lop = FiniteDifference2D()
        else:
            raise Exception('Choice of L operator "{}" not recognised'.format(L))
        return TikhonovPhillips(device=device, L=Lop)
    
    else:
        raise Exception('Name of regulariser not recognised')


def populate_reg_options(options):
    r"""
    Create a dictionary of all options of the regulariser 
    where unspecified options have been included
    """
    defaults = {
               'name':'FieldOfExperts',
                'filter_shape':3,
                 'filter_num':3,
                    'expert':'huber',
              'gamma':.1,
              'L':'Identity',
              'eps':0}
        
    return {**defaults, **options}

def create_cost_functions(A,xexacts,optns,device=None, y=None):
    
    optns = populate_reg_options(optns)
    
    R = create_regulariser(regulariser=optns['name'], filter_shape=optns['filter_shape'],
                           filter_num=optns['filter_num'],
                           expert=optns['expert'],device=device, gamma=optns['gamma'],
                           L=optns['L'])
    
    ll_fun = lower_level(A,R,xexacts.shape[1:], y=y, eps=optns['eps'])
    ul_fun = upper_level(xexacts)
    
    return ul_fun, ll_fun

def create_problem(optns,device=None):
    A = create_forward_op(optns['forward_op'])
    xexact, y =  create_dataset(n=optns['n'], ground_truth=optns['ground_truth'], A=A, data_num=optns['data_num'],
                                         noiselevel=optns['noiselevel'],device=device)
    
    ul_fun, ll_fun = create_cost_functions(A,xexact,y, optns=optns['regulariser'],device=device)
    return ul_fun, ll_fun, xexact, y, A
