# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 22:12:16 2023

Forward operators for inverse problems. Identity and Inpainting.

@author: Sebastian J. Scott
"""

import torch


class LinearOperator:
    def __init__(self):
        pass

    def __call__(self, x):
        raise NotImplementedError()

    def T(self, z):
        raise NotImplementedError()
        
    def to(self,device):
        return self
        

class IdentityOperator(LinearOperator):
    def __init__(self):
        super().__init__()
        
    def __call__(self, x):
        return x
    
    def T(self, z):
        return z
    
   
class Inpainting(LinearOperator):
    def __init__(self,mask, tensor_size,  device=None):
        super().__init__()
        if isinstance(mask, torch.Tensor):
            self.mask = mask > 0
        elif isinstance(mask, float):
            # Intepret mask as a subsampling rate
            mask_randn = torch.randn(tensor_size, device=device)
            to_be_mask = torch.ones_like(mask_randn, dtype=bool)
            to_be_mask[mask_randn<mask] = 0
            self.mask = to_be_mask
            
        self.device = device
            
    def to(self,device):
        self.device = device
        self.mask = self.mask.to(device)
        return self
    
    def __call__(self, x):
        if len(x.shape) == 3:
            return x[0][self.mask].unsqueeze(0)
        else:
            return x[self.mask].unsqueeze(0)
    
    def T(self, z):
        out = torch.zeros_like(self.mask, dtype=z.dtype)
        out[self.mask] = z
        return out

class MatrixOperator(LinearOperator):
    def __init__(self, A):
        super().__init__()
        self.matrix = A
        
    def __call__(self, x):
        return self.matrix@x
    
    def T(self, z):
        return self.matrix.T@z

class FiniteDifference2D:
    '''
    Linear operator computing vertical and horizontal first-order finite differences
    with zero boundary conditions. The result is a tensor of shape (2, H, W),
    where:
        - output[0, :, :] = vertical differences (D @ x)
        - output[1, :, :] = horizontal differences (x @ D^T)
    '''
    def __call__(self, x):
        """
        Args:
            x: torch.Tensor of shape (H, W)
        Returns:
            torch.Tensor of shape (2, H, W)
        """
        H, W = x.shape

        # Vertical differences (along rows)
        vert_diff = torch.zeros_like(x)
        vert_diff[1:, :] = x[1:, :] - x[:-1, :]

        # Horizontal differences (along columns)
        horiz_diff = torch.zeros_like(x)
        horiz_diff[:, 1:] = x[:, 1:] - x[:, :-1]

        return torch.stack([vert_diff, horiz_diff], dim=0)

    def T(self, z):
        """
        Adjoint (transpose) of the finite difference operator.
        Args:
            z: torch.Tensor of shape (2, H, W) — result of __call__()
        Returns:
            torch.Tensor of shape (H, W)
        """
        vert, horiz = z[0], z[1]
        H, W = vert.shape

        # Adjoint of vertical differences
        vert_adj = torch.zeros_like(vert)
        vert_adj[:-1, :] -= vert[1:, :]
        vert_adj[1:, :] += vert[1:, :]

        # Adjoint of horizontal differences
        horiz_adj = torch.zeros_like(horiz)
        horiz_adj[:, :-1] -= horiz[:, 1:]
        horiz_adj[:, 1:] += horiz[:, 1:]

        return vert_adj + horiz_adj


def create_gaussian_filter(kernel_size=None, sigma=1, device=None):
    '''
    Create filter to be used by a Gaussian blur with standard deviation sigma
    
    g(x,y) =  exp(-(x^2 + y^2)/(2*sigma^2)) / 2 pi sigma^2
    
    Parameters
    ----------
    sigma : Standard deviation of Gaussian filter


    '''
    c =  int(sigma / 0.3 + 1)
    filter_size = 2* c + 1
    
    delta = torch.arange(filter_size)
    
    x, y = torch.meshgrid(delta, delta, indexing="ij")
    x = x - c
    y = y - c
    filt = (x / sigma).pow(2)
    filt += (y / sigma).pow(2)
    filt = torch.exp(-filt / 2.0)

    filt = filt / filt.flatten().sum() # Normalise

    return filt.to(device)
    

class GaussianBlur(LinearOperator):
    def __init__(self, kernel_size=5, sigma=1, device=None):
        super().__init__()
        self.filter = create_gaussian_filter(kernel_size,sigma,device=device) # Symmetric so don't need to flip
        self.device = device
        
        self.conv_op = torch.nn.Conv2d(1,1,kernel_size=self.filter.shape,padding='same', bias=False, device=device)
        self.conv_op.weight = torch.nn.Parameter(self.filter.unsqueeze(0).unsqueeze(0).to(device),requires_grad=False)
        
    def to(self, device):
        if isinstance(device, int):
            if device == 32:
                # Convert to float32 and reassign as Parameter
                self.conv_op.weight = torch.nn.Parameter(self.conv_op.weight.to(torch.float32), requires_grad=False)
            else:
                # Convert to float64 and reassign as Parameter
                self.conv_op.weight = torch.nn.Parameter(self.conv_op.weight.to(torch.float64), requires_grad=False)
        else:
            self.device = device
            self.conv_op = self.conv_op.to(device)

        return self

    def __call__(self, x):
        return self.conv_op(x.unsqueeze(0)).detach()[0]
    
    def T(self, z):# Symmetric
        return self.conv_op(z.unsqueeze(0)).detach()[0]


def construct_matrix(op, n, device=None):
    '''

    Parameters
    ----------
    op : Callable function representing multiplication of a square n-by-n matrix
    n : Width of square matrix

    Returns
    -------
    mat : Tensor representation of op

    '''
    mat = torch.empty(n,n,device=device)
    for ind in range(n):
        basis_vec = torch.zeros(n, device=device)
        basis_vec[ind] = 1
        mat[:,ind] = op(basis_vec)
        
    return mat