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
        

        
            
        
class IdentityOperator(LinearOperator):
    def __init__(self):
        super().__init__()
        
    def __call__(self, x):
        return x
    
    def T(self, z):
        return z
    
    
#%%
torch.manual_seed(0)
n = 5
x = torch.randn(n,n) 

mask = torch.zeros_like(x, dtype=bool)
mask[:,0] = 1 # Keep only first column 

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
            
    def __call__(self, x):
        return x[self.mask].unsqueeze(0)
    
    def T(self, z):
        out = torch.zeros_like(self.mask, dtype=z.dtype)
        out[self.mask] = z
        return out

inpaint = Inpainting(mask, x.shape)
xinpaint = inpaint(x)
xpadinpaint = inpaint.T(xinpaint)
    
#%%
class MatrixOperator(LinearOperator):
    def __init__(self, A):
        super().__init__()
        self.matrix = A
        
    def __call__(self, x):
        return self.matrix@x
    
    def T(self, z):
        return self.matrix.T@z
    


class FiniteDifference2D(LinearOperator):
    '''
    Linear operator taking horizontal and vertical first order finite difference
    with zero boundary conditions and returns the concaternated result.
    
    The Vertical derivative matrix with zero boundary conditions is given by
    D = \begin{matrix}
     1  0  0 ...  0  0
    -1  1  0 ...  0  0
     0 -1  1 ...  0  0
      ...    ...   ...
     0  0  0 ... -1  1
     0  0  0 ... 0  -1
    \end{matrix}
    '''
    def __init__(self):
        super().__init__()
    
    def __call__(self, x):
        xhoriz_pad = torch.nn.functional.pad(x, (1,1))
        xvert_pad = torch.nn.functional.pad(x.T, (1,1))
        vert_dif = (xvert_pad[:,1:] - xvert_pad[:,:-1]).T   # D @ x
        horiz_dif = xhoriz_pad[:,1:] - xhoriz_pad[:,:-1]    # x @ D.T
        return torch.cat([vert_dif, horiz_dif.T])
    
    def T(self, z):
        _, n = z.shape 
        zvert_transp = z[:n] - z[1:n+1] # Transpose of vertical derivative operator
        ztmp = z[n+1:].T
        zhoriz_transp = ztmp[:,:-1] - ztmp[:,1:] # Transpose of horizontal derivative operator
        return zvert_transp + zhoriz_transp
    

    
#%% Forward operator

def create_gaussian_filter(sigma=1, device=None):
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
    def __init__(self,sigma=1, device=None):
        super().__init__()
        self.filter = create_gaussian_filter(sigma,device=device) # Symmetric so don't need to flip
        self.device = device
        
        self.conv_op = torch.nn.Conv2d(1,1,kernel_size=self.filter.shape,padding='same', bias=False, device=device)
        self.conv_op.weight = torch.nn.Parameter(self.filter.unsqueeze(0).unsqueeze(0),requires_grad=False)
        
        self.conv_transp_op = torch.nn.ConvTranspose2d(1,1,kernel_size=self.filter.shape, padding_mode='zeros', bias=False)
        self.conv_transp_op.weight = torch.nn.Parameter(self.filter.unsqueeze(0).unsqueeze(0),requires_grad=False)
        
    def __call__(self, x):
        return self.conv_op(x.unsqueeze(0)).detach()[0]
    
    def T(self, z):
        p = int((self.filter.shape[0]-1)/2) # Amount of padding
        return self.conv_transp_op(z.unsqueeze(0).unsqueeze(0)).detach()[0][0][p:-p,p:-p] # Zero boundary condition



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
    
    
