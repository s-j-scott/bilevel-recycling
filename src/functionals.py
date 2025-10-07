# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 21:55:04 2023

Implementation of various cost functions associated with solving inverse
    problems, including regularizers such as the Fields of Experts model.

@author: Sebastian J. Scott
"""

import torch
import numpy as np

from scipy.fftpack import idct

from src.linear_ops import IdentityOperator, FiniteDifference2D
#%%

class Functional:
    def __init__(self):
        pass

    def __call__(self, x):
        raise NotImplementedError()

    def grad(self, x):
        raise NotImplementedError()

    def hess(self, x, w):
        raise NotImplementedError()
        
    def grad_jac_adjoint(self,x,w):
        raise NotImplementedError()
    def grad_jac(self,x,z):
        raise NotImplementedError()
        
    def __add__(self, F2):
        F1 = self
        class Functional_sum(Functional):
            def __init__(self, F1, F2):
                super().__init__()
                self.F1 = F1
                self.F2 = F2

            def __call__(self, x):
                return self.F1(x) + self.F2(x)
    
            def grad(self, x):
                return self.F1.grad(x) + self.F2.grad(x)
    
            def hess(self, x, w):
                return self.F1.hess(x,w) + self.F2.hess(x,w)
                
            def grad_jac_adjoint(self,x,w):
                return self.F1.grad_jac_adjoint(x,w) + self.F2.grad_jac_adjoints(x,w)
            def grad_jac(self,x,z):
                return self.F1.grad_jac(x,z) + self.F2.grad_jac(x,z)
            
            def __repr__(self):
                return 'Sum of functionals (' +str(self.F1) +') and ('+ str(self.F2)+')'
        return Functional_sum(F1,F2)
    
    def __mul__(self,const):
        F = self
        class Functional_rescale(Functional):
            def __init__(self, F, const):
                super().__init__()
                self.F = F
                self.const = const

            def __call__(self, x):
                return self.const *  self.F(x)
    
            def grad(self, x):
                return self.const * self.F.grad(x)
    
            def hess(self, x, w):
                return self.const * self.F.hess(x,w) 
                
            def grad_jac_adjoint(self,x,w):
                return self.const *  self.F.grad_jac_adjoint(x,w) 
            def grad_jac(self,x,z):
                return self.const *  self.F.grad_jac(x,z) 
            
            def __repr__(self):
                return 'Rescaling of (' +str(self.F) +') by constant ('+ str(self.const)+')'
        return Functional_rescale(F,const)
    
    def __rmul__(self,const):
        return self * const


class upper_level(Functional):
    # def __init__(self):
        
    def __call__(self,x, xexacts):
        if isinstance(x,list):
            out = 0.0
            for i, xi in enumerate(x):
                out += torch.norm(xi - xexacts[i])**2 / 2        
        
        elif isinstance(x,dict):
            out = 0.0
            for i, data_id in enumerate(x.keys()):
                # Assume the order of the keys is the same as the order of the xexacts
                # print(f"Recon shape: {x[data_id]['recon'].shape} | xexact shape : {xexacts[i].shape}")
                out += torch.norm(x[data_id]['recon'] - xexacts[i])**2 / 2
        else:
            out = torch.norm(x - xexacts)**2 / x.size(0)    
        return out 
    
    def grad(self, x, xexacts):
        if isinstance(x,list):
            out = torch.zeros_like(x[0])
            for i, xi in enumerate(x):
                out += xi - xexacts[i]
            out /= len(x)
            
        elif isinstance(x,dict):
            for _, data_id in x.items():
                out = torch.zeros_like(x[data_id]['recon'][0])
                break
            
            for i, data_id in enumerate(x.keys()):
                # Assume the order of the keys is the same as the order of the xexacts
                out += torch.norm(x[data_id]['recon'] - xexacts[i])**2 / 2
        else:
            out = (x - xexacts) / x.size(0)
        return out
    
    def __repr__(self):
        return 'l2 squared upper level cost function' 


class lower_level(Functional):
    r""" Data fidelity of the 2 norm squared
    """
    def __init__(self, R, A=None, y=None, eps=0):
        super().__init__()
        self.A = A # Forward operator, Linear operator class
        self.R = R # Choice of regulariser
        self.y = y # Single noisy measurement
        self.eps = eps # Regularisation parameter for Tikhonov regularisation 
    
    @property
    def params(self):
        return self.R.params
    @params.setter
    def params(self, newparams):
        self.R.params = newparams
        
    def __call__(self,x):
        return 0.5* torch.norm(self.A ( x ) - self.y)**2 + self.R(x) + self.eps/2 * x.norm()**2
    
    def grad(self,x):
        return  self.A.T (  (self.A(x) - self.y)) + self.R.grad(x) + self.eps * x
    
    def hess(self,x,w):
        return self.A.T (self.A (w)) + self.R.hess(x,w) + self.eps * w
    
    def grad_jac_adjoint(self,x,w):
        return self.R.grad_jac_adjoint(x,w)
    def grad_jac(self,x,z):
        return self.R.grad_jac(x,z)
    
    def __repr__(self):
        '''Return ``repr(self)``.'''
        return 'Lower level cost function with regulariser '+str(self.R)




# %% Regularisers

class Huber(Functional):
    r""" Tikhonov regularisation, scaled by a regularisation parameter
    """
    def __init__(self,params=None,gamma=1e-2,device=None):
        super().__init__()
        if params is None: params = torch.tensor([1.],device=device)
        self.params = params # Regularisation parameter
        self.gamma = gamma # Smoothing parameter
        
    def __call__(self,x):        
        norm = torch.abs(x) #  May need to do pointwise
        
        ind = norm <= self.gamma
        
        out_summands = norm - self.gamma/3
        out_summands[ind] = norm[ind]**2/self.gamma - norm[ind]**3 / (3*self.gamma**2)
        
        return self.params * torch.sum(out_summands)
    
    def grad(self,x):
        out = torch.sign(x)
        
        ind = torch.abs(x) < self.gamma
        out[ind] = 2/self.gamma * x[ind] - x[ind]**2/self.gamma**2*out[ind]
        
        return  self.params * out
    
    def hess(self,x,w):
        h = torch.zeros_like(x)
        
        norm = torch.abs(x)
        ind = norm <= self.gamma
        
        h[ind] = 2/self.gamma - 2/self.gamma**2 * norm[ind]
        
        # Pointwise multiplication
        return self.params * h * w
    
    def grad_jac_adjoint(self,x,w):
        rho = torch.sign(x)
        ind = torch.abs(x) < self.gamma
        rho[ind] = 2/self.gamma * x[ind] - x[ind]**2/self.gamma**2*rho[ind]
        return rho.ravel().inner(w.ravel())
    
    def __repr__(self):
        '''Return ``repr(self)``.'''
        return 'Twice differentiable Huber norm with reg param {}.'.format(self.params)

        
class HuberTV(Functional):
    r""" Smooth total variation regularisation, scaled by a regularisation parameter
    """
    def __init__(self,params=None,L=None, device=None, gamma=0.01):
        super().__init__()
        if params is None: params = torch.tensor([1.],device=device)
        self.params = params # Regularisation parameter
        if L is None: L = FiniteDifference2D()
        self.L = L # Callable linear operator
        self.norm = Huber(1,device=device, gamma=gamma)
        
    def __call__(self,x):
        return self.params * self.norm(self.L(x))
    
    def grad(self,x):
        return  self.params * self.L.T(self.norm.grad(self.L(x)))
    
    def hess(self,x,w):
        return  self.params * self.L.T(self.norm.hess(self.L(x),self.L(w)))
    
    def grad_jac_adjoint(self,x,w):
        return self.L.T(self.norm.grad(self.L(x))).ravel().inner(w.ravel())
    
    def __repr__(self):
        '''Return ``repr(self)``.'''
        return 'Huber TV with reg param {}.'.format(self.params)

#%% Field of Experts Regulariser
class expert_l2(Functional):
    
    def __init__(self):
        super().__init__()
    def __call__(self,x):
        if len(x.shape)==4 and x.shape[1]==1:
            return 0.5*torch.norm(x)**2
        else:
            return 0.5*torch.norm(x, dim=[2,3])**2
            # return 0.5*torch.norm(x)**2
    def grad(self,x):
        return x
    def grad2(self,x):
        return torch.ones_like(x)
    
class expert_lorentzian(Functional):
    def __init__(self):
        super().__init__()
    def __call__(self,x):
        return torch.sum(torch.log(1+x**2))
    def grad(self,x):
        return 2*x / (1+x**2)
    def grad2(self,x):
        x2 = x**2
        return -2* (x2 - 1) / (1+x2)**2
    
class expert_huber(Functional):
    # Pointwise application of the Huber function. Derivatives are interpreted elementwise
    def __init__(self,gamma=0.01):
        super().__init__()
        self.gamma = gamma
    def __call__(self,x):
        # Evaluation of Huber function. Returns scalar
        norm = torch.abs(x) #  May need to do pointwise
    
        ind = norm <= self.gamma
        
        out = norm - self.gamma/3
        out[ind] = norm[ind]**2/self.gamma - norm[ind]**3 / (3*self.gamma**2)
        
        return torch.sum(out)
    
    def grad(self,x):
        #  Evaluation of gradient of Huber function. Returns vector
        out = torch.sign(x)
        
        ind = torch.abs(x) < self.gamma
        out[ind] = 2/self.gamma * x[ind] - x[ind]**2/self.gamma**2*out[ind]
        
        return  out
    def grad2(self,x):
        # Second derivative of Huber function applied elementwise. Returns vector.
        out = torch.zeros_like(x)
        
        norm = torch.abs(x)
        ind = norm <= self.gamma
        
        out[ind] = 2/self.gamma - 2/self.gamma**2 * norm[ind]
        
        return out

class expert_smooth_l1(Functional):
    # Pointwise application of the smoothed 1norm, given by
    # R(x) = sqrt(gamma**2 + x**2)
    def __init__(self,gamma=0.01):
        super().__init__()
        self.gamma2 = gamma**2
    def __call__(self,x):        
        return torch.sum(torch.sqrt(self.gamma2 + x**2))
    
    def grad(self,x):
        #  Evaluation of gradient 
        return  x / torch.sqrt(self.gamma2 + x**2)
    def grad2(self,x):
        # Second derivative 
        return self.gamma2 / torch.pow(self.gamma2 + x**2, 1.5)



def FoE(params,x,device=None, expert=None):
    r'''
    Field of Experts Regulariser

    Parameters
    ----------
    params : Torch tensor
        Convolution kernel
    x : Torch tensor
        Point to evaluate FoE regularizer at.

    Returns
    -------
    Evaluation of FoE regularizer.

    '''
    if expert is None:
        expert = expert_l2()
    
    if len(params.shape)==1: kernel = params.flip(0)
    else:  kernel = params.flip(0).flip(1) # Flip the kernel for a convolution as opposed to correlation of native pytorch conv
    conv_op = torch.nn.Conv2d(1,1,kernel_size=kernel.shape,padding='same', bias=False, device=device)
    conv_op.weight = torch.nn.Parameter(kernel.unsqueeze(0).unsqueeze(0),requires_grad=False)
    
    x_conv = conv_op(x.unsqueeze(0).unsqueeze(0)).detach()
    
    return expert(x_conv)

def FoE_grad(params,x,device=None, expert=None):
    r'''
    
    Gradient of Field of Experts Regulariser

    Parameters
    ----------
    params : Torch tensor
        Convolution kernel
    x : Torch tensor
        Point to evaluate FoE gradient at.

    Returns
    -------
    Torch tensor, Gradient of FoE regularizer.

    '''
    if expert is None:
        expert = expert_l2()
    
    kernel = params.flip(0).flip(1) # Flip the kernel to have a valid convolution
    
    # Do convolution and then evalate at gradient of expert function
    conv_op = torch.nn.Conv2d(1,1,kernel_size=kernel.shape,padding='same', bias=False,device=device)
    conv_op.weight = torch.nn.Parameter(kernel.unsqueeze(0).unsqueeze(0),requires_grad=False)
    evald_x_conv = expert.grad(conv_op(x.unsqueeze(0).unsqueeze(0)).detach())
    
    #Then do the cross correlation with that quantity
    cross_op = torch.nn.Conv2d(1,1,kernel_size=kernel.shape,padding='same', bias=False, device=device)
    cross_op.weight = torch.nn.Parameter(params.unsqueeze(0).unsqueeze(0),requires_grad=False)
    return cross_op(evald_x_conv).detach()

def FoE_hess(params,x, w, device=None, expert=None):
    r'''rgrad_true
    
    Hessian of Field of Experts Regulariser

    Parameters
    ----------
    params : Torch tensor
        Convolution kernel
    x : Torch tensor
        Point to evaluate FoE gradient at.

    Returns
    -------
    Torch tensor, Gradient of FoE regularizer.

    '''
    if expert is None:
        expert = expert_l2()
    kernel = params.flip(0).flip(1) # Flip the kernel for a convolution
    
    # Do convolution and then evaluate at second order derivative of expert function
    conv_op = torch.nn.Conv2d(1,1,kernel_size=kernel.shape,padding='same', bias=False, device=device)
    conv_op.weight = torch.nn.Parameter(kernel.unsqueeze(0).unsqueeze(0),requires_grad=False)
    
    evald_x_conv = expert.grad2(conv_op(x.unsqueeze(0).unsqueeze(0)).detach())
    w_conv = conv_op(w.unsqueeze(0).unsqueeze(0)).detach() 

    # Cross correlation of the elementwise product of the above quantities
    conv_op.weight = torch.nn.Parameter(params.unsqueeze(0).unsqueeze(0),requires_grad=False)
    
    out = conv_op(evald_x_conv * w_conv).detach()
    return out 


def FoE_mixed_deriv_adjoint(params,x,w, device=None, expert=None):
    ''' Vector product between w and adjoint Jacobian of FoE gradient evaluated at x'''
    
    if expert is None:
        expert = expert_l2()
    kernel = params.flip(0).flip(1) # Flip the kernel to have a valid convolution
    
    ##### First term - x cross correlation with (expert''(convolved_x) * convolved_w )
    # Do convolution and then evaluate at gradient of expert function
    conv_op = torch.nn.Conv2d(1,1,kernel_size=kernel.shape,padding='same', bias=False, device=device)
    conv_op.weight = torch.nn.Parameter(kernel.unsqueeze(0).unsqueeze(0),requires_grad=False)
    convolved_x = conv_op(x.unsqueeze(0).unsqueeze(0)).detach()
    g2_x_conv = expert.grad2(convolved_x)
    w_conv = conv_op(w.unsqueeze(0).unsqueeze(0)).detach()
    
    # Only perform cross correlation for the indices that won't be subsampled. Due to padding, only works for 2d images and not 1d signals
    adjoint_conv_op_sampled = torch.nn.Conv2d(1,1,kernel_size=x.shape,padding=int((params.shape[-1]-1)/2), bias=False, device=device)  
    adjoint_conv_op_sampled.weight = torch.nn.Parameter(g2_x_conv*w_conv,requires_grad=False)
    term1 = adjoint_conv_op_sampled(x.unsqueeze(0).unsqueeze(0)).detach()[0][0].flip(1).flip(0) 

    g1_x_conv = expert.grad(convolved_x[0][0])        
    adjoint_conv_op_sampled.weight = torch.nn.Parameter(g1_x_conv.unsqueeze(0).unsqueeze(0),requires_grad=False)
    term2 = adjoint_conv_op_sampled(w.unsqueeze(0).unsqueeze(0)).detach()[0][0].flip(0).flip(1)
    
    return term1 + term2


def FoE_mixed_deriv(params,x,z, device=None, expert=None):
    ''' Vector product between z and Jacobian of FoE gradient evaluated at x'''

    if expert is None:
        expert = expert_l2()
    kernel = params.flip(0).flip(1) # Flip the kernel to have a valid convolution
    
    ##### First term - cross correlation between filters and (expert''(convolved_x) * (x convolved with z) )
    # Calculate D_{zz}^2 J(Mx)
    conv_op = torch.nn.Conv2d(1,1,kernel_size=kernel.shape,padding='same', bias=False, device=device)
    conv_op.weight = torch.nn.Parameter(kernel.unsqueeze(0).unsqueeze(0),requires_grad=False)
    convolved_x = conv_op(x.unsqueeze(0).unsqueeze(0)).detach()
    g2_x_conv = expert.grad2(convolved_x)
    
    # Quantity that we will take cross correlation of, D_{zz}^2 J(Mx) \odot (x * z)
    conv_op.weight = torch.nn.Parameter(z.view(kernel.shape).flip(0).flip(1).unsqueeze(0).unsqueeze(0),requires_grad=False)
    to_be_crossed = g2_x_conv * conv_op(x.unsqueeze(0).unsqueeze(0)).detach()
    
    # Final cross correlation for the first term
    conv_op.weight = torch.nn.Parameter(params.unsqueeze(0).unsqueeze(0),requires_grad=False)
    term1 = conv_op(to_be_crossed).detach()[0][0]
    
    ##### Second term - convolution between z and expert'(convolved_x)
    g1_x_conv = expert.grad(convolved_x)
    conv_op.weight = torch.nn.Parameter( z.view(kernel.shape).flip(0).flip(1).unsqueeze(0).unsqueeze(0))
    term2 = conv_op(g1_x_conv).detach()[0][0]
        
    return  term1 + term2


def idct2(f):
    """2D Inverse Discrete Cosine Transform
    """
    return idct(idct(f, axis=0 , norm='ortho'), axis=1 , norm='ortho')

def create_dct_basis(N):
    """ 
    Construct basis of the Discrete Cosine Transformation
    """
    dct_basis = torch.zeros((N**2,N**2))
    k = 0
    for i in range(N):
        for j in range(N):
            g = np.zeros((N,N))
            g[i,j]=1
            dct_basis[k,:] = torch.tensor(idct2(g).flatten())
            k+=1
    return torch.nn.functional.normalize(dct_basis, dim=0) # Normalise each filter

class FieldOfExperts(Functional):
    r""" Field of experts regulariser functional. Zero boundary conditions used for convolution
    """
    def __init__(self,params=None, filter_num = 1, filter_shape = 5, device=None, expert='l2', gamma=0.01, learn_filters=True, learn_reg_params=True):
        super().__init__()
        
        # Whether to learn the convolution filters and/or the regularisation parameters
        self.learn_filters = learn_filters
        self.learn_reg_params = learn_reg_params
        
        # Specify convolution kernel and regularisation parameters
        if params is None or params == 'Normal': 
            # Final value of each row will be the associated regularisation parameter
            
            # Gaussian init
            params_tmp = torch.zeros(filter_num, filter_shape**2 + 1,device=device)
            params_tmp = torch.normal(params_tmp)
            params_tmp[:, -1] = -1*torch.ones(filter_num,device=device) # Regularisation parameter exponent
            self.params = params_tmp
            
        elif params == 'DCT':
            # Basis of Discrete Cosine Transformation
            dct_basis  = create_dct_basis(filter_shape)
            filter_num = filter_shape**2 - 1 # Do not use constant filter
            params_tmp = torch.zeros(filter_num, filter_shape**2+1, device=device)
            params_tmp[:,:-1] = dct_basis[1:,:]
            params_tmp[:, -1] = -1*torch.ones(filter_num,device=device) # Regularisation parameter exponent
            self.params = params_tmp
        else: 
            self.params = params
            filter_num, params_len = params.shape
            filter_shape = int( np.round((params_len-1)**.5)) # Account for regularisation parameter
        
        self.expert_name = expert
        if   expert == 'l2'        : expert = expert_l2()
        elif expert == 'huber'     : expert = expert_huber(gamma=gamma)
        elif expert == 'smooth_l1' : expert = expert_smooth_l1(gamma=gamma)
        elif expert == 'lorentzian': expert = expert_lorentzian()
        else:
            raise NotImplementedError("Only 'l2', 'huber', 'lorentzian', and 'smooth_l1' expert functions implemented")
        self.expert = expert # Choice of expert function
        
        self.device=device
        self.filter_num = filter_num
        self.filter_shape = filter_shape
        
    def _get_kernel(self,k):
        # Return the parameters associated with the kth convolution kernel
        return  self.params[k][:-1].view(self.filter_shape, self.filter_shape) 
        
    def __call__(self,x):
        out = torch.zeros(1,device=self.device)
        for filter_ind in range(self.filter_num):
            out += FoE(self._get_kernel(filter_ind), x, expert=self.expert, device=self.device) * 10**self.params[filter_ind][-1]
        return out
    
    def grad(self,x):
        out = torch.zeros_like(x)
        for filter_ind in range(self.filter_num):
            out += FoE_grad(self._get_kernel(filter_ind), x, expert=self.expert, device=self.device)[0][0] * 10**self.params[filter_ind][-1]
        return  out
    
    def hess(self,x,w):
        out = torch.zeros_like(x)
        for filter_ind in range(self.filter_num):
            out += FoE_hess(self._get_kernel(filter_ind), x, w, expert=self.expert, device=self.device)[0][0] * 10**self.params[filter_ind][-1]
        return out

    def grad_jac_adjoint(self,x,w):
        out = torch.zeros_like(self.params)
        
        # Calculate value with respect to the Filters
        for filter_ind in range(self.filter_num):
            reg_p = 10**self.params[filter_ind][-1] 
            foe_filter = self._get_kernel(filter_ind)
            if self.learn_filters:
                out[filter_ind][:-1] +=  FoE_mixed_deriv_adjoint(foe_filter, x, w,device=self.device, expert=self.expert).ravel() * reg_p
            if self.learn_reg_params:
                # Account for the regularisation parameters
                grad_w_dot = torch.dot(FoE_grad(foe_filter, x, expert=self.expert)[0][0].ravel() , w.ravel())  
                out[filter_ind][-1] += grad_w_dot * reg_p  * np.log(10)
        
        return out
    
    def grad_jac(self,x,z):
        out = torch.zeros_like(x)
        
        for filter_ind in range(self.filter_num):
            reg_p = 10**self.params[filter_ind][-1] 
            foe_filter = self._get_kernel(filter_ind)
            if self.learn_filters:
                # With respect to convolution filters
                z_filter_part = z[filter_ind,:-1]
                out += FoE_mixed_deriv(foe_filter, x, z_filter_part,device=self.device, expert=self.expert) * reg_p
            if self.learn_reg_params:
                # With respect to the regularisation parameter
                out += FoE_grad(foe_filter, x, expert=self.expert)[0][0] * z[filter_ind, -1] * reg_p * np.log(10)
        
        return out
  
    def __repr__(self):
        '''Return ``repr(self)``.'''
        return 'Field of Experts functional. {} filters of width {} expert {}.'.format(self.filter_num , self.filter_shape, self.expert_name)