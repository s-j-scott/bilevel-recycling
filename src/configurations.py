# -*- coding: utf-8 -*-
"""
Default options which will populate unspecified options in dictionaries.
"""

def select_problem(problem):
    
    if problem == 'inpainting':    
        
        raise NotImplementedError('Inpainting option not set up') #!!! Once settings decided, populate all options
        # pass
    elif problem == 'deblur':
        raise NotImplementedError('Deblur option not set up')
    
    
    elif problem == 'denoise':
        raise NotImplementedError('Denoising option not set up')


def populate_solver_options(options):
    r"""
    Default options to be used in solving a given
    bilevel problem. Populate unspecified options with said default values.
    """
    defaults = {
            'll_solve': {
                'solver': 'BFGS',
                'max_its': 10000,
                'tol': 1e-8,
                'warm_start' : True,
                'verbose' : False,
                'store_iters': False,
                'num_store': None,
                'full_info':False
            },
            'hess_sys': {
                'max_its': 10000,
                'tol': 1e-8,
                'warm_start' : True,
                'verbose' : False,
                'store_solns': False,
                'solver' : 'MINRES',
                'recycle_strategy' : None,
                'recycle_dim' : 5
            },
            'ul_solve': {
                'solver': 'GD',
                'max_its': 10000,
                'tol': 1e-8,
                'verbose' : True,
                'store_iters':True,
                'full_info':True
            }
        }
    
    # Populate unspecified main options
    out = {**defaults, **options}
    
    # Populate unspecified sub options
    for key in options:
        out[key] = {**defaults[key] , **options[key]}
    return out


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
                    'params':'Normal',
              'gamma':.1,
              'L':'Identity',
              'eps':0,
              'learn_filters':True,
              'learn_reg_params':True}
        
    return {**defaults, **options}