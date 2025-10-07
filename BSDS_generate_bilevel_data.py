# -*- coding: utf-8 -*-
"""

Learn optimal filters of a fields of experts regularizer via bilevel learning.
    
  
@author: Sebastian J. Scott

"""

import torch
import torch.distributed     as dist
import torch.multiprocessing as mp

from tqdm import tqdm
from time import time

import json
import os
import sys

import src.utils    as utils
import src.plotting as pltt
from   src.bilevel        import bilevel_fun
from   src.load_dataset   import create_dataset, create_dataloader
from   src.configurations import populate_solver_options

from src.create_cost_functions import create_cost_functions 

#%% Create and save details of the problem
def main(rank=0, world_size=1):
    torch.no_grad()
    setup = {
            # Specify inverse problem and cost function
            'problem_setup': 
                {
                'ground_truth':'BSDS',
                # 'ground_truth':'MNIST',
                'data_num':64,
                'num_crops_per_image':8,
                'batch_size':16,
                
                'float':64,
                
                    # 'forward_op': 'identity',
                    # 'forward_op': 'inpainting',
                    'forward_op': 'gaussian_blur',
                    'sigma':3., 
                    'mask':.7, # Proportion of pixels that are removed with inpainting
                    'n':64, # For MNIST must be 28
                    'noiselevel':0.2,
                    'multiple_As':True,
                    'train':True,
            
                    'regulariser':
                        {
                        'name':'FieldOfExperts',
                        'filter_shape':5,
                        'filter_num':25-1,
                        # 'expert':'l2',
                        'expert':'lorentzian',
                        'gamma':.1,
                        'eps':1e-6,
                        'params':'DCT',
                        'learn_filters':True,
                        'learn_reg_params':True
                        }
              },
                    'solver_optns': populate_solver_options(
                        {
                            'ul_solve':
                                {
                                'max_its': 50,
                                'tol': 1e-2,
                                'backtrack':False,
                                'step_size':1e-2,
                                'solver':'Adam'
                                
                                },
                            'll_solve':
                                {
                                'verbose':False, 
                                'warm_start':True, 
                                'max_its':16000,
                                'tol': 1e-3,
                                'solver':'L-BFGS',
                                'store_iters':False
                               ,'num_store':10
                               },
                            'hess_sys':
                                {
                                'warm_start':True,
                                'tol': 1e-3,
                                'max_its':16000,
                                'solver':'MINRES',
                                'recycle_strategy':None
                                }
                            })}
                    
    
    if setup['problem_setup']['float'] == 64:
        torch.set_default_dtype(torch.float64)
    else:
        torch.set_default_dtype(torch.float32)
    
    RESULTS_DIR = './'
    FIGURES_DIR = './'
    
    utils.set_seed(0) # Reproducability
    dataset = create_dataset(setup['problem_setup'])
    
    on_hpc = True
    
    training(rank, world_size, setup, dataset, FIGURES_DIR, RESULTS_DIR, on_hpc)
    
    
def training(rank, world_size, setup, dataset, FIGURES_DIR, RESULTS_DIR, on_hpc=False):
    # device = utils.set_device(verbose=True)
    device = utils.ddp_setup(rank, world_size)
    distributed = dist.is_initialized()
    
    # In Distributed learning only one process should print and log
    is_main_process = utils.is_main_process()
    if is_main_process:
        with open(RESULTS_DIR+'setup.json', 'w') as f:
            json.dump(setup, f, indent=6)
        
    # Reproducability
    utils.set_seed(0, deterministic=True)
    
    
#%% Create the bilevel problem

    train_dataloader = create_dataloader(dataset, 
                                         batch_size=setup['problem_setup']['batch_size'],
                                         world_size=world_size,
                                         rank=rank,)

#%% Do training
    bl_fun = bilevel_fun(setup,device)
    if distributed:
        bl_fun = torch.nn.parallel.DistributedDataParallel(bl_fun, device_ids=[rank])
    
    optimizer = torch.optim.Adam(bl_fun.parameters(), lr=setup['solver_optns']['ul_solve']['step_size'])
    lr_scheduler = None
    
    total_epochs = setup['solver_optns']['ul_solve']['max_its']
    
    ul_history = []
    ul_history_epoch = []
    gnorm_history = []
    params_history = []
    steplength_history = []
    
    all_warm_starts = {}
    
    all_logs = {}
    
    
    # -------------  Training loop ----------------
    if is_main_process:
        pltt.display_filters(bl_fun.module.params.detach().cpu() if distributed else bl_fun.params.detach().cpu(),
                             save=FIGURES_DIR+'plot_filters_init.png', square=False)
        pbar = tqdm(total=total_epochs, desc='Training', file=sys.stdout, dynamic_ncols=False) 

    step = 0
    for epoch in range(total_epochs):
        if distributed:
            train_dataloader.sampler.set_epoch(epoch)
            
        t_start = time()
        
        total_loss  = torch.tensor(0, device=device, dtype=torch.float32)
        total_items = torch.tensor(0, device=device, dtype=torch.float32)
                
        for batch_idx, batch in enumerate(train_dataloader):
            if distributed:
                dist.barrier()
            batch = utils.to_device(batch, device)
            bl_fun_use = bl_fun.module if distributed else bl_fun
            
            xexacts = batch['image']
            ys = batch['forward_image']
            As = batch['forward_op']
            As = [A.to(setup['problem_setup']['float']) for A in As]
            idxs = batch['idx']
            
            # Get the warm starts relevant to the batch
            warm_starts = {i: all_warm_starts.get(i, {}) for i in idxs}
            
            # -------------------------------------------------------
            # -----  Solve lower level and evaluate upper level -----
            ul_eval = bl_fun(xexacts, As, ys, idxs, warm_starts)
            
            if distributed:
                dist.barrier()
            
            # -------------------------------------------------------
            # ----- Solve Hessian system and compute gradient  ------
            grad = bl_fun_use.grad(xexacts, As, ys, idxs, warm_starts, step)
            item_num = torch.tensor(len(As), device=device)
        
            
            # -------------------------------------------------------
            # -----   Accumulate information across devices   -------
            
            # Store information logged on current device
            batch_log = bl_fun_use.batch_log
            
            if distributed:
                dist.barrier()    
                dist.all_reduce(ul_eval,  op=torch.distributed.ReduceOp.SUM)
                dist.all_reduce(grad,     op=torch.distributed.ReduceOp.SUM)
                dist.all_reduce(item_num, op=torch.distributed.ReduceOp.SUM)
                
                gathered_warm_starts = [None for _ in range(world_size)]
                dist.all_gather_object(gathered_warm_starts, warm_starts)
                for w_s_ in gathered_warm_starts:
                    w_s_device = move_nested_dict_to_device(w_s_, device)
                    all_warm_starts.update(w_s_device)
                    
                gathered_logs = [None for _ in range(world_size)]
                dist.all_gather_object(gathered_logs, batch_log)
                for log in gathered_logs:
                    update_log(log, all_logs)
            else:
                all_warm_starts.update(warm_starts)
                update_log(batch_log, all_logs)
                
            # ------------------------------------------------------
            # -------------------   Logging   ----------------------

            if is_main_process:
                gnorm        = grad.norm().item()
                total_loss  += ul_eval * len(idxs)
                total_items += len(idxs)
                ul_history.append(ul_eval.item())
                params_history.append(bl_fun_use.params.detach().cpu())
                gnorm_history.append(gnorm)
                
                if lr_scheduler is not None:
                    steplength_history.append(lr_scheduler.get_last_lr()[0])
                
                end_batch_str = f"[{epoch}] ({batch_idx+1}/{len(train_dataloader)}) | Loss:{ul_eval.item():.3f} | Grad-norm:{gnorm:.3f}"
                utils.tqdm_compliant_print(end_batch_str, not on_hpc)
            
            # ----- Update the parameters -----
            optimizer.zero_grad()
            bl_fun_use.params.grad = grad
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()
            
            step += 1
        
        t_epoch = time() - t_start
        
        
        if is_main_process:
            avg_loss = total_loss.item() / total_items.item()
            ul_history_epoch.append(avg_loss)
            end_epoch_str = f"Epoch [{epoch}] Completed | Time: {(t_epoch):.2f}s | Average loss: {avg_loss:.3f}\n"
            utils.tqdm_compliant_print(end_epoch_str, not on_hpc)
            
            display_str = f'Epoch loss: {avg_loss:.3f}'
            pbar.set_postfix_str(display_str, refresh=False)
            pbar.update()
    
    utils.cleanup_distributed()
    
    #%% Performance on the test set
    test_setup = setup['problem_setup'].copy()
    if test_setup['forward_op'] == 'identity':
        test_setup['forward_op'] = 'gaussian_blur'
    test_setup['train']= False
    test_setup['data_num'] = 50
    
    torch.manual_seed(0)
    test_dataset = create_dataset(test_setup)
    test_dataloader = create_dataloader(test_dataset, batch_size = 1)
    
    #% Compute and save reconstructions of test dataset
    for batch in test_dataloader: 
        batch = utils.to_device(batch, device)
        y_test = batch['forward_image'][0][0]
        xexact_test = batch['image'][0][0]
        A_test = batch['forward_op'][0]
        break
    
    ul_fun, ll_fun = create_cost_functions(setup['problem_setup']['regulariser'],device=device)
    ll_solver = utils.get_lower_level_solver(setup, ll_fun, xexact_test.shape, device)
    recon_test = ll_solver(params_history[-1], A_test, y_test)
    psnr_test = utils.psnr(xexact_test, recon_test)
    
    recon0_test = ll_solver(params_history[0], A_test, y_test)
    psnr0_test = utils.psnr(xexact_test, recon0_test)
    psnry_test = utils.psnr(xexact_test, y_test)
    
    pltt.display_recons(xexact_test,[y_test, recon_test, recon0_test],
                   subtitles=[f'Measurement\n{psnry_test:.3f}', 
                              f'FoE Recon\n{psnr_test:.3f}', 
                              f'DCT Recon\n{psnr0_test:.3f}'],
                   save=FIGURES_DIR+'plot_test_reconstructions.png')
    
#%% Save important files
    if is_main_process:
        
        pltt.display_filters( bl_fun_use.params.detach().cpu(), square=False,
                      save=FIGURES_DIR+'plot_filters_learned.png')
        
        
        for i, logs in all_logs.items(): #!!!! temporary
            pltt.plot_data(logs['t_ll'], title=f'{i} LL time')
            pltt.plot_data(logs['t_hess'], title=f'{i} Hess time')
            pltt.plot_data(logs['stop_it_ll'], title=f'{i} LL stop its')
            pltt.plot_data(logs['stop_it_hess'], title=f'{i} Hess stop its')
            break
        
        torch.save([ul_history, gnorm_history, steplength_history], RESULTS_DIR+'history_ul_gnorm_step.pt')
        torch.save(params_history, RESULTS_DIR+'history_params.pt')
        
        torch.save(all_logs, 'all_logs.pt')
        
        #%% Display figures 
        state_sum = utils.sum_state_keys(setup, all_logs)
        pltt.plot_timings(state_sum, save=FIGURES_DIR+'plot_timings.png')
        

        pltt.plot_data(ul_history, xlabel='Training step', ylabel = 'Batch loss', 
                       save=FIGURES_DIR + 'plot_training_batch_loss.png')
        pltt.plot_data(ul_history_epoch, xlabel='Epoch', ylabel = 'Epoch loss', 
                       save=FIGURES_DIR + 'plot_training_epoch_loss.png')
        pltt.plot_data(gnorm_history, xlabel='Training step', ylabel = 'Batch grad norm', 
                       save=FIGURES_DIR + 'plot_training_gnorm.png')
        
        if lr_scheduler is not None:
            pltt.plot_data(steplength_history, xlabel='Training step', ylabel = 'Steplength', 
                        save=FIGURES_DIR + 'plot_steplength.png')
            
def move_nested_dict_to_device(nested_dict, device):
    """
    Recursively move all tensors in a dict of dicts to the given device.
    """
    def move(value):
        if isinstance(value, torch.Tensor):
            return value.to(device)
        elif isinstance(value, dict):
            return {k: move(v) for k, v in value.items()}
        else:
            return value  # Leave non-tensors unchanged

    return {k: move(v) for k, v in nested_dict.items()}

def update_log(batch_log, all_logs):
        for data_id, log in batch_log.items():
            try:
                for logged_key, logged_value in log.items():
                    all_logs[data_id][logged_key].append(logged_value)
            except:
                for data_id, log in batch_log.items():
                    all_logs[data_id] = {logged_key:[logged_value] for logged_key, logged_value in log.items()}


if __name__ == "__main__":    
    world_size = torch.cuda.device_count()
    # If more than one GPU is detected, train in parallel
    if world_size > 1:
        mp.set_start_method('spawn', force=True)
        os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
        mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
    else:
        main() # Implicitly use world_size 1

