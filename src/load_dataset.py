# -*- coding: utf-8 -*-
"""
Created on Thu Oct  2 14:59:30 2025

@author: sebsc
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct  2 13:16:36 2025

@author: sebsc
"""

import torch

from torch.utils.data import DistributedSampler
import torch.distributed as dist

import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import v2

from src.linear_ops import IdentityOperator, GaussianBlur, Inpainting

from torchvision.datasets import MNIST, FashionMNIST
from torchvision.datasets.folder import default_loader

def create_forward_op(forward_op='identity',kernel_size=5, sigma=1., device=None, mask=0.7, xshape=None, seed=None):
    
    if seed is not None:
        torch.manual_seed(seed)
        
    if forward_op=='identity':
        return IdentityOperator()
    
    elif forward_op =='gaussian_blur':
        return GaussianBlur(kernel_size=kernel_size, sigma=sigma, device=device)
    
    elif forward_op =='inpainting':
        return Inpainting(mask=mask, tensor_size=xshape, device=device)
    
    else: raise ValueError('Chocie of forward operator "{}" not recognised.'.format(forward_op))
    
def make_forward_op(setup, xshape, seed):
    return create_forward_op(setup['forward_op'], sigma=setup['sigma'],
                             mask=setup['mask'], xshape=xshape, seed=seed)


class InverseProblemDataset(Dataset):
    def __init__(self, dataset_name, root_dir, split='train',
                 transform=None, problem_setup=None, download=True):
        """
        Unified Dataset for inverse problems on MNIST / FashionMNIST / BSDS300.

        Args:
            dataset_name (str): One of 'MNIST', 'FashionMNIST', or 'BSDS300'.
            root_dir (str): Root directory for dataset.
            split (str): 'train', 'val', or 'test'.
            transform (callable): Transformations to apply to images.
            problem_setup (dict): Setings for inverse problem.
            download (bool): Whether to download dataset if missing.
        """
        self.dataset_name = dataset_name.lower()
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.problem_setup = problem_setup
        
        self.data_num = problem_setup['data_num']
        self.num_crops_per_image = problem_setup['num_crops_per_image']

        if self.dataset_name in ['mnist', 'fashionmnist']:
            self._load_mnist_like(download)
        elif self.dataset_name == 'bsds300':
            self._load_bsds300()
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

    def _load_mnist_like(self, download):
        dataset_cls = MNIST if self.dataset_name == 'mnist' else FashionMNIST
        train = self.split == 'train'
        self.dataset = dataset_cls(root=self.root_dir, train=train,
                                   transform=self.transform, download=download)
        self.dataset.data = self.dataset.data[:self.data_num]
        self.dataset.targets = self.dataset.targets[:self.data_num]
        

    def _load_bsds300(self):
        # Expecting folder structure: root_dir/split/*.jpg
        split_dir = os.path.join(self.root_dir, self.split)
        self.image_paths = sorted([
            os.path.join(split_dir, fname)
            for fname in os.listdir(split_dir)
            if fname.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])[:self.data_num]
        if not self.image_paths:
            raise RuntimeError(f"No images found in {split_dir}")
        self.loader = default_loader

    def __len__(self):
        if self.dataset_name in ['mnist', 'fashionmnist']:
            base_len = len(self.dataset)
        else:
            base_len = len(self.image_paths)
        return  base_len * self.num_crops_per_image

    def __getitem__(self, idx):
    
        if self.num_crops_per_image > 1:
            image_idx = idx // self.num_crops_per_image
            crop_idx = idx % self.num_crops_per_image
            unique_id = image_idx * self.num_crops_per_image + crop_idx
        else:
            image_idx = idx
            unique_id = idx
            seed = idx
        # print(f'img {image_idx} | crp {crop_idx} | id {unique_id}')
            
        if self.dataset_name in ['mnist', 'fashionmnist']:
            image, _ = self.dataset[image_idx]
        else:
            img_path = self.image_paths[image_idx]
            image = self.loader(img_path).convert('L')
            if self.transform:
                image = self.transform(image)  # ToTensor, Grayscale, etc.
        
        # Deterministic crop using fixed seed
        if self.split == 'train':
            # Manual deterministic crop calculation (non-random)
            # crop_size = self.problem_setup['n']
            # # print(image.shape)
            # img_channels, img_width, img_height = image.shape  # Assuming image is PIL.Image type
            
            # # Calculate the deterministic crop position based on unique_id
            # rows = (img_height - crop_size) // crop_size
            # cols = (img_width - crop_size) // crop_size
            
            # print(f'rows:{rows} | cols{cols}')
            
            # # Unique crop position based on unique_id
            # row = (unique_id // cols) % rows
            # col = unique_id % cols
            
            # # Compute the crop box
            # crop_box = (col * crop_size, row * crop_size, (col + 1) * crop_size, (row + 1) * crop_size)
            # image = torch.crop(image,crop_box)
            
            torch.manual_seed(unique_id)
            crop_size = self.problem_setup['n']
            crop = transforms.RandomCrop(crop_size)
            image = crop(image)       
        # Make A deterministic if it is not already
        if self.problem_setup['forward_op'] not in ['inpainting']:
            seed = None
        
        A = make_forward_op(self.problem_setup, image[0].shape, seed=seed)
        measurement = add_relative_noise(A(image),self.problem_setup['noiselevel'])


        return {
            'image': image,
            'forward_image': measurement,
            'forward_op': A , # optional use in custom collate_fn
            'idx':unique_id, # Stable identifier
        }

def add_relative_noise(y, sigma):
    C = y.size(0) # Number of colour channels
    noise = torch.randn_like(y)

    # Compute L2 norm of for each batch item
    # print(y.shape)
    # print(y)
    # print(y.view(C, -1))
    y_norms = torch.norm(y.reshape(C, -1), dim=1) 
    noise_norms = torch.norm(noise.view(C, -1), dim=1) + 1e-12  # avoid division by zero

    # Compute scaling factor per sample
    scales = sigma * y_norms / noise_norms  
    scales = scales.view(C, 1, 1) # Reshape to broadcast

    # Add scaled noise to clean measurements
    y_noisy = y + noise * scales

    return y_noisy

def custom_collate(batch):
    # Batch is a list of dicts with keys: image, forward_image, A
    # batch_collated = {key:([item[key] for item in batch]) for key in batch.keys() }
    batch_collated = {
        'image': ([item['image'] for item in batch]),
        'forward_image': [item['forward_image'] for item in batch],
        'forward_op': [item['forward_op'] for item in batch]  ,
        'idx': [item['idx'] for item in batch]
    }
    return batch_collated

def create_dataset(optns):
    
    n = optns['n']
    
    if optns['ground_truth'] == 'MNIST':
        trnsfrm = [v2.ToImage() , v2.ToDtype(torch.float32), v2.Resize(n) ]
        dataset_name = 'MNIST'
        root_dir = './data'

    elif optns['ground_truth'] == 'BSDS':
        trnsfrm = [transforms.ToTensor(), transforms.Grayscale()]
        dataset_name = 'BSDS300'
        root_dir = './data/BSDS300-images/BSDS300/images'
        
        
    train_dataset = InverseProblemDataset(
        dataset_name=dataset_name,
        root_dir=root_dir,
        split='train' if optns['train'] else 'test',
        transform=transforms.Compose(trnsfrm),
        problem_setup=optns
        )
    return train_dataset

def create_dataloader(train_dataset, batch_size=8,
                world_size=1, rank=0):
    # Prepare for distributed learning. Always distribute the training set but
    #    only distribute the validation and test set if specified.
    if dist.is_initialized():
        train_sampler =  DistributedSampler(train_dataset, 
                                                 num_replicas=world_size,
                                                 rank=rank)
        # batch_size = batch_size // world_size
        num_workers   = int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) 
        if rank == 0:  # Only main process prints
            print("="*60)
            print("Distributed Training Setup")
            print(f"  World size (total processes)  : {world_size}")
            print(f"  Effective global batch size   : {batch_size}")
            print(f"  Batch size per process        : {batch_size// world_size}")
            print(f"  Number of DataLoader workers  : {num_workers}")
            print("="*60)
    else:
        world_size     = 1
        train_sampler  = None
        num_workers    = int(os.environ.get("SLURM_CPUS_PER_TASK", "0"))
        print("="*60)
        print("Non-Distributed Training Setup")
        print(f"  Batch size                    : {batch_size}")
        print(f"  Number of DataLoader workers  : {num_workers}")
        print("="*60)
    persistent_workers = num_workers > 0
    
    train_loader = DataLoader(train_dataset, 
                              batch_size=batch_size // world_size, 
                              collate_fn=custom_collate,
                              shuffle=(train_sampler is None), 
                              sampler=train_sampler,
                              num_workers=num_workers,
                              pin_memory=True,
                              persistent_workers=persistent_workers)
    
    return train_loader