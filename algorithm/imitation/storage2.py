from collections import deque
import numpy as np
import torch
import os

EPS = 1e-8 

class RolloutBuffer:
    def __init__(
            self, device:torch.device, 
            total_steps:int,
            batch_size:int) -> None:
        self.device = device
        self.total_steps = total_steps
        self.batch_size = batch_size

    ################
    # Public Methods
    ################

    def addBatch(self, states, actions):
            """
            Load data into memory. 
            We are now moving ALL data to GPU (VRAM) to save CPU RAM.
            Since VRAM is ~8GB and data is small enough for VRAM but big for System RAM issues, 
            this is the best strategy if VRAM > 1GB.
            """
            # Move directly to GPU tensors. This will free up system RAM as soon as the numpy arrays are garbage collected.
            self.states = torch.tensor(states, dtype=torch.float32, device=self.device)
            self.actions = torch.tensor(actions, dtype=torch.float32, device=self.device)
            
            # We no longer need numpy references in CPU memory.
            # The caller (main_il.py) should delete the original numpy arrays after calling this.
            
            self.splitData()

    def splitData(self, val_ratio=0.2):
        num_val = int(self.total_steps * val_ratio)
        num_train = self.total_steps - num_val
        
        # Use torch.randperm on GPU to keep indices on GPU/avoid CPU overhead
        indices = torch.randperm(self.total_steps, device=self.device)
        
        self.train_indices = indices[:num_train]
        self.val_indices = indices[num_train:]
        
        self.num_train = num_train
        self.num_val = num_val
        
        # Force clear unused memory if any
        import gc
        gc.collect()

    @torch.no_grad()
    def getBatches(self):
        # legacy method
        # Randomly sample indices from GPU tensors
        idx = torch.randint(0, len(self.train_indices), (self.batch_size,), device=self.device)
        indices = self.train_indices[idx]

        # Data is already on GPU
        states_tensor = self.states[indices]
        actions_tensor = self.actions[indices]

        return states_tensor, actions_tensor

    def getDataLoader(self, mode='train'):
        """
        Yields batches of data for one full epoch
        mode: 'train' or 'val'
        """
        if mode == 'train':
            indices = self.train_indices
            num_data = self.num_train
            # shuffle using GPU
            idx = torch.randperm(num_data, device=self.device)
            indices = indices[idx]
        else:
            indices = self.val_indices
            num_data = self.num_val
        
        for i in range(0, num_data, self.batch_size):
            # Slicing
            batch_indices = indices[i : i + self.batch_size]
            
            # Slicing GPU tensors keeps them on GPU
            states_tensor = self.states[batch_indices]
            actions_tensor = self.actions[batch_indices]
            
            yield states_tensor, actions_tensor