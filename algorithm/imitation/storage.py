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
            # Ensure states and actions are numpy arrays
            if not isinstance(states, np.ndarray):
                states = np.array(states, dtype=np.float32)
            if not isinstance(actions, np.ndarray):
                actions = np.array(actions, dtype=np.float32)
            
            # Make contiguous copies to avoid memory issues
            self.states = np.ascontiguousarray(states, dtype=np.float32)
            self.actions = np.ascontiguousarray(actions, dtype=np.float32)
            
            # Update total_steps to actual data size
            actual_size = len(states)
            if actual_size != self.total_steps:
                print(f'[Storage] Updating total_steps from {self.total_steps} to actual data size {actual_size}')
                self.total_steps = actual_size
        
    @torch.no_grad()
    def getBatches(self):
        # Use actual data size instead of fixed total_steps
        data_size = len(self.states)
        indices = np.random.permutation(data_size)[:self.batch_size]

        # convert to tensor
        states_tensor = torch.tensor(self.states[indices], device=self.device, dtype=torch.float32)
        actions_tensor = torch.tensor(self.actions[indices], device=self.device, dtype=torch.float32)

        return states_tensor, actions_tensor
    