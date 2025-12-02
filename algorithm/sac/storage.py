from collections import deque
import numpy as np
import torch
import os

EPS = 1e-8 

import numpy as np
import torch

class ReplayBuffer:
    def __init__(
            self, device:torch.device, 
            obs_dim:int, 
            action_dim:int, 
            n_envs:int,
            buffer_size: int) -> None:

        self.device = device
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_envs = n_envs
        self.buffer_size = buffer_size # must be divisible by number of environments
        assert self.buffer_size % self.n_envs == 0
        self.initialize()

    ################
    # Public Methods
    ################

    def initialize(self):
        self.states = np.empty((self.buffer_size, self.obs_dim), dtype=np.float32)
        self.next_states = np.empty((self.buffer_size, self.obs_dim), dtype=np.float32)
        self.actions = np.empty((self.buffer_size, self.action_dim), dtype=np.float32)
        self.rewards = np.empty((self.buffer_size, 1), dtype=np.float32)
        self.dones = np.empty((self.buffer_size, 1), dtype=np.float32)
        self.fails = np.empty((self.buffer_size, 1), dtype=np.float32)

        self.idx = 0
        self.full = False

    def addTransition(self, states:np.ndarray, actions:np.ndarray, rewards:np.ndarray, dones:np.ndarray, fails:np.ndarray, next_states:np.ndarray):
        self.states[self.idx:self.idx+self.n_envs] = states.copy()
        self.actions[self.idx:self.idx+self.n_envs] = actions.copy()
        self.rewards[self.idx:self.idx+self.n_envs] = rewards[:, None].copy()
        self.next_states[self.idx:self.idx+self.n_envs] = next_states.copy()
        self.dones[self.idx:self.idx+self.n_envs] = dones[:, None].copy()
        self.fails[self.idx:self.idx+self.n_envs] = fails[:, None].copy()

        self.idx = (self.idx + self.n_envs) % self.buffer_size
        self.full = self.full or self.idx == 0

    def getBatches(self, batch_size, state_rms, reward_rms):
        idxs = np.random.randint(0,
                                self.buffer_size if self.full else self.idx,
                                size=batch_size)

        states = state_rms.normalize(self.states[idxs])
        next_states = state_rms.normalize(self.next_states[idxs])
        rewards = reward_rms.normalize(self.rewards[idxs])
        
        states = torch.as_tensor(states, device=self.device)
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_states = torch.as_tensor(next_states, device=self.device)
        dones = torch.as_tensor(self.dones[idxs], device=self.device)
        fails = torch.as_tensor(self.fails[idxs], device=self.device)

        return states, actions, rewards, next_states, dones, fails