'''
    DO NOT CHANGE THIS FILE !
'''

from abc import ABC, abstractmethod

import pickle
import numpy as np
import torch

from utils.vectorize import RunningMeanStd

'''
    Basic method
'''
@torch.jit.script
def normalize(a, maximum, minimum):
    temp_a = 2.0/(maximum - minimum)
    temp_b = (maximum + minimum)/(minimum - maximum)
    temp_a = torch.ones_like(a)*temp_a
    temp_b = torch.ones_like(a)*temp_b
    return temp_a*a + temp_b

@torch.jit.script
def unnormalize(a, maximum, minimum):
    temp_a = (maximum - minimum)/2.0
    temp_b = (maximum + minimum)/2.0
    temp_a = torch.ones_like(a)*temp_a
    temp_b = torch.ones_like(a)*temp_b
    return temp_a*a + temp_b

@torch.jit.script
def clip(a, maximum, minimum):
    clipped = torch.where(a > maximum, maximum, a)
    clipped = torch.where(clipped < minimum, minimum, clipped)
    return clipped

'''
    Agent base
'''
class AgentBase(ABC):
    def __init__(self, args):
        # save args
        with open(f'{args.backup_dir}/args.pkl', 'wb') as f:
            pickle.dump(args, f)

        # base
        self.device = args.device
        self.name = args.name
        self.algo_idx = args.algo_idx
        self.log_list = args.log_list

        # directory
        self.save_dir = args.save_dir
        self.checkpoint_dir=f'{args.save_dir}/checkpoint'
        self.sim2real_dir = f"{args.save_dir}/sim2real"

        # for train
        self.train_frequency = args.train_frequency

        # for env
        self.discount_factor = args.discount_factor
        self.obs_dim = args.obs_dim
        self.action_dim = args.action_dim
        self.action_bound_min = torch.tensor(args.action_bound_min, device=args.device)
        self.action_bound_max = torch.tensor(args.action_bound_max, device=args.device)

        # for normalization
        self.norm_obs = args.norm_obs
        self.norm_reward = args.norm_reward
        self.obs_rms = RunningMeanStd('agent_obs', self.obs_dim)
        self.reward_rms = RunningMeanStd('agent_reward', 1)

    # ==================== common methods ==================== #
    # ======================================================== #

    def getAction(self, state:np.ndarray, deterministic:bool = True) -> torch.Tensor:
        # Check for NaN/Inf in input state
        if np.any(np.isnan(state)) or np.any(np.isinf(state)):
            state = np.nan_to_num(state, nan=0.0, posinf=10.0, neginf=-10.0)
        
        normalized_state = self.obs_rms.normalize(state)
        
        # Check normalized state
        if np.any(np.isnan(normalized_state)) or np.any(np.isinf(normalized_state)):
            normalized_state = np.nan_to_num(normalized_state, nan=0.0, posinf=10.0, neginf=-10.0)
        
        state_tensor = torch.tensor(normalized_state, dtype=torch.float32, device=self.device)
        
        # Final check before passing to actor
        if torch.any(torch.isnan(state_tensor)) or torch.any(torch.isinf(state_tensor)):
            state_tensor = torch.nan_to_num(state_tensor, nan=0.0, posinf=10.0, neginf=-10.0)
        
        action_dists = self.actor(state_tensor)

        if not deterministic:
            norm_action = action_dists.rsample()
            action = self.unnormalizeAction(norm_action)
        else:
            norm_action = action_dists.loc
            action = self.unnormalizeAction(norm_action)
        clipped_action = clip(action, self.action_bound_max, self.action_bound_min)

        self.state = state.copy()
        self.action = norm_action.detach().cpu().numpy()
        self.log_prob = torch.sum(action_dists.log_prob(norm_action), dim=-1).detach().cpu().numpy()
        return clipped_action

    @abstractmethod
    def step(self, rewards, dones, fails, next_states):
        """
            For add transitions into buffer.
        """

    @abstractmethod
    def train(self):
        """
            For train function in main.py
        """

    @abstractmethod
    def save(self):
        """
            Save agent's parameters such as actors, critics, and optimizers.
        """

    @abstractmethod
    def load(self):
        """
            Load agent's parameters such as actors, critics, and optimizers.
        """

    def normalizeAction(self, a:torch.Tensor) -> torch.Tensor:
        return normalize(a, self.action_bound_max, self.action_bound_min)

    def unnormalizeAction(self, a:torch.Tensor) -> torch.Tensor:
        return unnormalize(a, self.action_bound_max, self.action_bound_min)