from algorithm.common import *
from utils.color import cprint

from .storage import RolloutBuffer

import numpy as np
import torch
import os

EPS = 1e-8


class Agent(AgentBase):
    def __init__(self, args):
        super().__init__(args)

        # for expert rollout
        self.total_steps = args.total_steps

        # for training
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.max_grad_norm = args.max_grad_norm

        # for model
        self.actor = Actor(args).to(self.device)
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)

        # for buffer
        self.rollout_dataset = RolloutBuffer(self.device, self.total_steps, self.batch_size)


    def step(self, states, actions):
        # Check for NaN/Inf in input
        if np.any(np.isnan(states)) or np.any(np.isinf(states)):
            cprint(f'[WARNING] NaN/Inf in states before normalization!', color='red')
            states = np.nan_to_num(states, nan=0.0, posinf=1e6, neginf=-1e6)
        
        if np.any(np.isnan(actions)) or np.any(np.isinf(actions)):
            cprint(f'[WARNING] NaN/Inf in actions!', color='red')
            actions = np.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # update statistics
        if self.norm_obs:
            self.obs_rms.update(states)
            states = self.obs_rms.normalize(states)
            
            # Check normalized states
            if np.any(np.isnan(states)) or np.any(np.isinf(states)):
                cprint(f'[WARNING] NaN/Inf after normalization!', color='red')
                print(f'  obs_rms.mean stats: min={self.obs_rms.mean.min():.4f}, max={self.obs_rms.mean.max():.4f}')
                print(f'  obs_rms.var stats: min={self.obs_rms.var.min():.4f}, max={self.obs_rms.var.max():.4f}')
                states = np.nan_to_num(states, nan=0.0, posinf=1e6, neginf=-1e6)

        self.rollout_dataset.addBatch(states, actions)

    def train(self):
        states_tensor, actions_tensor = self.rollout_dataset.getBatches()
        
        # Check input tensors
        if torch.any(torch.isnan(states_tensor)) or torch.any(torch.isinf(states_tensor)):
            cprint(f'[ERROR] NaN/Inf in states_tensor!', color='red')
            print(f'  states_tensor stats: min={states_tensor.min():.4f}, max={states_tensor.max():.4f}, mean={states_tensor.mean():.4f}')
            states_tensor = torch.nan_to_num(states_tensor, nan=0.0, posinf=1e6, neginf=-1e6)
        
        if torch.any(torch.isnan(actions_tensor)) or torch.any(torch.isinf(actions_tensor)):
            cprint(f'[ERROR] NaN/Inf in actions_tensor!', color='red')
            actions_tensor = torch.nan_to_num(actions_tensor, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Check model parameters before forward
        for name, param in self.actor.named_parameters():
            if torch.any(torch.isnan(param)) or torch.any(torch.isinf(param)):
                cprint(f'[ERROR] NaN/Inf in model parameter: {name}!', color='red')
                return {'actor_loss': float('inf')}
        
        action_dists = self.actor(states_tensor)
        
        # Check model output
        if torch.any(torch.isnan(action_dists.mean)) or torch.any(torch.isinf(action_dists.mean)):
            cprint(f'[ERROR] NaN/Inf in action_dists.mean!', color='red')
            print(f'  states_tensor stats: min={states_tensor.min():.4f}, max={states_tensor.max():.4f}')
            print(f'  actions_tensor stats: min={actions_tensor.min():.4f}, max={actions_tensor.max():.4f}')
            # Check intermediate activations
            with torch.no_grad():
                x = states_tensor
                x = self.actor.activ(self.actor.fc1(x))
                if torch.any(torch.isnan(x)) or torch.any(torch.isinf(x)):
                    cprint(f'  [DEBUG] NaN/Inf after fc1!', color='yellow')
                x = self.actor.activ(self.actor.fc2(x))
                if torch.any(torch.isnan(x)) or torch.any(torch.isinf(x)):
                    cprint(f'  [DEBUG] NaN/Inf after fc2!', color='yellow')
                mean_raw = self.actor.fc_mean(x)
                if torch.any(torch.isnan(mean_raw)) or torch.any(torch.isinf(mean_raw)):
                    cprint(f'  [DEBUG] NaN/Inf in mean_raw (before tanh)!', color='yellow')
                    print(f'    mean_raw stats: min={mean_raw.min():.4f}, max={mean_raw.max():.4f}')
            return {'actor_loss': float('inf')}

        # ============================ implement here ============================ #
        MSE_Loss = torch.nn.MSELoss()
        action_mean = action_dists.mean
        actor_loss = MSE_Loss(action_mean, actions_tensor)
        # ======================================================================== #
        
        # Check loss
        if torch.isnan(actor_loss) or torch.isinf(actor_loss):
            cprint(f'[ERROR] NaN/Inf in actor_loss!', color='red')
            return {'actor_loss': float('inf')}
        
        # update
        self.optimizer.zero_grad()
        actor_loss.backward()
        
        # Check gradients
        total_norm = 0.0
        for p in self.actor.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                if torch.any(torch.isnan(p.grad)) or torch.any(torch.isinf(p.grad)):
                    cprint(f'[ERROR] NaN/Inf in gradients! Parameter: {p.shape}', color='red')
                    return {'actor_loss': float('inf')}
        total_norm = total_norm ** (1. / 2)
        
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        # Check parameters after update
        for name, param in self.actor.named_parameters():
            if torch.any(torch.isnan(param)) or torch.any(torch.isinf(param)):
                cprint(f'[ERROR] NaN/Inf in model parameter after update: {name}!', color='red')
                return {'actor_loss': float('inf')}

        train_results = {
            'actor_loss': actor_loss.item(),
            }
        
        return train_results

    def save(self, model_num=None, log=True):
        if model_num is None:
            checkpoint_file = f"{self.checkpoint_dir}/model.pt"
            # for sim2real
            self.obs_rms.save(self.sim2real_dir)
            self.reward_rms.save(self.sim2real_dir)
            torch.save(self.actor, f"{self.sim2real_dir}/actor.pt")
        else:
            checkpoint_file = f"{self.checkpoint_dir}/model_{model_num}.pt"

        # save rms
        self.obs_rms.save(self.save_dir)
        self.reward_rms.save(self.save_dir)

        # save models
        torch.save({
            'actor': self.actor.state_dict(),
            'optim': self.optimizer.state_dict(),
            }, checkpoint_file)
        if log: cprint(f'[{self.name}] save success.', bold=True, color="blue")

    def load(self, model_num=None):
        # load rms
        self.obs_rms.load(self.save_dir)
        self.reward_rms.load(self.save_dir)

        # load models
        if not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
            os.makedirs(self.sim2real_dir)
        if model_num is None:
            checkpoint_file = f"{self.checkpoint_dir}/model.pt"
        else:
            checkpoint_file = f"{self.checkpoint_dir}/model_{model_num}.pt"
        if os.path.isfile(checkpoint_file):
            checkpoint = torch.load(checkpoint_file)
            self.actor.load_state_dict(checkpoint['actor'])
            self.optimizer.load_state_dict(checkpoint['optim'])
            cprint(f'[{self.name}_{self.algo_idx}] load success.', bold=True, color="blue")
        else:
            self.actor.initialize()
            cprint(f'[{self.name}] load fail.', bold=True, color="red")