from algorithm.common import *
from utils.color import cprint

from .storage import ReplayBuffer

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.clip_grad import clip_grad_norm_
import os

EPS = 1e-8

def syncTargetNet(net, target_net):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(param)
        
def softTargetUpdate(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(target_param.data*tau + param.data*(1.0-tau))

class Agent(AgentBase):
    def __init__(self, args):
        super().__init__(args)

        # for training
        self.n_envs = args.n_envs
        self.lr = args.lr
        self.alpha_lr = args.alpha_lr
        self.max_grad_norm = args.max_grad_norm
        self.init_entropy_alpha = args.init_entropy_alpha
        self.buffer_size = args.buffer_size
        self.batch_size = args.batch_size
        self.polyak_tau = args.polyak_tau
        self.train_epochs = args.train_epochs
        self.init_update_steps = args.init_update_steps
        
        # for buffer
        self.replay_buffer = ReplayBuffer(self.device, self.obs_dim, self.action_dim, self.n_envs, self.buffer_size)
        
        # for model
        self.actor = SquashedActor(args).to(self.device)
        self.critic = DoubleQCritic(args).to(self.device)
        self.target_critic = DoubleQCritic(args).to(self.device)
        self.log_entropy_alpha = torch.log(self.init_entropy_alpha*torch.ones(1, dtype=torch.float32, device=self.device))
        self.log_entropy_alpha.requires_grad_(True)
        
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.lr*3)
        self.alpha_optim = torch.optim.Adam([self.log_entropy_alpha], lr=self.alpha_lr)
        
        self.target_entropy = float(-self.action_dim)
        
        syncTargetNet(self.critic, self.target_critic)

    
    def step(self, rewards, dones, fails, next_states):
        self.replay_buffer.addTransition(self.state, self.action, rewards, dones, fails, next_states)

        # update statistics
        if self.norm_obs:
            self.obs_rms.update(self.state)
        if self.norm_reward:
            self.reward_rms.update(rewards)

    def train(self, steps):
        if steps < self.init_update_steps:
            return {
                'critic_loss':0,
                'actor_loss':0,
                'entropy_loss':0,
                'alpha':0,
                'entropy':0,
                'epochs':0,
                'lr': 0,
                'average_q': 0
                }
        # logs
        epochs = 0

        for train_epoch in range(self.train_epochs):
            epochs += 1

            # get batches
            states_tensor, actions_tensor, rewards_tensor, \
                next_states_tensor, _, fails_tensor = self.replay_buffer.getBatches(self.batch_size, self.obs_rms, self.reward_rms)
            
            # ============================ implement here ============================ #
            with torch.no_grad():
                entropy_alpha = self.log_entropy_alpha.exp()

            # critic update
            with torch.no_grad():
                ##############
                ##  eq.1~2  ##
                ##############
                next_actions_tensor, next_log_probs_tensor = self.actor(next_states_tensor)
                target_critic1, target_critic2 = self.target_critic(next_states_tensor, next_actions_tensor)
                target_min_q = torch.min(target_critic1, target_critic2)
                target_q = rewards_tensor + self.discount_factor * (1 - fails_tensor) * (target_min_q - entropy_alpha * next_log_probs_tensor)

            self.critic_optim.zero_grad()
            critic1, critic2 = self.critic(states_tensor, actions_tensor)
            critic_loss = F.mse_loss(critic1, target_q) + F.mse_loss(critic2, target_q)
            critic_loss.backward()
            clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optim.step()
            
            # target critic update
            softTargetUpdate(self.critic, self.target_critic, self.polyak_tau)

            # alpha update
            ##############
            ##   eq.4   ##
            ##############
            with torch.no_grad():
                _, log_probs_tensor = self.actor(states_tensor)
                entropy = -log_probs_tensor.mean()
                    
            self.alpha_optim.zero_grad()
            entropy_loss = -(self.log_entropy_alpha * (log_probs_tensor + self.target_entropy).detach()).mean()
            entropy_loss.backward()
            self.alpha_optim.step()
                
            # actor update
            ##############
            ##  eq.5~6  ##
            ##############
            self.critic.eval()
            # Re-sample actions to get gradients properly
            actions_tensor_new, log_probs_tensor_new = self.actor(states_tensor)
            critic1_new, critic2_new = self.critic(states_tensor, actions_tensor_new)
            min_q_new = torch.min(critic1_new, critic2_new)
            
            actor_loss = (entropy_alpha * log_probs_tensor_new - min_q_new).mean()
            
            self.actor_optim.zero_grad()
            actor_loss.backward()
            clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optim.step()
            self.critic.train()
            
            q = min_q_new
            # ======================================================================== #
    
        train_results = {
            'critic_loss':critic_loss.item(),
            'actor_loss':actor_loss.item(),
            'entropy_loss':entropy_loss.item(),
            'alpha':entropy_alpha.item(),
            'entropy':entropy.item(),
            'epochs':epochs,
            'lr': self.lr,
            'average_q': q.mean().item(),
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
            'critic': self.critic.state_dict(),
            'log_entropy_alpha': self.log_entropy_alpha,
            'actor_optim': self.actor_optim.state_dict(),
            'critic_optim': self.critic_optim.state_dict(),
            'alpha_optim': self.alpha_optim.state_dict()
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
            self.critic.load_state_dict(checkpoint['critic'])
            self.log_entropy_alpha = checkpoint['log_entropy_alpha']
            self.actor_optim.load_state_dict(checkpoint['actor_optim'])
            self.critic_optim.load_state_dict(checkpoint['critic_optim'])
            self.alpha_optim.load_state_dict(checkpoint['alpha_optim'])
            cprint(f'[{self.name}_{self.algo_idx}] load success.', bold=True, color="blue")
        else:
            self.actor.initialize()
            self.critic.initialize()
            cprint(f'[{self.name}] load fail.', bold=True, color="red")
            