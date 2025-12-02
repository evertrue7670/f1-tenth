# backup
from shutil import copyfile

from copy import deepcopy
import argparse
import os
import time
import random
import numpy as np
import torch
import wandb
from ruamel.yaml import YAML
from easydict import EasyDict

import gymnasium as gym

from utils.env import F1Wrapper
from utils.logger import Logger
from utils.video import create_video
from utils.color import cprint

from algorithm import algo_dict

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'        # for debug
EPS = 1e-8

MAPS = ['Sochi', 'Spa', 'Nuerburgring', 'Monza', 'Melbourne', 'Austin', 
        'Silverstone', 'Sakhir', 'IMS', 'Budapest', 'Montreal', 'Sepang', 
        'Oschersleben', 'YasMarina', 'MoscowRaceway', 'Zandvoort', 'Catalunya', 
        'BrandsHatch', 'Shanghai', 'Hockenheim', 'SaoPaulo', 'Spielberg', 'MexicoCity']

EVAL_MAPS = ['Sochi', 'IMS', 'Catalunya', 'Zandvoort', 'Silverstone', 'SaoPaulo',       # easy
             'YasMarina', 'MoscowRaceway', 'Shanghai', 'MexicoCity', 'Montreal']        # hard

def getParser():
    parser = argparse.ArgumentParser(description='F1tenth.')
    # mode
    parser.add_argument('--test',  action='store_true', help='for test.')
    # algorithm
    parser.add_argument('--algo', type=str, default='ppo', help='name of algorithm: ppo, sac, trpo')
    parser.add_argument('--name', type=str, default=None, help='algo name.')
    parser.add_argument('--algo_idx', type=int, default=1, help='algo index.')
    parser.add_argument('--model_num', type=int, default=None, help='for checkpoint num.')
    # common
    parser.add_argument('--wandb',  action='store_true', help='use wandb?')
    parser.add_argument('--comment', type=str, default=None, help='wandb comment saved in run name.')
    parser.add_argument('--device', type=str, default='gpu', help='gpu or cpu.')
    parser.add_argument('--gpu_idx', type=int, default=0, help='GPU index.')
    parser.add_argument('--seed', type=int, default=1, help='seed number.')
    parser.add_argument('--log_freq', type=int, default=int(1e4), help='# of time steps for logging.')
    parser.add_argument('--save_freq', type=int, default=int(5e5), help='# of time steps for save.')
    # train
    parser.add_argument('--n_envs', type=int, default=5, help='# of vercotrized environments.')
    parser.add_argument('--n_steps', type=int, default=5000, help='# of steps for update.')
    parser.add_argument('--total_steps', type=int, default=int(2.5e6), help='total interaction steps.')
    # for eval
    parser.add_argument('--eval_num', type=int, default=5, help='# of evaluation epochs.')
    parser.add_argument('--eval_freq', type=int, default=int(5e4), help='# of time steps for eval.')
    parser.add_argument('--save_video', action='store_true', help='save video.')
    parser.add_argument('--live_video', action='store_true', help='live video.')
    return parser


def eval(args, env, agent, name, eval_num=1):
    cprint('[Evaluation]\t{:^15}\tLap Time'.format('Map'), color='cyan')

    steps, scores, frames = [], [], []
    for _ in range(eval_num):
        step, score, done, _frames = 0, 0, False, []
        observation, info = env.reset()
        while not done and step < env._env.max_episode_steps:
            step += 1

            with torch.no_grad():
                clipped_action_tensor = agent.getAction(observation, deterministic=True)
                clipped_action = clipped_action_tensor.detach().cpu().numpy()
                observation, reward, terminate, truncate, info = env.step(clipped_action)
            score += reward
            done = terminate or truncate
            
            # Collect frames for video
            if args.live_video or args.save_video or args.wandb:
                frame = env.render()
                if frame is not None:
                    _frames.append(frame)

        if truncate or info['checkpoint_done']:
            print('[Success]   \t{:^15}\t{:>5s}'.format(env._env.map, f'{env._env.lap_times[0]:.2f}'))
        else:
            print('[Fail]      \t{:^15}\t{:>5s}'.format(env._env.map, f'{env._env.lap_times[0]:.2f}'))

        steps.append(step)
        scores.append(score)
        frames.append(_frames)

    max_score_idx = np.argmax(scores)
    step_mean, score_mean = np.mean(steps), np.mean(scores)
    cprint(f'[Eval]-{name} \t steps: {step_mean:.2f} \t score: {score_mean:.2f}', color='cyan')
    best_frames = frames[max_score_idx]
    
    # Save video locally
    if args.save_video and len(best_frames) > 0:
        create_video(best_frames, output_name=f'{args.save_dir}/video/{name}')
    
    return best_frames, score_mean

def train(args):
    # backup
    copyfile('configs/task/f1tenth.yaml', f'{args.backup_dir}/f1tenth.yaml')
    copyfile('configs/task/dynamic.yaml', f'{args.backup_dir}/dynamic.yaml')
    copyfile(f'configs/algorithm/{args.algo}.yaml', f'{args.backup_dir}/{args.algo}.yaml')

    # for random seed
    np.random.seed(args.seed)
    random.seed(args.seed)    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # define train vectorized environment
    vec_env_id = lambda: F1Wrapper(args=args, maps=MAPS)
    vec_env = gym.vector.AsyncVectorEnv([vec_env_id for _ in range(args.n_envs)])

    # define eval environment
    render_mode = None
    if args.live_video:
        render_mode = "human_fast"
    if args.save_video or args.wandb:  # Enable rendering for wandb video logging
        render_mode = "rgb_array"
    eval_env = F1Wrapper(args=args, maps=EVAL_MAPS, render_mode=render_mode)

    # set args value for env
    args.action_bound_min = vec_env.single_action_space.low
    args.action_bound_max = vec_env.single_action_space.high

    # define agent
    args.train_frequency = min(args.train_frequency, args.n_steps)      # for code compatibility
    agent = algo_dict[args.algo](args)
    agent.load(args.model_num)

    # wandb
    if args.wandb:
        wandb.init(
            project='[F1tenth] Baselines', 
            config=vars(args),  # Convert args to dict
            name=f"{args.name}/{args.comment}" if args.comment else f"{args.name}/seed_{args.algo_idx}",
            tags=[args.algo, args.name],
            reinit=True
        )
        print(f"[Wandb] Initialized. Project: [F1tenth] Baselines, Run: {wandb.run.name}")

    # for log
    log_list = deepcopy(agent.log_list)
    rollout_logger = {arg: Logger(args.log_dir, arg) for arg in log_list['rollout']}
    train_logger = {arg: Logger(args.log_dir, arg) for arg in log_list['train']}

    # train
    total_step = 0
    train_step = 0
    log_step = 0
    save_step = 0
    eval_step = 0
    best_score = -np.inf
    while total_step < args.total_steps:
        
        # ======= collect trajectories ======= #
        start_time = time.time()
        step = 0
        observations, info = vec_env.reset()
        reward_history = [[] for _ in range(args.n_envs)]
        while step < args.n_steps:
            step += args.n_envs
            total_step += args.n_envs

            with torch.no_grad():
                clipped_action_tensor = agent.getAction(observations, deterministic=False)
                clipped_actions = clipped_action_tensor.detach().cpu().numpy()
                observations, rewards, terminates, truncates, infos = vec_env.step(clipped_actions)

            temp_fails = []
            temp_dones = []
            temp_observations = []

            for env_idx in range(args.n_envs):
                reward_history[env_idx].append(rewards[env_idx])
                fail = (not truncates[env_idx]) and terminates[env_idx]
                done = terminates[env_idx] or truncates[env_idx]
                temp_observations.append(observations[env_idx])
                temp_fails.append(fail)
                temp_dones.append(done)

                if done:
                    ep_len = len(reward_history[env_idx])
                    score = np.sum(reward_history[env_idx])
                    if 'score' in rollout_logger.keys():
                        rollout_logger['score'].write(ep_len, score)
                    if 'ep_len' in rollout_logger.keys():
                        rollout_logger['ep_len'].write(ep_len, ep_len)
                    
                    # Log episode-level metrics to wandb
                    if args.wandb:
                        wandb.log({
                            'episode/reward': score,
                            'episode/length': ep_len,
                            'rollout/step': total_step
                        }, step=total_step)
                    
                    reward_history[env_idx] = []

            temp_fails = np.array(temp_fails)
            temp_dones = np.array(temp_dones)
            temp_observations = np.array(temp_observations)
            agent.step(rewards, temp_dones, temp_fails, temp_observations)
            if total_step - train_step >= agent.train_frequency:
                train_results = agent.train(total_step)
                train_step = total_step
        # ==================================== #

        # logging
        for key, value in train_results.items():
            if key in log_list['train']:
                train_logger[key].write(step, value)

        print_len = max(int(args.n_steps/args.max_episode_steps), args.n_envs)
        log_data = {
            'rollout/step': total_step,
            'train/fps': args.n_steps/(time.time()-start_time),
            'train/best': 0         # if best model updated, becomes True
        }

        # evaluation
        if total_step - eval_step >= args.eval_freq:
            eval_step = total_step
            eval_frames, eval_score = eval(args, eval_env, agent, name=str(eval_step//args.eval_freq))
            
            # Log video to wandb
            if args.wandb and eval_frames is not None and len(eval_frames) > 0:
                # Save temporary file for wandb upload
                # Note: create_video automatically appends .mp4
                video_base_path = f'{args.video_dir}/eval_step_{total_step}'
                create_video(eval_frames, output_name=video_base_path)
                
                video_path = f"{video_base_path}.mp4"
                
                if os.path.exists(video_path):
                    wandb.log({
                        "eval/video": wandb.Video(video_path, fps=50, format="mp4"),
                        "eval/score": eval_score,
                        "rollout/step": total_step
                    }, step=total_step)
                    print(f"[Wandb] Uploaded video for step {total_step}")
                else:
                    print(f"[Warning] Video file not found: {video_path}")
            
            # save best model
            if eval_score > best_score:
                cprint(f'[{args.name}] save best model. Score: {eval_score:.2f}', bold=True, color="green")
                agent.save(log=False)
                best_score = eval_score
                log_data['train/best'] = 1

        # logger
        for arg, logger in rollout_logger.items():
            if arg == 'step': continue
            log_data[f'rollout/{arg}'] = logger.get_avg(print_len)
        for arg, logger in train_logger.items():
            log_data[f'train/{arg}'] = logger.get_avg()
        if total_step - log_step >= args.log_freq:
            log_step = total_step
            print(log_data)
            if args.wandb:
                wandb.log(log_data, step=total_step)
                print(f"[Wandb] Logged at step {total_step}")

        # save
        if total_step - save_step >= args.save_freq:
            save_step = total_step
            agent.save(model_num=int(total_step))
            for logger in rollout_logger.values():
                logger.save()
            for logger in train_logger.values():
                logger.save()

    vec_env.close()
    eval_env.close()


def test(args):
    # for random seed
    np.random.seed(args.seed)
    random.seed(args.seed)    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # define Environment
    render_mode = None
    if args.live_video:
        render_mode = "human_fast"
    if args.save_video:
        render_mode = "rgb_array"
    args.max_episode_steps = 10000
    test_env = F1Wrapper(args=args, maps=args.map if args.map else EVAL_MAPS, render_mode=render_mode)

    # set args value for env
    args.action_bound_min = test_env.action_space.low
    args.action_bound_max = test_env.action_space.high

    # define agent
    agent = algo_dict[args.algo](args)
    agent.load(args.model_num)

    # test
    eval(args, test_env, agent, name='test', eval_num=args.eval_num)


if __name__ == "__main__":
    # base configurations
    parser = getParser()
    args = parser.parse_args()
    if args.name == None:
        args.name = args.algo
    args = EasyDict(vars(args))

    # load configurations & merge them
    with open('configs/task/f1tenth.yaml', 'r') as f:
        task_args = EasyDict(YAML().load(f))
    with open('configs/task/dynamic.yaml', 'r') as f:
        dynamic_args = EasyDict(YAML().load(f))
    with open(f'configs/algorithm/{args.algo}.yaml', 'r') as f:
        agent_args = EasyDict(YAML().load(f))
    args.update(task_args)
    args.update(dynamic_args)
    args.update(agent_args)
    args.save_dir = f"results/{args.name}/{args.algo_idx}"

    # ==== processing args ==== #
    # directory
    args.log_dir = f'{args.save_dir}/logs'
    args.video_dir = f'{args.save_dir}/video'
    args.backup_dir = f'{args.save_dir}/backup'
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.video_dir, exist_ok=True)
    os.makedirs(args.backup_dir, exist_ok=True)
    # set gpu
    os.environ["CUDA_VISIBLE_DEVICES"]=f"{args.gpu_idx}"
    # device
    if torch.cuda.is_available() and args.device == 'gpu':
        device = torch.device('cuda:0')
        cprint('[torch] cuda is used.')
    else:
        device = torch.device('cpu')
        cprint('[torch] cpu is used.')
    args.device = device

    if args.test:
        test(args)
    else:
        train(args)