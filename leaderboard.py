import argparse
import os
import random
import numpy as np
import torch
import pickle
from ruamel.yaml import YAML
from easydict import EasyDict

import gymnasium as gym

from utils.env import F1Wrapper
from utils.video import create_video
from utils.color import cprint, cprint_banner
from utils.vectorize import RunningMeanStd

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'        # for debug
EPS = 1e-8

MAPS = ['Sochi', 'Spa', 'Nuerburgring', 'Monza', 'Melbourne', 'Austin', 
        'Silverstone', 'Sakhir', 'IMS', 'Budapest', 'Montreal', 'Sepang', 
        'Oschersleben', 'YasMarina', 'MoscowRaceway', 'Zandvoort', 'Catalunya', 
        'BrandsHatch', 'Shanghai', 'Hockenheim', 'SaoPaulo', 'Spielberg', 'MexicoCity']

def main(args, device):
    # fix seed for comparison
    args.seed = 124
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
    args.reset_config = {"type": "cl_grid_static"}
    env = F1Wrapper(args=args, maps=MAPS, render_mode=render_mode)

    # load sim2real model
    obs_rms = RunningMeanStd('agent_obs', env.obs_dim)
    obs_rms.load(args.save_dir)
    actor = torch.load(f'{args.save_dir}/actor.pt').to(device)

    # test
    s_maps, s_laptimes = [], []
    f_maps, f_laptimes = [], []
    cprint(f'\n[Leaderboard] start.\n', bold=True, underline=True)
    for i, map in enumerate(MAPS):
        cprint(f'Lap No.{i+1}', bold=True)
        cprint(f'\tMap : {map}')
        done, frames = False, []
        observation, info = env.reset(options={'map':map})
        while not done:
            with torch.no_grad():
                state_tensor = torch.tensor(obs_rms.normalize(observation), dtype=torch.float32, device=device)
                action = actor(state_tensor).loc.cpu().numpy()
                observation, reward, terminate, truncate, info = env.step(action)
            done = terminate or truncate
            frames.append(env.render())

        lap_time = env._env.lap_times[0]
        if info['checkpoint_done']:       # success - logging laptime
            s_maps.append(map)
            s_laptimes.append(lap_time)
            print('\tResult : ', end="")
            cprint('Success', color='blue')
            cprint_banner(f'\tLap time: {lap_time:.3f}\n')
        else:
            f_maps.append(map)
            f_laptimes.append(lap_time)
            print('\tResult : ', end="")
            cprint('Fail', color='red')
            cprint_banner(f'\tLap time: {lap_time:.3f}\n')
            
        if args.save_video:
            create_video(frames, output_name=f'{args.save_dir}/video/{map}')

    cprint(f'Final Result.', bold=True, underline=True)
    print('\tSuccess Ratio : ', end=""); cprint(f'{(len(s_maps)/len(MAPS)):.3f}', bold=True)
    print('\tMean Laptime : ', end=""); cprint(f'{np.mean(s_laptimes):.3f}', bold=True)
    print(f'\tSuccess Maps : {s_maps}')
    print(f'\tFail Maps : {f_maps}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Leaderboard for F1tenth')
    parser.add_argument('--name', type=str, default='ppo', help='name of algorithm.')
    parser.add_argument('--algo_idx', type=int, default=1, help='index of algorithm.')
    parser.add_argument('--live_video', action='store_true', help='live video.')
    parser.add_argument('--save_video', action='store_true', help='save video.')
    args = parser.parse_args()
    args = EasyDict(vars(args))

    # load task configurations & merge them
    with open('configs/task/f1tenth.yaml', 'r') as f:
        task_args = EasyDict(YAML().load(f))
    with open('configs/task/dynamic.yaml', 'r') as f:
        dynamic_args = EasyDict(YAML().load(f))
    args.update(task_args)
    args.update(dynamic_args)

    args.save_dir = f"results/{args.name}/{args.algo_idx}/sim2real"
    os.makedirs(f'{args.save_dir}/video', exist_ok=True)
    # ==== processing args ==== #
    # device
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        cprint('[torch] cuda is used.')
    else:
        device = torch.device('cpu')
        cprint('[torch] cpu is used.')

    main(args, device)