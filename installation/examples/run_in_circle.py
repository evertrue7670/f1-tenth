### path append ###
import sys
sys.path.append('../..')
###################

import time
import argparse
import numpy as np
import gymnasium as gym
from ruamel.yaml import YAML
from easydict import EasyDict

from utils.env import F1Wrapper

EPS = 1e-8

PARAM = {
    "mass": 3.463388126201571,
    "lf": 0.15597534362552312,
    "tlad": 0.82461887897713965,
    "vgain": 0.5,
}

def main(args):
    """
    main entry point
    """

    env = F1Wrapper(args=args, maps=['CIRCLE'], render_mode="human_fast" if args.render else None)
    env.min_speed = 1.0
    env.max_speed = 2.0
    
    env.reset()
    default_pos = env.position
    max_distance = -np.inf

    for _ in range(1000):
        env.step(np.array([1.0, 1.0]))
        
        # calculate the radius of circle when the car goes in maximum steering angle.
        cur_pos = env.position
        distance = np.linalg.norm(cur_pos - default_pos)
        if distance >= max_distance:
            max_distance = distance
        else:
            print(f'Radius: {max_distance/2:.3f}')
        
        if args.render:
            env.render()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1, help='seed number.')
    parser.add_argument("--render", action='store_true', help="for render.")
    args = parser.parse_args()
    args = EasyDict(vars(args))
    
    # load configurations & merge them
    with open('../../configs/task/f1tenth.yaml', 'r') as f:
        task_args = EasyDict(YAML().load(f))
    with open('../../configs/task/dynamic.yaml', 'r') as f:
        dynamic_args = EasyDict(YAML().load(f))
    args.update(task_args)
    args.update(dynamic_args)

    main(args)