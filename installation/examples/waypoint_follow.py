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
from utils.rule_based import PurePursuitPlanner

EPS = 1e-8

MAPS = ['Sochi', 'Spa', 'Nuerburgring', 'Monza', 'Melbourne', 'Austin', 
        'Silverstone', 'Sakhir', 'IMS', 'Budapest', 'Montreal', 'Sepang', 
        'Oschersleben', 'YasMarina', 'MoscowRaceway', 'Zandvoort', 'Catalunya', 
        'BrandsHatch', 'Shanghai', 'Hockenheim', 'SaoPaulo', 'Spielberg', 'MexicoCity']

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

    env = F1Wrapper(args=args, maps=MAPS, render_mode="human_fast" if args.render else None)
    
    while True:
        observation, info = env.reset()
        track = env._env.unwrapped.track
        planner = PurePursuitPlanner(track=track, wb=0.17145 + 0.15875)

        if args.render:
            env.unwrapped.add_render_callback(track.raceline.render_waypoints)
            env.unwrapped.add_render_callback(planner.render_local_plan)
            env.unwrapped.add_render_callback(planner.render_lookahead_point)

        ep_reward = 0.0
        start = time.time()

        terminate = False
        while not terminate:
            speed, steer = planner.plan(info['obs_dict'], PARAM)
            action = np.clip([steer/(env.max_steer+EPS), speed/env.max_speed], -np.ones(2), np.ones(2))

            obs, reward, terminate, truncate, info = env.step(action)
            ep_reward += reward
            if args.render:
                env.render()

        print(f"Real elapsed time: {time.time() - start} / reward: {ep_reward}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1, help='seed number.')
    parser.add_argument("--map", type=str, default="Spielberg", help="map name")
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