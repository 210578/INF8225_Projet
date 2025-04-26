import gym
import torch
import random
import numpy as np
import os
import json

def make_env(args):
    return gym.make(args.env_name)

def set_seed(env, seed):
    env.reset(seed=seed)
    if hasattr(env.action_space, 'seed'):
        env.action_space.seed(seed)
    if hasattr(env.observation_space, 'seed'):
        env.observation_space.seed(seed)


def print_args(args):
    print(json.dumps(vars(args), indent=4))

def save_args(args, path='./results'):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, f"{args.algorithm}_{args.env_name}_args.json"), 'w') as f:
        json.dump(vars(args), f, indent=4)
