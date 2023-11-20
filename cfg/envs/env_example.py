from sanding import SandingEnv
from gymnasium.envs.registration import register

import numpy
import yaml, time, torch
import gymnasium as gym

def create_env(config_file):
    with open(config_file, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Register the environment
    register(
        id=cfg['env_name'],
        entry_point="%s:SandingEnv" % __name__,
        max_episode_steps=cfg['max_episode_steps'],
        kwargs=cfg.get('env_params', {})
    )
    
    # Create the environment
    env = gym.make(cfg['env_name'], render_mode='rgb_array')
    return env

# Example usage:
env = create_env('cfg/difficult_env.yaml')
env = create_env('cfg/middle_env.yaml')
env = create_env('cfg/easy_env.yaml')
