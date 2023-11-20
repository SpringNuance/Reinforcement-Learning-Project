import torch.nn.functional as F
from torch import nn
from collections import namedtuple
import numpy as np
import torch

# from torch.distributions import Categorical
from torch.distributions import Normal, Independent

import pickle, os, random, torch

from collections import defaultdict
import pandas as pd 
import gymnasium as gym
import matplotlib.pyplot as plt


from sanding import SandingEnv
from gymnasium.envs.registration import register
from pathlib import Path

import imageio
import yaml, time

def save_rgb_arrays_to_gif(rgb_arrays, file_name):
    with imageio.get_writer(file_name, duration=0.1) as writer:
        for rgb_array in rgb_arrays:
            writer.append_data(rgb_array)
    print(f"Saved GIF to {file_name}")
    
class Logger(object):
    def __init__(self,):
        self.metrics = defaultdict(list)
        
    def log(self, **kwargs):
        for key, value in kwargs.items():
            self.metrics[key].append(value)
    
    def save(self, path, seed=None):
        df = pd.DataFrame.from_dict(self.metrics)
        print('logger and seed', seed)
        if seed is None:
            df.to_csv(f'{path}.csv')
        else:
            fname = f'{path}'+'_'+str(seed)+'.csv'
            print(fname)
            df.to_csv(fname)
        

def create_env(config_file):
    with open(config_file, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Register the environment
    register(
        id=cfg['env_name'],
        entry_point=__name__ + ":SandingEnv",  # Updated to use __name__ for current module
        max_episode_steps=cfg['max_episode_steps'],
        kwargs=cfg.get('env_config', {})  # Updated to use 'env_config'
    )
    
    # Create the environment
    env = gym.make(cfg['env_name'], render_mode=cfg['env_config'].get('render_mode', 'rgb_array'))
    return env



def make_dir(dir_path):
    """Create directory if it does not already exist."""
    try:
        os.makedirs(dir_path)
    except OSError:
        pass
    return dir_path

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed(seed)

def get_space_dim(space):
    t = type(space)
    if t is gym.spaces.Discrete:
        return space.n
    elif t is gym.spaces.Box:
        return space.shape[0]
    else:
        raise TypeError("Unknown space type:", t)
        
def plot_reward(path, seed, env_name, algo_name):
    if seed is None:
        fname = 'logs.csv'
    else:
        fname = 'logs_'+str(seed)+'.csv'
       
    
    df = pd.read_csv(str(path / fname))
    steps = df['total_step']
    average_return = df['average_return']
    plt.figure(figsize=(6,4))
    plt.plot(steps, average_return, linewidth=1.2)
    plt.xlabel('Timestep', fontweight=10)
    plt.ylabel('Average Reward', fontweight=10)
    plt.title(algo_name+'_'+env_name, fontweight=12)
    plt.gca().ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    plt.savefig(str(path)+'/figure_'+str(seed)+'.pdf')
    plt.plot()

def get_statistical_plots_data(path, seeds, env_name, algo_name):
    all_steps = []
    all_average_return = []
    for seed in seeds:
        fname = 'logs_'+str(seed)+'.csv'
       
        df = pd.read_csv(str(path / fname))
        steps = df['total_step']
        average_return = df['average_return']

        all_steps.append(steps)
        all_average_return.append(average_return)

     # Calculate mean and std across different runs
    mean_average_return = np.mean(all_average_return, axis=0)
    std_average_return = np.std(all_average_return, axis=0)

    return steps, mean_average_return, std_average_return
    

def plot_algorithm_training(path, seeds, env_name, algo_name):
    

     # Calculate mean and std across different runs
    steps, mean_average_return, std_average_return = get_statistical_plots_data(path, seeds, env_name, algo_name)

    
    plt.figure(figsize=(6, 4))
    
    # Plot mean curve
    plt.plot(steps[1:], mean_average_return[1:], linewidth=1.2, label='Mean')
    
    # Plot std deviation area
    plt.fill_between(steps[1:], mean_average_return[1:] - std_average_return[1:], mean_average_return[1:] + std_average_return[1:], alpha=0.2, label='Std Dev')
    
    plt.xlabel('Timestep', fontweight='bold', fontsize=12)
    plt.ylabel('Average Reward', fontweight='bold', fontsize=12)
    plt.title(algo_name+'_'+env_name, fontweight='bold', fontsize=14)
    plt.gca().ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    plt.legend()
    plt.savefig(path / f'figure_statistical_{env_name}.pdf')
    return steps, mean_average_return, std_average_return


def compare_algorithm_training(algo1, algo2, seeds):
    steps1, mean_average_return1, std_average_return1 = get_statistical_plots_data(algo1.logging_dir, seeds, algo1.env_name, algo1.algo_name)
    steps2, mean_average_return2, std_average_return2 = get_statistical_plots_data(algo2.logging_dir, seeds, algo2.env_name, algo2.algo_name)

    plt.figure(figsize=(6, 4))
    
    # Plot mean curve for algo1
    plt.plot(steps1[1:], mean_average_return1[1:], linewidth=1.2, label=f'{algo1.cfg.algo_name}')
    
    # Plot std deviation area for algo1
    plt.fill_between(steps1[1:], mean_average_return1[1:] - std_average_return1[1:], mean_average_return1[1:] + std_average_return1[1:], alpha=0.2)
    
    # Plot mean curve for algo2
    plt.plot(steps2[1:], mean_average_return2[1:], linewidth=1.2, label=f'{algo2.cfg.algo_name}')
    
    # Plot std deviation area for algo2
    plt.fill_between(steps2[1:], mean_average_return2[1:] - std_average_return2[1:], mean_average_return2[1:] + std_average_return2[1:], alpha=0.2)
    
    print()
    cur_dir=Path().cwd()
    save_path =  cur_dir/'results'/algo1.env_name

    plt.xlabel('Timestep', fontweight='bold', fontsize=12)
    plt.ylabel('Average Reward', fontweight='bold', fontsize=12)
    plt.title(algo1.env_name, fontweight='bold', fontsize=14)
    plt.gca().ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    plt.legend()
    plt.savefig(save_path / f'compare_{algo1.algo_name}_{algo2.algo_name}.pdf')
    plt.show()

    
def load_object(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

def save_object(obj, filename): 
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    
def soft_update_params(m, m_target, tau):
    """Update slow-moving average of online network (target network) at rate tau."""
    with torch.no_grad():
        for p, p_target in zip(m.parameters(), m_target.parameters()):
            p_target.data.lerp_(p.data, tau)