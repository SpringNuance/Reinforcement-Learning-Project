import sys, os
sys.path.insert(0, os.path.abspath(".."))
os.environ["MUJOCO_GL"] = "egl" # for mujoco rendering
import time
from pathlib import Path
import pickle 
import torch, hydra,  warnings, yaml
import gymnasium as gym
import numpy as np
warnings.filterwarnings("ignore", category=UserWarning) 
warnings.filterwarnings("ignore", category=DeprecationWarning) 
from matplotlib import pyplot as plt
import utils as u

def to_numpy(tensor):
    return tensor.cpu().numpy().flatten()
   
def setup(cfg_path, cfg_args={}):
    
    with open(cfg_path, 'r')as f:
        d = yaml.safe_load(f)
        d.update(cfg_args)
        cfg = u.Struct(**d)
        
    # Setting library seeds
    if cfg.seed == None:
        seed = np.random.randint(low=1, high=1000)
    else:
        seed = cfg.seed
    
    print("Numpy/Torch/Random Seed: ", seed)
    u.set_seed(seed) # set seed
    
    #cfg.run_id = int(time.time())

    # create folders if needed
    work_dir = Path().cwd()/'results'/f'{cfg.env_name}'
    if cfg.save_model: u.make_dir(work_dir/"model")
    if cfg.save_logging: 
        u.make_dir(work_dir/"logging")

    # Model filename
    if cfg.model_path == 'default':
        cfg.model_path = work_dir/'model'/f'{cfg.env_name}_params.pt'


    # create a env
    env = gym.make(cfg.env_name, render_mode='rgb_array' if cfg.save_video else None)

    if cfg.save_video:
        # During testing, save every episode
        if cfg.testing:
            ep_trigger = 9
            video_path = work_dir/'video'/'test'
        # During training, save every 50th episode
        else:
            ep_trigger = 50
            video_path = work_dir/'video'/'train'
        env = gym.wrappers.RecordVideo(env, video_path,
                                        episode_trigger=lambda x: x % ep_trigger == 0,
                                        name_prefix=cfg.exp_name) # save video every 50 episode
    
    state_shape = env.observation_space.shape
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]
    
    return env, cfg

# Policy training function
def train_iteration(agent, env, ep,max_episode_steps=1000):
    # Run actual training        
    reward_sum, timesteps, done = 0, 0, False
    # Reset the environment and observe the initial state
    obs, _ = env.reset()
    while not done:
        
        # Sample action from policy
        action, act_logprob = agent.get_action(obs)

        # Perform the action on the environment, get new state and reward
        next_obs, reward, done, _, _ = env.step(to_numpy(action))

        # Store action's outcome (so that the agent can improve its policy)        
        if agent.name == 'pg':
            done_bool = done
            agent.record(obs, act_logprob, next_obs, reward, done_bool)
        elif agent.name == 'ddpg':
            # ignore the time truncated terminal signal
            done_bool = float(done) if timesteps < max_episode_steps else 0 
            agent.record(obs, action, next_obs, reward, done_bool)
            
        # Store total episode reward
        reward_sum += reward
        timesteps += 1
        
        if timesteps >= max_episode_steps:
            done = True
        # update observation
        obs = next_obs.copy()

    # update the policy after one episode
    info = agent.update()
    
    # Return stats of training
    info.update({'episode':ep,
                 'timesteps': timesteps,
                'ep_reward': reward_sum,
                })
    
    end = time.perf_counter()
    return info

def train(agent, cfg_path, cfg_args={}):
    
    env, cfg = setup(cfg_path, cfg_args=cfg_args)
    if cfg.save_logging:
        L = u.Logger() # create a simple logger to record stats
    start = time.perf_counter()
    for ep in range(cfg.train_episodes + 1):
        # collect data and update the policy
        train_info = train_iteration(agent, env,ep)

        if cfg.save_logging:
            L.log(**train_info)
        if (not cfg.silent) and (ep % 50 == 0):
            print({**train_info})        

    if cfg.save_model:
        agent.save(cfg.model_path)
        print("Saving model to", cfg.model_path)
        
    work_dir = Path().cwd()/'results'/cfg.env_name
    logging_path = work_dir / 'logging' / 'logging.pkl'    
    if cfg.save_logging:
        L.save(logging_path)

    end = time.perf_counter()
    train_time = (end-start)/60
    print('------ Training Finished ------')
    print(f'Total traning time is {train_time}mins')


# Function to test a trained policy
@torch.no_grad()
def test(agent, cfg_path, cfg_args={}, num_episode=10,max_episode_steps=1000):
    
    env, cfg = setup(cfg_path, cfg_args=cfg_args)
    
    
    if cfg.model_path == 'default':
        cfg.model_path = work_dir/'model'/f'{cfg.env_name}_params.pt'
    print("Loading model from", cfg.model_path, "...")

    # load model
    agent.load(cfg.model_path)

    print('Testing ...')
        
    total_test_reward = 0
    for ep in range(num_episode):
        (obs,_), done= env.reset(), False
        test_reward = 0
        timesteps = 0

        while not done:
            # Similar to the training loop above -
            # get the action, act on the environment, save total reward
            # (evaluation=True makes the agent always return what it thinks to be
            # the best action - there is no exploration at this point)
            action, _ = agent.get_action(obs, evaluation=True)
            #print(env.step(to_numpy(action)))
            obs, reward, done,_, _ = env.step(to_numpy(action))
            test_reward += reward
            timesteps += 1
        
            if timesteps >= max_episode_steps:
                done = True

        total_test_reward += test_reward
        print(f'Ep{ep}: Test ep_reward is {test_reward}')

    print("Average test reward:", total_test_reward/num_episode)


def plot(cfg_path, cfg_args={}, save_name=None):
    env, cfg = setup(cfg_path, cfg_args=cfg_args)

    # create folders if needed
    work_dir = Path().cwd()/'results'/cfg.env_name
    logging_path = work_dir / 'logging' / 'logging.pkl'
    plot_path = work_dir / f'{cfg.agent_name}.png'

    log_data = u.load_object(logging_path)
    
    
    Rs = log_data['ep_reward']
    Eps = log_data['episode']
    
    
    # Plotting the training loss against episode numbers
    plt.plot(Eps, Rs)
    plt.xlabel('Episode Number')
    plt.ylabel('Returns')
    plt.title('Task returns over Episodes')
    plt.grid(True)
    plt.savefig(plot_path)
    plt.show()