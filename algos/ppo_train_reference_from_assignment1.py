import yaml, time, wandb, torch
import gymnasium as gym
import utils as u
from agent import Agent, Policy
import numpy as np 

from pathlib import Path
work_dir = Path().cwd()/'results'

class Struct:
    def __init__(self, **entries):
        self.entries = entries
        self.__dict__.update(entries)
    
    def __str__(self):
        return str(self.entries)
        
def setup(cfg_path, cfg_args={}, print_info=False):
    
    with open(cfg_path, 'r') as f:
        d = yaml.safe_load(f)
        d.update(cfg_args)
        cfg = Struct(**d)
    
    # Setting library seeds
    if cfg.seed == None:
        seed = np.random.randint(low=1, high=1000)
    else:
        seed = cfg.seed
    
    print("Numpy/Torch/Random Seed: ", seed)
    u.set_seed(seed) # set seed
    
    run_id = int(time.time())

    # create folders if needed
    if cfg.save_model: u.make_dir(work_dir/"model")
    if cfg.save_logging:
        u.make_dir(work_dir/"logging")
        L = u.Logger() # create a simple logger to record stats

    # use wandb to store stats; we aren't currently logging anything into wandb during testing (might be useful to
    # have the cfg.testing check here if someone forgets to set use_wandb=false)
    if cfg.use_wandb and not cfg.testing:
        wandb.init(project="rl_aalto",
                    name=f'{cfg.exp_name}-{cfg.env_name}-{str(cfg.seed)}-{str(run_id)}',
                    group=f'{cfg.exp_name}-{cfg.env_name}',
                    config=cfg)

    # create env
    env = gym.make(cfg.env_name, 
                    max_episode_steps=cfg.max_episode_steps,
                    render_mode='rgb_array')

    if cfg.save_video:
        # During testing, save every episode
        if cfg.testing:
            ep_trigger = 1
            video_path = work_dir/'video'/cfg.env_name/'test'
        # During training, save every 50th episode
        else:
            ep_trigger = 50
            video_path = work_dir/'video'/cfg.env_name/'train'
            
        env = gym.wrappers.RecordVideo(env, video_path,
                                        episode_trigger=lambda x: x % ep_trigger == 0, # save video every 50 episode
                                        name_prefix=cfg.exp_name, disable_logger=True)
    # Get dimensionalities of actions and observations
    action_space_dim = u.get_space_dim(env.action_space)
    observation_space_dim = u.get_space_dim(env.observation_space)

    # Instantiate agent and its policy
    policy = Policy(observation_space_dim, action_space_dim)
    agent = Agent(policy, cfg.batch_size)
    
    # Print some stuff
    if print_info:
        print("Configuration Settings:", cfg)
        print("Training device:", agent.train_device)
        print("Observation space dimensions:", observation_space_dim)
        print("Action space dimensions:", action_space_dim)
        print()
    return env, policy, agent, cfg
    
# Policy training function
def train_iteration(agent, env, min_update_samples=2000, max_episode_steps=200, seed=None):
    # Run actual training        
    reward_sum, timesteps, num_updates = 0, 0, 0
    done = False

    # Reset the environment and observe the initial state
    observation, _  = env.reset(seed=seed)

    while not done and timesteps < max_episode_steps:
        # Get action from the agent
        action, action_log_prob = agent.get_action(observation)
        previous_observation = observation.copy()

        # Perform the action on the environment, get new state and reward
        observation, reward, done, _, _ = env.step(action)

        # Store action's outcome (so that the agent can improve its policy)
        agent.store_outcome(previous_observation, action, observation,
                reward, action_log_prob, done)

        # Store total episode reward
        reward_sum += reward
        timesteps += 1

        # Update the policy, if we have enough data
        if len(agent.states) > min_update_samples:
            agent.update_policy()
            num_updates += 1

    # Return stats of training
    update_info = {'timesteps': timesteps,
            'ep_reward': reward_sum,
            'num_updates': num_updates}
    return update_info

# Training
def train(cfg_path, cfg_args={}):
        
    env, policy, agent, cfg = setup(cfg_path, cfg_args=cfg_args, print_info=True)

    if cfg.save_logging: 
        L = u.Logger() # create a simple logger to record stats
    
    for ep in range(cfg.train_episodes+1):
        train_info = train_iteration(agent, env, min_update_samples=cfg.min_update_samples, max_episode_steps=cfg.max_episode_steps, seed=cfg.seed)
        train_info.update({'episodes': ep})

        if not cfg.silent:
            print(f"Episode {ep} finished. Total reward: {train_info['ep_reward']} ({train_info['timesteps']} timesteps)")

        if cfg.use_wandb: 
            wandb.log(train_info)
        if cfg.save_logging:
            L.log(**train_info)

    # Save the model
    if cfg.save_model:
        model_path = work_dir/'model'/f'{cfg.env_name}_params.pt'
        torch.save(policy.state_dict(), model_path)
        print("Model saved to", model_path)

    if cfg.save_logging:
        logging_path = work_dir/'logging'/f'{cfg.env_name}_logging.pkl'
        L.save(logging_path)

    print("------Training finished.------")
    
# Function to test a trained policy
    
def test(episodes, cfg_path, cfg_args={}):
    
    env, policy, agent, cfg  = setup(cfg_path, cfg_args=cfg_args, print_info=False)
    
    # Testing 
    if cfg.model_path == 'default':
        cfg.model_path = work_dir/'model'/f'{cfg.env_name}_params.pt'
    print("Loading model from", cfg.model_path, "...")

    # load model
    state_dict = torch.load(cfg.model_path)
    policy.load_state_dict(state_dict)

    print("Testing...")
    total_test_reward, total_test_len = 0, 0
    for ep in range(episodes):
        done = False
        if cfg.seed == None:
            seed = np.random.randint(low=1, high=1000)
        else:
            seed = cfg.seed
            
        observation, _ = env.reset(seed=seed)

        test_reward, test_len = 0, 0
        while not done:
            # Similar to the training loop above -
            # get the action, act on the environment, save total reward
            # (evaluation=True makes the agent always return what it thinks to be
            # the best action - there is no exploration at this point)
            action, _ = agent.get_action(observation, evaluation=True)
            observation, reward, done, truncated, info = env.step(action)

            test_reward += reward
            test_len += 1
        total_test_reward += test_reward
        total_test_len += test_len
        print("Test ep reward:", test_reward, "seed:", seed)
    print("Average test reward:", total_test_reward/episodes, "episode length:", total_test_len/episodes)