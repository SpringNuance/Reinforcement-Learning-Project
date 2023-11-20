from .agent_base import BaseAgent
from .ppo_utils import Policy
import utils.common_utils as cu
import torch
import numpy as np
import torch.nn.functional as F
import time

class PPOAgent(BaseAgent):
    def __init__(self, config=None):
        super(PPOAgent, self).__init__(config)
        self.device = self.cfg.device  # ""cuda" if torch.cuda.is_available() else "cpu"
        self.policy=
        self.lr=self.cfg.lr

        self.batch_size = self.cfg.batch_size
        self.gamma = self.cfg.gamma
        self.tau = self.cfg.tau
        self.clip = self.cfg.clip
        self.epochs = self.cfg.epochs
        self.running_mean = None
        self.states = []
        self.actions = []
        self.next_states = []
        self.rewards = []
        self.dones = []
        self.action_log_probs = []
        self.silent = self.cfg.silent

    def update_policy(self):
       


    def get_action(self, observation, evaluation=False):
        return


        
    def train_iteration(self,ratio_of_episodes):
        # Run actual training        
        reward_sum, episode_length, num_updates = 0, 0, 0
        done = False

        # Reset the environment and observe the initial state
        observation, _  = self.env.reset()

        while not done and episode_length < self.cfg.max_episode_steps:
            # Get action from the agent
            action = 
            previous_observation = observation.copy()

            # Perform the action on the environment, get new state and reward
            observation, reward, done, _, _ = self.env.step(action)
            
            # Store action's outcome (so that the agent can improve its policy)
            self.store_outcome(previous_observation, action, observation,
                                reward, action_log_prob, done)

            # Store total episode reward
            reward_sum += reward
            episode_length += 1

            # Update the policy, if we have enough data
            if len(self.states) > self.cfg.min_update_samples:
                self.update_policy()
                num_updates += 1

                # Update policy randomness
                self.policy.set_logstd_ratio(ratio_of_episodes)

        # Return stats of training
        update_info = {'episode_length': episode_length,
                    'ep_reward': reward_sum}
        return update_info
        
    def train(self):
        if self.cfg.save_logging: 
            L = cu.Logger() # create a simple logger to record stats
        total_step=0
        run_episode_reward=[]
        start = time.perf_counter()

        for ep in range(self.cfg.train_episodes+1):
            ratio_of_episodes = (self.cfg.train_episodes - ep) / self.cfg.train_episodes
            train_info = self.train_iteration(ratio_of_episodes)
            train_info.update({'episodes': ep})
            total_step+=train_info['episode_length']
            train_info.update({'total_step': total_step})
            run_episode_reward.append(train_info['ep_reward'])
            
            logstd = self.policy.actor_logstd
            
            if total_step%self.cfg.log_interval==0:
                average_return=sum(run_episode_reward)/len(run_episode_reward)
                if not self.cfg.silent:
                    print(f"Episode {ep} Step {total_step} finished. Average episode return: {average_return} ({train_info['episode_length']} episode_length, {logstd} logstd)")

                if self.cfg.save_logging:
                    train_info.update({'average_return':average_return})
                    L.log(**train_info)
                run_episode_reward=[]

        # Save the model
        if self.cfg.save_model:
            self.save_model()

        logging_path = str(self.logging_dir)+'/logs'   
        if self.cfg.save_logging:
            L.save(logging_path, self.seed)
        self.env.close()
        
        end = time.perf_counter()
        train_time = (end-start)/60
        print("------Training finished.------")
        print(f'Total traning time is {train_time}mins')
    
    def load_model(self):
        filepath=str(self.model_dir)+'/model_parameters_'+str(self.seed)+'.pt'
        state_dict = torch.load(filepath)
        self.policy.load_state_dict(state_dict)
    
    def save_model(self):
        filepath=str(self.model_dir)+'/model_parameters_'+str(self.seed)+'.pt'
        torch.save(self.policy.state_dict(), filepath)
        print("Saved model to", filepath, "...")
