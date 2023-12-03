from .agent_base import BaseAgent
from .ppo_utils import Policy
import utils.common_utils as cu
import torch
import numpy as np
import torch.nn.functional as F
import time


class PPOAgent(BaseAgent):
    def __init__(self, config=None):
        try:
            if config['seed'] is not None:
                print("Setting seed to", config['seed'])
                torch.manual_seed(config['seed'])
                np.random.seed(config['seed'])
        except:
            pass
        super(PPOAgent, self).__init__(config)
        self.device = self.cfg.device  # ""cuda" if torch.cuda.is_available() else "cpu"
        self.seed = self.cfg.seed
        self.state_dim = self.observation_space_dim
        self.action_dim = self.action_space_dim
        self.max_action = self.cfg.max_action

        self.policy = Policy(self.state_dim, self.action_dim, self.max_action).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=3e-4)
        self.lr=self.cfg.lr
        self.name = 'ppo'

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
        if not self.silent:
            print("Updating the policy...")

        self.states = torch.stack(self.states)
        self.actions = torch.stack(self.actions).squeeze()
        self.next_states = torch.stack(self.next_states)
        self.rewards = torch.stack(self.rewards).squeeze()
        self.dones = torch.stack(self.dones).squeeze()
        self.action_log_probs = torch.stack(self.action_log_probs).squeeze()

        for e in range(self.epochs):
            self.ppo_epoch()

        # Clear the replay buffer
        self.states = []
        self.actions = []
        self.next_states = []
        self.rewards = []
        self.dones = []
        self.action_log_probs = []
        if not self.silent:
            print("Updating finished!")
        return

    def ppo_epoch(self):
        indices = list(range(len(self.states)))
        returns = self.compute_returns()
        while len(indices) >= self.batch_size:
            # Sample a minibatch
            batch_indices = np.random.choice(indices, self.batch_size,
                    replace=False)

            # Do the update
            self.ppo_update(self.states[batch_indices], self.actions[batch_indices],
                self.rewards[batch_indices], self.next_states[batch_indices],
                self.dones[batch_indices], self.action_log_probs[batch_indices],
                returns[batch_indices])

            # Drop the batch indices
            indices = [i for i in indices if i not in batch_indices]

    def ppo_update(self, states, actions, rewards, next_states, dones, old_log_probs, targets):
        action_dists, values = self.policy(states)
        values = values.squeeze()
        new_action_probs = action_dists.log_prob(actions)
        ratio = torch.exp(new_action_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1-self.clip, 1+self.clip)

        advantages = targets - values
        advantages -= advantages.mean()
        advantages /= advantages.std()+1e-8
        advantages = advantages.detach()
        policy_objective = -torch.min(ratio*advantages, clipped_ratio*advantages)

        value_loss = F.smooth_l1_loss(values, targets, reduction="mean")

        policy_objective = policy_objective.mean()
        entropy = action_dists.entropy().mean()
        loss = policy_objective + 0.5*value_loss - 0.01*entropy

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def compute_returns(self):
        returns = []
        with torch.no_grad():
            _, values = self.policy(self.states)
            _, next_values = self.policy(self.next_states)
            values = values.squeeze()
            next_values = next_values.squeeze()
        gaes = torch.zeros(1)
        timesteps = len(self.rewards)
        for t in range(timesteps-1, -1, -1):
            deltas = self.rewards[t] + self.gamma * next_values[t] *\
                     (1-self.dones[t]) - values[t]
            gaes = deltas + self.gamma*self.tau*(1-self.dones[t])*gaes
            returns.append(gaes + values[t])

        return torch.Tensor(list(reversed(returns)))
    


    def get_action(self, observation, evaluation=False):
        if observation.ndim == 1: observation = observation[None] # add the batch dimension
        try:
            x = torch.from_numpy(observation).float().to(self.device)
        except:
            x = observation
        action_dist, _ = self.policy.forward(x)
        if evaluation:
            # Use the mean of the distribution as the action during evaluation
            # mean is the most likely action in Gaussian distribution
            action = action_dist.mean
            
            action = action.detach()
        else:
            # Sample an action from the distribution during training
            action = action_dist.sample()
        # print("Returned by Policy", action)
        # aprob = action_dist.log_prob(action).sum(dim=-1).squeeze()
        
        #print(type(action))
        #print(action)
        
        # action = self.max_action * torch.tanh(action)
        # action = action.clamp(-self.max_action, self.max_action)
        aprob = action_dist.log_prob(action)
        # action = action.item()
        print("Processed", action)
        print(action)
        print(type(action))
        print(action.shape)
        return action, aprob

    def train_iteration(self,ratio_of_episodes):
        # Run actual training        
        reward_sum, episode_length, num_updates = 0, 0, 0
        done = False

        # Reset the environment and observe the initial state
        observation, _  = self.env.reset(seed=self.seed)

        while not done and episode_length < self.cfg.max_episode_steps:
            # Get action from the agent
            action, action_log_prob = self.get_action(observation)
            action = action.flatten()
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
                # self.policy.set_logstd_ratio( 3.912 * ratio_of_episodes - 4.605)
                self.policy.set_logstd_ratio( 4.605 * ratio_of_episodes - 4.605)

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
            ratio_of_episodes = (self.cfg.train_episodes - ep) / self.cfg.train_episodes + 1e-8
            train_info = self.train_iteration(ratio_of_episodes)
            train_info.update({'episodes': ep})
            total_step+=train_info['episode_length']
            train_info.update({'total_step': total_step})
            run_episode_reward.append(train_info['ep_reward'])
            
            logstd = self.policy.actor_logstd
            
            if total_step%self.cfg.log_interval==0:
                average_return=sum(run_episode_reward)/len(run_episode_reward)
                if not self.cfg.silent:
                    # print(f"Episode {ep} Step {total_step} finished. Average episode return: {average_return} ({train_info['episode_length']} episode_length, {logstd} logstd)")
                    print(f"Episode {ep} Step {total_step} finished. Average episode return: {average_return} ({train_info['episode_length']} episode_length)")
                    # logstd is a torch tensor
                    print("logstd: ", logstd.detach().cpu().numpy(), " | std: ", np.exp(logstd.detach().cpu().numpy()))

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

    def store_outcome(self, state, action, next_state, reward, action_log_prob, done):
        self.states.append(torch.from_numpy(state).float())
        # self.actions.append(torch.Tensor([action]))
        self.actions.append(action)
        self.action_log_probs.append(action_log_prob.detach())
        self.rewards.append(torch.Tensor([reward]).float())
        self.dones.append(torch.Tensor([done]))
        self.next_states.append(torch.from_numpy(next_state).float())
        
    def load_model(self):
        filepath=str(self.model_dir)+'/model_parameters_'+str(self.seed)+'.pt'
        state_dict = torch.load(filepath)
        self.policy.load_state_dict(state_dict)
    
    def save_model(self):
        filepath=str(self.model_dir)+'/model_parameters_'+str(self.seed)+'.pt'
        torch.save(self.policy.state_dict(), filepath)
        print("Saved model to", filepath, "...")