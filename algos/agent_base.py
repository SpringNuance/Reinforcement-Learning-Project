
class BaseAgent(object):
    def __init__(self, config=None):
        self.cfg=config["args"]
        self.env=config["env"]
        self.eval_env=config["eval_env"]
        self.action_space_dim=config["action_space_dim"] # 2
        self.observation_space_dim=config["observation_space_dim"] # 6
        self.train_device=self.cfg.device # default as "cpu"
        self.seed=config["seed"]
        self.algo_name=self.cfg.algo_name
        self.env_name=self.cfg.env_name
        self.max_action=self.cfg.max_action
        
        self.work_dir=self.cfg.work_dir# project_folder/results/env_name/algo_name/
        self.model_dir=self.cfg.model_dir
        self.logging_dir=self.cfg.logging_dir
        self.video_train_dir=self.cfg.video_train_dir
        self.video_test_dir=self.cfg.video_test_dir
        
    def get_action(self, observation, evaluation=False):
        """Given an observation, we will use this function to output an action."""
        raise NotImplementedError()
    
    def load_model(self):
        """Load the pre-trained model from the default model directory."""
        raise NotImplementedError()
    
    def save_model(self):
        """Save the trained models to the default model directory, for example, your value network
        and policy network. However, it depends on your agent/algorithm to decide what kinds of models
        to store."""
        raise NotImplementedError()
    
    def train(self):
        raise NotImplementedError()