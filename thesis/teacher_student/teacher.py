import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from ..shared.agent import BaseAgent
import numpy as np

class TeacherPolicy(BaseAgent):
    def __init__(self, envs, config):
        super().__init__(envs, config)
        # Teacher uses full observations
        self.obs_keys = config.full_keys
        
        # Get all observation keys
        self.all_keys = []
        for k in envs.obs_space.keys():
            if k not in ['reward', 'is_first', 'is_last', 'is_terminal']:
                self.all_keys.append(k)
        
        # Get observation space and keys
        obs_space = envs.obs_space
        
    def collect_transitions(self, envs, num_steps):
        """Collect transitions from the environment using the teacher policy.
        
        Args:
            envs: Vectorized environment
            num_steps: Number of steps to collect
            
        Returns:
            list of transitions, each containing:
                - obs: dict of full observations
                - action: action taken
                - reward: reward received
                - next_obs: dict of next full observations
                - done: whether episode ended
        """
        transitions = []
        
        # Initialize observation storage
        obs = {}
        for key in self.all_keys:
            if len(envs.obs_space[key].shape) == 3 and envs.obs_space[key].shape[-1] == 3:  # Image observations
                obs[key] = torch.zeros((num_steps, self.config.num_envs) + envs.obs_space[key].shape).to(self.device)
            else:  # Non-image observations
                size = np.prod(envs.obs_space[key].shape)
                obs[key] = torch.zeros((num_steps, self.config.num_envs, size)).to(self.device)
        
        # Initialize action storage
        if self.is_discrete:
            action_shape = (num_steps, self.config.num_envs, envs.act_space['action'].shape[0])
        else:
            action_shape = (num_steps, self.config.num_envs) + envs.act_space['action'].shape
        actions = torch.zeros(action_shape).to(self.device)
        
        # Initialize reward and done storage
        rewards = torch.zeros((num_steps, self.config.num_envs)).to(self.device)
        dones = torch.zeros((num_steps, self.config.num_envs)).to(self.device)
        
        # Initialize actions with zeros and reset flags
        action_shape = envs.act_space['action'].shape
        acts = {
            'action': np.zeros((self.config.num_envs,) + action_shape, dtype=np.float32),
            'reset': np.ones(self.config.num_envs, dtype=bool)  # Reset all environments initially
        }
        
        # Get initial observations using step with reset flags
        obs_dict = envs.step(acts)
        next_obs = {}
        for key in self.all_keys:
            next_obs[key] = torch.Tensor(obs_dict[key].astype(np.float32)).to(self.device)
        next_done = torch.Tensor(obs_dict['is_last'].astype(np.float32)).to(self.device)
        
        # Collect transitions
        for step in range(num_steps):
            # Store observations
            for key in self.all_keys:
                obs[key][step] = next_obs[key]
            dones[step] = next_done
            
            # Get action from policy
            with torch.no_grad():
                action, _, _, _ = self.get_action_and_value(next_obs)
                actions[step] = action
            
            # Step environment
            action_np = action.cpu().numpy()
            if self.is_discrete:
                action_np = action_np.reshape(self.config.num_envs, -1)
            
            acts = {
                'action': action_np,
                'reset': next_done.cpu().numpy()
            }
            
            obs_dict = envs.step(acts)
            
            # Process observations
            for key in self.all_keys:
                next_obs[key] = torch.Tensor(obs_dict[key].astype(np.float32)).to(self.device)
            next_done = torch.Tensor(obs_dict['is_last'].astype(np.float32)).to(self.device)
            rewards[step] = torch.tensor(obs_dict['reward'].astype(np.float32)).to(self.device)
            
            # Store transition
            for env_idx in range(self.config.num_envs):
                transition = {
                    'obs': {key: obs[key][step, env_idx].cpu().numpy() for key in self.all_keys},
                    'action': actions[step, env_idx].cpu().numpy(),
                    'reward': rewards[step, env_idx].cpu().numpy(),
                    'next_obs': {key: next_obs[key][env_idx].cpu().numpy() for key in self.all_keys},
                    'done': next_done[env_idx].cpu().numpy()
                }
                transitions.append(transition)
        
        return transitions
