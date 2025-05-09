import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from ..shared.agent import BaseAgent
import numpy as np
import re
from thesis.embodied import Space
from thesis.teacher_student.encoder import DualEncoder, layer_init
import torch.nn.functional as F

class TeacherPolicy(BaseAgent):
    def __init__(self, envs, config, dual_encoder):
        super().__init__(envs, config, dual_encoder)
        
        # Get all observation keys
        self.all_keys = []
        for k in envs.obs_space.keys():
            if k not in ['reward', 'is_first', 'is_last', 'is_terminal']:
                self.all_keys.append(k)
        
        # Check if action space is discrete
        self.is_discrete = envs.act_space['action'].discrete
        
        # Initialize critic
        self.critic = nn.Sequential(
            layer_init(nn.Linear(config.encoder.output_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0)
        )
        
        # Initialize actor
        if self.is_discrete:
            self.actor = nn.Sequential(
                layer_init(nn.Linear(config.encoder.output_dim, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, envs.act_space['action'].shape[0]), std=0.01)
            )
        else:
            action_size = np.prod(envs.act_space['action'].shape)
            self.actor_mean = nn.Sequential(
                layer_init(nn.Linear(config.encoder.output_dim, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, action_size), std=0.01)
            )
            self.actor_logstd = nn.Parameter(torch.zeros(1, action_size))
    
    def encode_observations(self, x):
        """Encode observations using teacher encoder."""
        return self.dual_encoder.encode_teacher_observations(x)
    
    def get_value(self, x):
        latent = self.encode_observations(x)
        return self.critic(latent)
    
    def get_action_and_value(self, x, action=None):
        """Get action and value from policy.
        
        Args:
            x: Dictionary of observations
            action: Optional action for computing log probability
            
        Returns:
            action: Action to take
            logprob: Log probability of action
            entropy: Entropy of action distribution
            value: Value estimate
            imitation_losses: Dictionary of imitation losses
        """
        # Get base action and value
        action, logprob, entropy, value = super().get_action_and_value(x, action)
        
        # Compute imitation losses if enabled
        imitation_losses = {}
        if self.config.encoder.teacher_to_student_imitation and self.config.encoder.teacher_to_student_lambda > 0:
            # Get teacher's latent representation
            teacher_latent = self.encode_observations(x)
            # Get student's latent representation with stop gradient
            with torch.no_grad():
                student_latent = self.dual_encoder.encode_student_observations(x)
            imitation_losses['teacher_to_student'] = self.dual_encoder.compute_teacher_to_student_loss(teacher_latent, student_latent)
        
        return action, logprob, entropy, value, imitation_losses

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
