import numpy as np
import torch
from collections import deque
import re

class ReplayBuffer:
    def __init__(self, capacity, obs_space, full_keys, partial_keys):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.full_keys = full_keys
        self.partial_keys = partial_keys
        self.obs_space = obs_space
        
        # Pre-compile regex patterns for efficiency
        self.full_mlp_pattern = re.compile(full_keys.mlp_keys) if hasattr(full_keys, 'mlp_keys') else None
        self.full_cnn_pattern = re.compile(full_keys.cnn_keys) if hasattr(full_keys, 'cnn_keys') else None
        self.partial_mlp_pattern = re.compile(partial_keys.mlp_keys) if hasattr(partial_keys, 'mlp_keys') else None
        self.partial_cnn_pattern = re.compile(partial_keys.cnn_keys) if hasattr(partial_keys, 'cnn_keys') else None
        
    def add(self, transition):
        """Add a transition to the buffer.
        
        Args:
            transition: dict containing:
                - obs: dict of full observations
                - action: action taken
                - reward: reward received
                - next_obs: dict of next full observations
                - done: whether episode ended
        """
        self.buffer.append(transition)
    
    def _should_include_key(self, key, is_image):
        """Check if a key should be included based on the patterns."""
        if is_image:
            full_match = self.full_cnn_pattern and self.full_cnn_pattern.match(key)
            partial_match = self.partial_cnn_pattern and self.partial_cnn_pattern.match(key)
        else:
            full_match = self.full_mlp_pattern and self.full_mlp_pattern.match(key)
            partial_match = self.partial_mlp_pattern and self.partial_mlp_pattern.match(key)
        
        return full_match or partial_match
    
    def sample(self, batch_size):
        """Sample a batch of transitions.
        
        Returns:
            dict containing:
                - full_obs: dict of observations matching full_keys pattern
                - partial_obs: dict of observations matching partial_keys pattern
                - action: actions taken
                - reward: rewards received
                - next_full_obs: dict of next observations matching full_keys pattern
                - next_partial_obs: dict of next observations matching partial_keys pattern
                - done: whether episodes ended
        """
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        transitions = [self.buffer[i] for i in indices]
        
        # Initialize batch dictionaries
        batch = {
            'full_obs': {},
            'partial_obs': {},
            'next_full_obs': {},
            'next_partial_obs': {},
            'action': [],
            'reward': [],
            'done': []
        }
        
        # Get all observation keys from the first transition
        if transitions:
            all_keys = set(transitions[0]['obs'].keys())
            all_keys.update(transitions[0]['next_obs'].keys())
            
            # Process each key
            for key in all_keys:
                if key in ['reward', 'is_first', 'is_last', 'is_terminal']:
                    continue
                
                is_image = len(self.obs_space[key].shape) == 3 and self.obs_space[key].shape[-1] == 3
                
                # Check if key should be included in full observations
                if is_image:
                    is_full = self.full_cnn_pattern and self.full_cnn_pattern.match(key)
                else:
                    is_full = self.full_mlp_pattern and self.full_mlp_pattern.match(key)
                
                # Check if key should be included in partial observations
                if is_image:
                    is_partial = self.partial_cnn_pattern and self.partial_cnn_pattern.match(key)
                else:
                    is_partial = self.partial_mlp_pattern and self.partial_mlp_pattern.match(key)
                
                # Add to full observations if it matches full_keys pattern
                if is_full:
                    batch['full_obs'][key] = torch.stack([t['obs'][key] for t in transitions])
                    batch['next_full_obs'][key] = torch.stack([t['next_obs'][key] for t in transitions])
                
                # Add to partial observations if it matches partial_keys pattern
                if is_partial:
                    batch['partial_obs'][key] = torch.stack([t['obs'][key] for t in transitions])
                    batch['next_partial_obs'][key] = torch.stack([t['next_obs'][key] for t in transitions])
        
        # Process actions, rewards, and dones
        batch['action'] = torch.stack([t['action'] for t in transitions])
        batch['reward'] = torch.tensor([t['reward'] for t in transitions])
        batch['done'] = torch.tensor([t['done'] for t in transitions])
        
        return batch
    
    def __len__(self):
        return len(self.buffer)
