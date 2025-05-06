import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from ..shared.agent import BaseAgent
import numpy as np
import re

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class StudentPolicy(BaseAgent):
    def __init__(self, envs, config):
        super().__init__(envs, config)
        # Student uses partial observations
        self.obs_keys = config.keys
        
        # Get observation space and keys
        obs_space = envs.obs_space
        
        # Get encoder config
        encoder_config = config.encoder
        
        # Filter keys based on the regex patterns for student
        self.mlp_keys = []
        self.cnn_keys = []
        for k in obs_space.keys():
            if k in ['reward', 'is_first', 'is_last', 'is_terminal']:
                continue
            if len(obs_space[k].shape) == 3 and obs_space[k].shape[-1] == 3:  # Image observations
                if re.match(config.keys.cnn_keys, k):
                    self.cnn_keys.append(k)
            else:  # Non-image observations
                if re.match(config.keys.mlp_keys, k):
                    self.mlp_keys.append(k)
        
        # Filter keys based on the regex patterns for teacher
        self.all_mlp_keys = []
        self.all_cnn_keys = []
        for k in obs_space.keys():
            if k in ['reward', 'is_first', 'is_last', 'is_terminal']:
                continue
            if len(obs_space[k].shape) == 3 and obs_space[k].shape[-1] == 3:  # Image observations
                if re.match(config.full_keys.cnn_keys, k):
                    self.all_cnn_keys.append(k)
            else:  # Non-image observations
                if re.match(config.full_keys.mlp_keys, k):
                    self.all_mlp_keys.append(k)
        
        # Calculate total input size for MLP
        self.total_mlp_size = 0
        self.mlp_key_sizes = {}  # Store the size of each MLP key
        for key in self.mlp_keys:
            if isinstance(obs_space[key].shape, tuple):
                size = np.prod(obs_space[key].shape)
            else:
                size = 1
            self.mlp_key_sizes[key] = size
            self.total_mlp_size += size

        # Initialize activation function
        if encoder_config.act == 'silu':
            self.act = nn.SiLU()
        elif encoder_config.act == 'relu':
            self.act = nn.ReLU()
        elif encoder_config.act == 'tanh':
            self.act = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation function: {encoder_config.act}")

        # Calculate CNN output dimension
        if self.cnn_keys:
            # Calculate number of stages based on minres
            input_size = 64  # From config.env.atari.size
            stages = int(np.log2(input_size) - np.log2(encoder_config.minres))
            final_depth = encoder_config.cnn_depth * (2 ** (stages - 1))
            cnn_output_dim = final_depth * encoder_config.minres * encoder_config.minres
            
            # CNN encoder for image observations
            self.cnn_encoder = nn.Sequential(
                ImageEncoderResnet(
                    depth=encoder_config.cnn_depth,
                    blocks=encoder_config.cnn_blocks,
                    resize=encoder_config.resize,
                    minres=encoder_config.minres,
                    output_dim=cnn_output_dim
                ),
                nn.LayerNorm(cnn_output_dim) if encoder_config.norm == 'layer' else nn.Identity()
            )
        else:
            self.cnn_encoder = None
            cnn_output_dim = 0

        # MLP encoder for non-image observations
        if self.mlp_keys:
            layers = []
            input_dim = self.total_mlp_size
            
            # Add MLP layers
            for _ in range(encoder_config.mlp_layers):
                layers.extend([
                    layer_init(nn.Linear(input_dim, encoder_config.mlp_units)),
                    self.act,
                    nn.LayerNorm(encoder_config.mlp_units) if encoder_config.norm == 'layer' else nn.Identity()
                ])
                input_dim = encoder_config.mlp_units
            
            self.mlp_encoder = nn.Sequential(*layers)
        else:
            self.mlp_encoder = None
        
        # Calculate total input dimension for latent projector
        total_input_dim = (cnn_output_dim if self.cnn_encoder is not None else 0) + (encoder_config.mlp_units if self.mlp_encoder is not None else 0)
        
        # Project concatenated features to latent space
        self.latent_projector = nn.Sequential(
            layer_init(nn.Linear(total_input_dim, encoder_config.output_dim)),
            self.act,
            nn.LayerNorm(encoder_config.output_dim) if encoder_config.norm == 'layer' else nn.Identity(),
            layer_init(nn.Linear(encoder_config.output_dim, encoder_config.output_dim)),
            self.act,
            nn.LayerNorm(encoder_config.output_dim) if encoder_config.norm == 'layer' else nn.Identity()
        )
        
        # Determine if action space is discrete or continuous
        self.is_discrete = envs.act_space['action'].discrete
        
        # Actor and critic networks operating on latent space
        self.critic = nn.Sequential(
            layer_init(nn.Linear(encoder_config.output_dim, encoder_config.output_dim // 2)),
            self.act,
            layer_init(nn.Linear(encoder_config.output_dim // 2, 1), std=1.0),
        )
        
        if self.is_discrete:
            self.actor = nn.Sequential(
                layer_init(nn.Linear(encoder_config.output_dim, encoder_config.output_dim // 2)),
                self.act,
                layer_init(nn.Linear(encoder_config.output_dim // 2, envs.act_space['action'].shape[0]), std=0.01),
            )
        else:
            action_size = np.prod(envs.act_space['action'].shape)
            self.actor_mean = nn.Sequential(
                layer_init(nn.Linear(encoder_config.output_dim, encoder_config.output_dim // 2)),
                self.act,
                layer_init(nn.Linear(encoder_config.output_dim // 2, action_size), std=0.01),
            )
            self.actor_logstd = nn.Parameter(torch.zeros(1, action_size))
    
    def encode_observations(self, x):
        if isinstance(x, dict):
            # Process CNN observations
            cnn_features = None
            if self.cnn_keys and self.cnn_encoder is not None:
                # Stack all CNN observations along channels
                cnn_inputs = []
                for key in self.cnn_keys:
                    if key not in x:  # Skip if key doesn't exist
                        continue
                    batch_size = x[key].shape[0]
                    img = x[key].permute(0, 3, 1, 2) / 255.0  # Convert to [B, C, H, W] and normalize
                    cnn_inputs.append(img)
                
                if cnn_inputs:  # Only process if we have any CNN features
                    # Stack along channel dimension
                    cnn_input = torch.cat(cnn_inputs, dim=1)
                    cnn_features = self.cnn_encoder(cnn_input)
            
            # Process MLP observations
            mlp_features = None
            if self.mlp_keys and self.mlp_encoder is not None:
                mlp_features = []
                batch_size = None
                for key in self.mlp_keys:
                    if key not in x:  # Skip if key doesn't exist
                        continue
                    if batch_size is None:
                        batch_size = x[key].shape[0]
                    # Ensure the observation is flattened
                    if x[key].dim() > 2:
                        mlp_features.append(x[key].view(batch_size, -1))
                    else:
                        mlp_features.append(x[key])
                
                if mlp_features:  # Only process if we have any MLP features
                    mlp_features = torch.cat(mlp_features, dim=1)
                    # Ensure the input size matches what the MLP encoder expects
                    if mlp_features.shape[1] != self.total_mlp_size:
                        raise ValueError(f"MLP input size mismatch. Expected {self.total_mlp_size}, got {mlp_features.shape[1]}")
                    mlp_features = self.mlp_encoder(mlp_features)
                        
            # Handle the case where neither exists
            if cnn_features is None and mlp_features is None:
                raise ValueError("No valid observations found in input dictionary")
            
            # Concatenate features if both exist, otherwise use whichever exists
            if cnn_features is not None and mlp_features is not None:
                features = torch.cat([cnn_features, mlp_features], dim=1)
            elif cnn_features is not None:
                features = cnn_features
            else:  # mlp_features is not None
                features = mlp_features
        else:
            # Handle tensor input (assumed to be MLP features)
            if self.mlp_encoder is None:
                raise ValueError("MLP encoder not initialized but received tensor input")
            batch_size = x.shape[0]
            if x.dim() > 2:
                x = x.view(batch_size, -1)
            features = self.mlp_encoder(x)
                
        # Project to final latent space
        latent = self.latent_projector(features)
        return latent

    def get_value(self, x):
        latent = self.encode_observations(x)
        return self.critic(latent)

    def get_action_and_value(self, x, action=None):
        latent = self.encode_observations(x)
        
        if self.is_discrete:
            logits = self.actor(latent)
            probs = Categorical(logits=logits)
            if action is None:
                action = probs.sample()
                # Convert to one-hot for environment
                one_hot_action = torch.zeros_like(logits)
                one_hot_action.scatter_(1, action.unsqueeze(1), 1.0)
                return one_hot_action, probs.log_prob(action), probs.entropy(), self.critic(latent)
            else:
                # Convert one-hot action back to indices for log_prob calculation
                action_indices = action.argmax(dim=1)
                return action, probs.log_prob(action_indices), probs.entropy(), self.critic(latent)
        else:
            action_mean = self.actor_mean(latent)
            action_logstd = self.actor_logstd.expand_as(action_mean)
            action_std = torch.exp(action_logstd)
            probs = Normal(action_mean, action_std)
            if action is None:
                action = probs.sample()
            return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(latent)

    def train_bc(self, replay_buffer, batch_size, optimizer, teacher_policy):
        """Train the student policy using behavioral cloning.
        
        Args:
            replay_buffer: ReplayBuffer containing student trajectories
            batch_size: Batch size for training
            optimizer: Optimizer for the student policy
            teacher_policy: Teacher policy to get actions for BC loss
            
        Returns:
            dict containing training metrics
        """
        # Sample batch from replay buffer
        batch = replay_buffer.sample(batch_size)
        
        # Get student's action predictions
        _, student_log_probs, _, _ = self.get_action_and_value(batch['partial_obs'])
        
        # Get teacher's actions on the same observations
        with torch.no_grad():
            teacher_actions, _, _, _ = teacher_policy.get_action_and_value(batch['full_obs'])
        
        # Compute loss
        if self.is_discrete:
            # For discrete actions, use cross entropy loss
            loss = -student_log_probs.mean()
        else:
            # For continuous actions, use MSE loss
            student_actions = self.actor_mean(self.encode_observations(batch['partial_obs']))
            loss = ((student_actions - teacher_actions) ** 2).mean()
        
        # Update policy
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return {
            'bc_loss': loss.item(),
            'student_actions': student_actions.detach() if not self.is_discrete else None,
            'teacher_actions': teacher_actions.detach()
        }
    
    def collect_transitions(self, envs, num_steps):
        """Collect transitions from the environment using the student policy.
        
        Args:
            envs: Vectorized environment
            num_steps: Number of steps to collect
            
        Returns:
            list of transitions, each containing:
                - obs: dict of observations (both partial and full)
                - action: action taken
                - reward: reward received
                - next_obs: dict of next observations (both partial and full)
                - done: whether episode ended
        """
        transitions = []
        
        # Get all keys needed for both student and teacher
        all_keys = set(self.mlp_keys + self.cnn_keys + self.all_mlp_keys + self.all_cnn_keys)
        
        # Initialize observation storage
        obs = {}
        for key in all_keys:
            if key in self.mlp_keys or key in self.all_mlp_keys:
                size = np.prod(envs.obs_space[key].shape)
                obs[key] = torch.zeros((num_steps, self.config.num_envs, size)).to(self.device)
            else:  # CNN keys
                obs[key] = torch.zeros((num_steps, self.config.num_envs) + envs.obs_space[key].shape).to(self.device)
        
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
        for key in all_keys:
            next_obs[key] = torch.Tensor(obs_dict[key].astype(np.float32)).to(self.device)
        next_done = torch.Tensor(obs_dict['is_last'].astype(np.float32)).to(self.device)
        
        # Collect transitions
        for step in range(num_steps):
            # Store observations
            for key in all_keys:
                obs[key][step] = next_obs[key]
            dones[step] = next_done
            
            # Get action from policy using only student's observation keys
            student_obs = {key: next_obs[key] for key in self.mlp_keys + self.cnn_keys}
            with torch.no_grad():
                action, _, _, _ = self.get_action_and_value(student_obs)
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
            for key in all_keys:
                next_obs[key] = torch.Tensor(obs_dict[key].astype(np.float32)).to(self.device)
            next_done = torch.Tensor(obs_dict['is_last'].astype(np.float32)).to(self.device)
            rewards[step] = torch.tensor(obs_dict['reward'].astype(np.float32)).to(self.device)
            
            # Store transition
            for env_idx in range(self.config.num_envs):
                transition = {
                    'obs': {key: obs[key][step, env_idx].cpu().numpy() for key in all_keys},
                    'action': actions[step, env_idx].cpu().numpy(),
                    'reward': rewards[step, env_idx].cpu().numpy(),
                    'next_obs': {key: next_obs[key][env_idx].cpu().numpy() for key in all_keys},
                    'done': next_done[env_idx].cpu().numpy()
                }
                transitions.append(transition)
        
        return transitions
