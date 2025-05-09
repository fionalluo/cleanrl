import torch
import torch.nn as nn
import numpy as np
from thesis.shared.nets import ImageEncoderResnet
import re

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Encoder(nn.Module):
    def __init__(self, obs_space, config, is_teacher=True):
        super().__init__()
        self.config = config
        self.is_teacher = is_teacher
        
        # Get all observation keys
        all_keys = list(obs_space.keys())
        
        # Filter keys based on the regex patterns
        self.mlp_keys = []
        self.cnn_keys = []
        for k in all_keys:
            if k in ['reward', 'is_first', 'is_last', 'is_terminal']:
                continue
            if len(obs_space[k].shape) == 3 and obs_space[k].shape[-1] == 3:  # Image observations
                if is_teacher:
                    if re.search(config.full_keys.cnn_keys, k):
                        self.cnn_keys.append(k)
                else:
                    if re.search(config.keys.cnn_keys, k):
                        self.cnn_keys.append(k)
            else:  # Non-image observations
                if is_teacher:
                    if re.search(config.full_keys.mlp_keys, k):
                        self.mlp_keys.append(k)
                else:
                    if re.search(config.keys.mlp_keys, k):
                        self.mlp_keys.append(k)
        
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
        if config.encoder.act == 'silu':
            self.act = nn.SiLU()
        elif config.encoder.act == 'relu':
            self.act = nn.ReLU()
        elif config.encoder.act == 'tanh':
            self.act = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation function: {config.encoder.act}")
        
        # Calculate CNN output dimension
        if self.cnn_keys:
            # Calculate number of stages based on minres
            input_size = 64  # From config.env.atari.size
            stages = int(np.log2(input_size) - np.log2(config.encoder.minres))
            final_depth = config.encoder.cnn_depth * (2 ** (stages - 1))
            self.cnn_output_dim = final_depth * config.encoder.minres * config.encoder.minres
            
            # Store CNN key shapes
            self.cnn_key_shapes = {}
            for key in self.cnn_keys:
                self.cnn_key_shapes[key] = obs_space[key].shape
            
            # CNN encoder for image observations
            self.cnn_encoder = nn.Sequential(
                ImageEncoderResnet(
                    depth=config.encoder.cnn_depth,
                    blocks=config.encoder.cnn_blocks,
                    resize=config.encoder.resize,
                    minres=config.encoder.minres,
                    output_dim=self.cnn_output_dim
                ),
                nn.LayerNorm(self.cnn_output_dim) if config.encoder.norm == 'layer' else nn.Identity()
            )
        else:
            self.cnn_encoder = None
            self.cnn_output_dim = 0
            self.cnn_key_shapes = {}
        
        # MLP encoder for non-image observations
        if self.mlp_keys:
            layers = []
            input_dim = self.total_mlp_size
            
            # Add MLP layers
            for _ in range(config.encoder.mlp_layers):
                layers.extend([
                    layer_init(nn.Linear(input_dim, config.encoder.mlp_units)),
                    self.act,
                    nn.LayerNorm(config.encoder.mlp_units) if config.encoder.norm == 'layer' else nn.Identity()
                ])
                input_dim = config.encoder.mlp_units
            
            self.mlp_encoder = nn.Sequential(*layers)
            self.mlp_output_dim = config.encoder.mlp_units
        else:
            self.mlp_encoder = None
            self.mlp_output_dim = 0
        
        # Calculate total input dimension for latent projector
        total_input_dim = self.cnn_output_dim + self.mlp_output_dim
        
        # Project concatenated features to latent space
        self.output_proj = nn.Sequential(
            layer_init(nn.Linear(total_input_dim, config.encoder.output_dim)),
            self.act,
            nn.LayerNorm(config.encoder.output_dim) if config.encoder.norm == 'layer' else nn.Identity(),
            layer_init(nn.Linear(config.encoder.output_dim, config.encoder.output_dim)),
            self.act,
            nn.LayerNorm(config.encoder.output_dim) if config.encoder.norm == 'layer' else nn.Identity()
        )
    
    def encode_observations(self, x):
        """Encode observations into a latent representation.
        
        Args:
            x: Dictionary of observations
            
        Returns:
            Latent representation
        """
        # Process MLP inputs
        mlp_features = []
        for key in self.mlp_keys:
            if key in x:
                # Reshape to (batch_size, -1) if needed
                if len(x[key].shape) > 2:
                    mlp_features.append(x[key].reshape(x[key].shape[0], -1))
                else:
                    mlp_features.append(x[key])
            else:
                # If key is missing, use zeros
                size = self.mlp_key_sizes[key]
                mlp_features.append(torch.zeros((x[list(x.keys())[0]].shape[0], size), device=x[list(x.keys())[0]].device))
        
        if mlp_features:
            mlp_features = torch.cat(mlp_features, dim=1)
            if mlp_features.shape[1] != self.total_mlp_size:
                raise ValueError(f"MLP input size mismatch. Expected {self.total_mlp_size}, got {mlp_features.shape[1]}. Keys: {self.mlp_keys}, Sizes: {self.mlp_key_sizes}")
            mlp_features = self.mlp_encoder(mlp_features)
        else:
            mlp_features = torch.zeros((x[list(x.keys())[0]].shape[0], self.mlp_output_dim), device=x[list(x.keys())[0]].device)
        
        # Process CNN inputs
        cnn_features = []
        for key in self.cnn_keys:
            if key in x:
                cnn_features.append(x[key])
            else:
                # If key is missing, use zeros
                shape = self.cnn_key_shapes[key]
                cnn_features.append(torch.zeros((x[list(x.keys())[0]].shape[0],) + shape, device=x[list(x.keys())[0]].device))
        
        if cnn_features:
            cnn_features = torch.cat(cnn_features, dim=1)
            cnn_features = self.cnn_encoder(cnn_features)
        else:
            cnn_features = torch.zeros((x[list(x.keys())[0]].shape[0], self.cnn_output_dim), device=x[list(x.keys())[0]].device)
        
        # Combine features
        features = torch.cat([mlp_features, cnn_features], dim=1)
        
        # Project to final representation
        return self.output_proj(features)


class DualEncoder(nn.Module):
    def __init__(self, obs_space, config):
        super().__init__()
        self.teacher_encoder = Encoder(obs_space, config, True)
        self.student_encoder = Encoder(obs_space, config, False)
        
        # Store imitation flags and weights
        self.student_to_teacher_imitation = config.encoder.student_to_teacher_imitation
        self.teacher_to_student_imitation = config.encoder.teacher_to_student_imitation
        self.student_to_teacher_lambda = config.encoder.student_to_teacher_lambda
        self.teacher_to_student_lambda = config.encoder.teacher_to_student_lambda
    
    def encode_teacher_observations(self, x):
        return self.teacher_encoder.encode_observations(x)
    
    def encode_student_observations(self, x):
        """Encode observations using student encoder.
        
        Args:
            x: Dictionary of observations
            
        Returns:
            Student's latent representation
        """
        # Create a new dictionary with only student keys
        student_obs = {}
        for key in x.keys():
            # Skip special keys
            if key in ['is_first', 'is_last', 'reward']:
                continue
            # Check if key matches any student key pattern
            if any(re.search(pattern, key) for pattern in self.student_encoder.mlp_keys + self.student_encoder.cnn_keys):
                student_obs[key] = x[key]
        
        return self.student_encoder.encode_observations(student_obs)

    def compute_teacher_to_student_loss(self, teacher_latent, student_latent):
        return torch.nn.functional.mse_loss(teacher_latent, student_latent)
    
    def compute_student_to_teacher_loss(self, teacher_latent, student_latent):
        return torch.nn.functional.mse_loss(student_latent, teacher_latent)
    