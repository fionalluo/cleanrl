import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np

class BehavioralCloning:
    def __init__(self, student_policy, teacher_policy, config):
        self.student = student_policy
        self.teacher = teacher_policy
        self.config = config
        self.optimizer = optim.Adam(
            student_policy.parameters(),
            lr=config.bc.learning_rate,
            eps=1e-5
        )
        
    def train_step(self, batch):
        """Perform a single BC training step.
        
        Args:
            batch: dict containing:
                - partial_obs: dict of partial observations from student trajectories
                - full_obs: dict of full observations from student trajectories
                
        Returns:
            dict containing training metrics
        """
        # Get student's action predictions using partial observations
        _, student_log_probs, _, _ = self.student.get_action_and_value(batch['partial_obs'])
        
        # Get teacher's actions using full observations
        with torch.no_grad():
            # Filter observations to only include teacher's keys
            teacher_obs = {}
            for key in self.teacher.mlp_keys + self.teacher.cnn_keys:
                if key in batch['full_obs']:
                    # Ensure the observation has the correct shape
                    obs = batch['full_obs'][key]
                    if len(obs.shape) == 2:  # [batch_size, flattened_size]
                        teacher_obs[key] = obs
                    else:  # [batch_size, *original_shape]
                        # For MLP observations, flatten all dimensions except batch
                        if key in self.teacher.mlp_keys:
                            teacher_obs[key] = obs.reshape(obs.shape[0], -1)
                        # For CNN observations, keep the image shape
                        else:
                            teacher_obs[key] = obs
            
            teacher_actions, _, _, _ = self.teacher.get_action_and_value(teacher_obs)
        
        # Compute loss
        if self.student.is_discrete:
            # For discrete actions, use cross entropy loss
            loss = -student_log_probs.mean()
        else:
            # For continuous actions, use MSE loss
            student_actions = self.student.actor_mean(self.student.encode_observations(batch['partial_obs']))
            loss = ((student_actions - teacher_actions) ** 2).mean()
        
        # Update policy
        self.optimizer.zero_grad()
        loss.backward()
        if self.config.bc.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.student.parameters(), self.config.bc.max_grad_norm)
        self.optimizer.step()
        
        # Convert tensors to scalars for metrics
        metrics = {
            'bc_loss': loss.item()
        }
        
        if not self.student.is_discrete:
            metrics['action_diff'] = (student_actions - teacher_actions).abs().mean().item()
        
        return metrics
    
    def train(self, envs, num_steps, batch_size, writer=None, global_step=0):
        """Train the student policy using BC for a number of steps.
        
        Args:
            envs: Vectorized environment
            num_steps: Number of training steps
            batch_size: Batch size for training
            writer: TensorBoard writer for logging
            global_step: Global step counter for logging
            
        Returns:
            dict containing training metrics
        """
        metrics = {
            'bc_loss': [],
            'action_diff': [] if not self.student.is_discrete else None
        }
        
        # Get union of teacher and student keys
        all_keys = set(self.teacher.mlp_keys + self.teacher.cnn_keys + self.student.mlp_keys + self.student.cnn_keys)
        
        # Initialize observation storage
        obs = {}
        for key in all_keys:
            if len(envs.obs_space[key].shape) == 3 and envs.obs_space[key].shape[-1] == 3:  # Image observations
                obs[key] = torch.zeros((num_steps, self.config.num_envs) + envs.obs_space[key].shape).to(self.student.device)
            else:  # Non-image observations
                size = np.prod(envs.obs_space[key].shape)
                obs[key] = torch.zeros((num_steps, self.config.num_envs, size)).to(self.student.device)
        
        # Initialize actions with zeros and reset flags
        action_shape = envs.act_space['action'].shape
        acts = {
            'action': np.zeros((self.config.num_envs,) + action_shape, dtype=np.float32),
            'reset': np.ones(self.config.num_envs, dtype=bool)
        }
        
        # Get initial observations
        obs_dict = envs.step(acts)
        next_obs = {}
        for key in all_keys:
            next_obs[key] = torch.Tensor(obs_dict[key].astype(np.float32)).to(self.student.device)
        next_done = torch.Tensor(obs_dict['is_last'].astype(np.float32)).to(self.student.device)
        
        # Collect trajectories
        for step in range(num_steps):
            # Store observations
            for key in all_keys:
                obs[key][step] = next_obs[key]
            
            # Get action from student policy
            with torch.no_grad():
                action, _, _, _ = self.student.get_action_and_value(next_obs)
            
            # Step environment
            action_np = action.cpu().numpy()
            if self.student.is_discrete:
                action_np = action_np.reshape(self.config.num_envs, -1)
            
            acts = {
                'action': action_np,
                'reset': next_done.cpu().numpy()
            }
            
            obs_dict = envs.step(acts)
            
            # Process observations
            for key in all_keys:
                next_obs[key] = torch.Tensor(obs_dict[key].astype(np.float32)).to(self.student.device)
            next_done = torch.Tensor(obs_dict['is_last'].astype(np.float32)).to(self.student.device)
        
        # Calculate total number of samples
        total_samples = num_steps * self.config.num_envs
        
        # Train on collected trajectories
        for start_idx in range(0, total_samples, batch_size):
            end_idx = min(start_idx + batch_size, total_samples)
            current_batch_size = end_idx - start_idx
            
            # Create batch
            batch = {
                'partial_obs': {},
                'full_obs': {}
            }
            
            # Filter observations for student and teacher
            for key in all_keys:
                # Reshape observations to [total_samples, *shape]
                flat_obs = obs[key].reshape(total_samples, *obs[key].shape[2:])
                # Get current batch
                batch_obs = flat_obs[start_idx:end_idx]
                
                if key in self.student.mlp_keys + self.student.cnn_keys:
                    batch['partial_obs'][key] = batch_obs
                if key in self.teacher.mlp_keys + self.teacher.cnn_keys:
                    batch['full_obs'][key] = batch_obs
            
            # Perform training step
            step_metrics = self.train_step(batch)
            
            # Update metrics
            for key, value in step_metrics.items():
                if value is not None:
                    metrics[key].append(value)
            
            # Log to TensorBoard
            if writer is not None:
                writer.add_scalar('bc/loss', step_metrics['bc_loss'], global_step + start_idx)
                
                if not self.student.is_discrete:
                    writer.add_scalar('bc/action_diff', step_metrics['action_diff'], global_step + start_idx)
        
        # Compute average metrics
        avg_metrics = {
            key: sum(values) / len(values) if values else None
            for key, values in metrics.items()
        }
        
        return avg_metrics
