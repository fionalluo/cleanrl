import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from ..shared.agent import BaseAgent

class StudentPolicy(BaseAgent):
    def __init__(self, envs, config):
        super().__init__(envs, config)
        # Student uses partial observations
        self.obs_keys = config.keys
        
    def train_bc(self, replay_buffer, batch_size, optimizer):
        """Train the student policy using behavioral cloning.
        
        Args:
            replay_buffer: ReplayBuffer containing teacher demonstrations
            batch_size: Batch size for training
            optimizer: Optimizer for the student policy
            
        Returns:
            dict containing training metrics
        """
        # Sample batch from replay buffer
        batch = replay_buffer.sample(batch_size)
        
        # Get student's action predictions
        _, student_log_probs, _, _ = self.get_action_and_value(batch['partial_obs'])
        
        # Get teacher's actions
        teacher_actions = batch['action']
        
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
                - obs: dict of partial observations
                - action: action taken
                - reward: reward received
                - next_obs: dict of next partial observations
                - done: whether episode ended
        """
        transitions = []
        
        # Initialize observation storage
        obs = {}
        for key in self.mlp_keys + self.cnn_keys:
            if key in self.mlp_keys:
                obs[key] = torch.zeros((num_steps, envs.num_envs, self.mlp_key_sizes[key])).to(self.device)
            else:  # CNN keys
                obs[key] = torch.zeros((num_steps, envs.num_envs) + envs.obs_space[key].shape).to(self.device)
        
        # Initialize action storage
        if self.is_discrete:
            action_shape = (num_steps, envs.num_envs, envs.act_space['action'].shape[0])
        else:
            action_shape = (num_steps, envs.num_envs) + envs.act_space['action'].shape
        actions = torch.zeros(action_shape).to(self.device)
        
        # Initialize reward and done storage
        rewards = torch.zeros((num_steps, envs.num_envs)).to(self.device)
        dones = torch.zeros((num_steps, envs.num_envs)).to(self.device)
        
        # Get initial observations
        next_obs = {}
        for key in self.mlp_keys + self.cnn_keys:
            next_obs[key] = torch.Tensor(envs.reset()[key]).to(self.device)
        next_done = torch.zeros(envs.num_envs).to(self.device)
        
        # Collect transitions
        for step in range(num_steps):
            # Store observations
            for key in self.mlp_keys + self.cnn_keys:
                obs[key][step] = next_obs[key]
            dones[step] = next_done
            
            # Get action from policy
            with torch.no_grad():
                action, _, _, _ = self.get_action_and_value(next_obs)
                actions[step] = action
            
            # Step environment
            action_np = action.cpu().numpy()
            if self.is_discrete:
                action_np = action_np.reshape(envs.num_envs, -1)
            
            acts = {
                'action': action_np,
                'reset': next_done.cpu().numpy()
            }
            
            obs_dict = envs.step(acts)
            
            # Process observations
            for key in self.mlp_keys + self.cnn_keys:
                next_obs[key] = torch.Tensor(obs_dict[key]).to(self.device)
            next_done = torch.Tensor(obs_dict['is_last']).to(self.device)
            rewards[step] = torch.tensor(obs_dict['reward']).to(self.device)
            
            # Store transition
            for env_idx in range(envs.num_envs):
                transition = {
                    'obs': {key: obs[key][step, env_idx].cpu().numpy() for key in self.mlp_keys + self.cnn_keys},
                    'action': actions[step, env_idx].cpu().numpy(),
                    'reward': rewards[step, env_idx].cpu().numpy(),
                    'next_obs': {key: next_obs[key][env_idx].cpu().numpy() for key in self.mlp_keys + self.cnn_keys},
                    'done': next_done[env_idx].cpu().numpy()
                }
                transitions.append(transition)
        
        return transitions
