import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

class BehavioralCloning:
    def __init__(self, student_policy, config):
        self.student = student_policy
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
                - partial_obs: dict of partial observations
                - action: teacher's actions
                
        Returns:
            dict containing training metrics
        """
        # Get student's action predictions
        _, student_log_probs, _, _ = self.student.get_action_and_value(batch['partial_obs'])
        
        # Get teacher's actions
        teacher_actions = batch['action']
        
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
        
        return {
            'bc_loss': loss.item(),
            'student_actions': student_actions.detach() if not self.student.is_discrete else None,
            'teacher_actions': teacher_actions.detach()
        }
    
    def train(self, replay_buffer, num_steps, batch_size, writer=None, global_step=0):
        """Train the student policy using BC for a number of steps.
        
        Args:
            replay_buffer: ReplayBuffer containing teacher demonstrations
            num_steps: Number of training steps
            batch_size: Batch size for training
            writer: TensorBoard writer for logging
            global_step: Global step counter for logging
            
        Returns:
            dict containing training metrics
        """
        metrics = {
            'bc_loss': [],
            'student_actions': [],
            'teacher_actions': []
        }
        
        for step in range(num_steps):
            # Sample batch from replay buffer
            batch = replay_buffer.sample(batch_size)
            
            # Perform training step
            step_metrics = self.train_step(batch)
            
            # Update metrics
            for key, value in step_metrics.items():
                if value is not None:
                    metrics[key].append(value)
            
            # Log to TensorBoard
            if writer is not None:
                writer.add_scalar('bc/loss', step_metrics['bc_loss'], global_step + step)
                
                if not self.student.is_discrete:
                    writer.add_scalar('bc/action_diff', 
                                    (step_metrics['student_actions'] - step_metrics['teacher_actions']).abs().mean().item(),
                                    global_step + step)
        
        # Compute average metrics
        avg_metrics = {
            key: sum(values) / len(values) if values else None
            for key, values in metrics.items()
        }
        
        return avg_metrics
