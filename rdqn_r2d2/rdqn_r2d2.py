import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
from collections import deque
import random

class NoisyLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size: int) -> torch.Tensor:
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            return F.linear(x,
                          self.weight_mu + self.weight_sigma * self.weight_epsilon,
                          self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(x, self.weight_mu, self.bias_mu)

class RDQNR2D2Network(nn.Module):
    def __init__(self,
                 state_dim: int = 16,  # 4x4 board
                 action_dim: int = 4,   # Up, Down, Left, Right
                 n_atoms: int = 51,     # Number of atoms for categorical DQN
                 v_min: float = -10.0,  # Minimum value for categorical support
                 v_max: float = 10.0,   # Maximum value for categorical support
                 hidden_dim: int = 256,  # Hidden dimension for LSTM
                 ):
        super(RDQNR2D2Network, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.hidden_dim = hidden_dim
        
        # Shared MLP torso
        self.mlp = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # LSTM layer
        self.lstm = nn.LSTM(64, hidden_dim, batch_first=True)
        
        # Noisy value stream
        self.value_net = nn.Sequential(
            NoisyLinear(hidden_dim, hidden_dim),
            nn.ReLU(),
            NoisyLinear(hidden_dim, n_atoms)
        )
        
        # Noisy advantage stream
        self.advantage_net = nn.Sequential(
            NoisyLinear(hidden_dim, hidden_dim),
            nn.ReLU(),
            NoisyLinear(hidden_dim, action_dim * n_atoms)
        )
        
        self.register_buffer('supports', torch.linspace(v_min, v_max, n_atoms))
        self.delta_z = (v_max - v_min) / (n_atoms - 1)

    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()

    def forward(self, 
                state: torch.Tensor,
                hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size = state.size(0)
        seq_length = state.size(1) if len(state.size()) > 2 else 1
        
        # Reshape if necessary and pass through MLP
        x = state.view(-1, self.state_dim)
        x = self.mlp(x)
        x = x.view(batch_size, seq_length, -1)
        
        # LSTM processing
        lstm_out, new_hidden = self.lstm(x, hidden)
        
        # Reshape for value and advantage streams
        x = lstm_out.reshape(-1, self.hidden_dim)
        
        # Get categorical value distribution
        value = self.value_net(x)
        advantage = self.advantage_net(x)
        
        # Reshape advantage
        advantage = advantage.view(-1, self.action_dim, self.n_atoms)
        
        # Combine value and advantage using dueling architecture
        value = value.view(-1, 1, self.n_atoms)
        q_dist = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        # Apply softmax to get probabilities
        q_dist = F.softmax(q_dist, dim=-1)
        
        # Reshape back to sequence format if necessary
        q_dist = q_dist.view(batch_size, seq_length, self.action_dim, self.n_atoms)
        
        return q_dist, new_hidden

class PrioritizedReplayBuffer:
    def __init__(self, 
                 capacity: int,
                 sequence_length: int = 20,
                 alpha: float = 0.6,
                 beta: float = 0.4,
                 beta_increment: float = 0.001
                 ):
        self.capacity = capacity
        self.sequence_length = sequence_length
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0
        
    def push(self, sequence: List[Tuple]):
        """Push a sequence of transitions into the buffer"""
        priority = self.max_priority ** self.alpha
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(sequence)
        else:
            self.buffer[self.position] = sequence
            
        self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size: int) -> Tuple:
        if len(self.buffer) == 0:
            return None
        
        # Sample indices based on priorities
        probs = self.priorities[:len(self.buffer)] / self.priorities[:len(self.buffer)].sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()
        
        # Increment beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Get sequences
        sequences = [self.buffer[idx] for idx in indices]
        
        return sequences, indices, weights
        
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        priorities = np.abs(priorities)
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority ** self.alpha
            self.max_priority = max(self.max_priority, priority)

class RDQNR2D2Agent:
    def __init__(self,
                 state_dim: int = 16,
                 action_dim: int = 4,
                 gamma: float = 0.99,
                 n_step: int = 5,
                 sequence_length: int = 20,
                 burn_in_length: int = 5,
                 learning_rate: float = 0.0001,
                 target_update_freq: int = 1000,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"
                 ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.n_step = n_step
        self.sequence_length = sequence_length
        self.burn_in_length = burn_in_length
        self.device = device
        
        # Networks
        self.online_net = RDQNR2D2Network(state_dim, action_dim).to(device)
        self.target_net = RDQNR2D2Network(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.replay_buffer = PrioritizedReplayBuffer(100000, sequence_length)
        
        # Training variables
        self.target_update_freq = target_update_freq
        self.steps = 0
        
        # Sequence buffer for collecting experiences
        self.current_sequence = []
        
    def select_action(self, 
                     state: np.ndarray,
                     hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                     epsilon: float = 0.0
                     ) -> Tuple[int, Tuple[torch.Tensor, torch.Tensor]]:
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1), hidden
            
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device)
            q_dist, new_hidden = self.online_net(state, hidden)
            q_values = (q_dist * self.online_net.supports).sum(-1)
            action = q_values.argmax(-1).item()
            
        return action, new_hidden
        
    def update(self, batch_size: int) -> float:
        # Sample sequences from replay buffer
        result = self.replay_buffer.sample(batch_size)
        if result is None:
            return 0.0
            
        sequences, indices, weights = result
        
        # Prepare sequences for processing
        states, actions, rewards, next_states, dones = zip(*[zip(*seq) for seq in sequences])
        
        # Convert to torch tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        # Get current Q-distribution
        q_dist, _ = self.online_net(states)
        
        # Get next Q-distribution from target network
        with torch.no_grad():
            next_q_dist, _ = self.target_net(next_states)
            next_q_values = (next_q_dist * self.target_net.supports).sum(-1)
            next_actions = next_q_values.argmax(-1)
            
        # Compute n-step returns
        returns = self._compute_n_step_returns(rewards, next_q_values, dones)
        
        # Compute loss
        loss = self._compute_categorical_loss(q_dist, actions, returns, weights)
        
        # Update priorities in replay buffer
        td_errors = loss.detach().cpu().numpy()
        self.replay_buffer.update_priorities(indices, td_errors)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()
        
        # Update target network if needed
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
            
        return loss.mean().item()
        
    def _compute_n_step_returns(self,
                              rewards: torch.Tensor,
                              next_q_values: torch.Tensor,
                              dones: torch.Tensor
                              ) -> torch.Tensor:
        returns = torch.zeros_like(rewards)
        future_return = next_q_values * (1 - dones)
        
        for t in reversed(range(rewards.size(1))):
            future_return = rewards[:, t] + self.gamma * future_return * (1 - dones[:, t])
            returns[:, t] = future_return
            
        return returns
        
    def _compute_categorical_loss(self,
                                q_dist: torch.Tensor,
                                actions: torch.Tensor,
                                returns: torch.Tensor,
                                weights: torch.Tensor
                                ) -> torch.Tensor:
        # Get the Q-distribution for the taken actions
        actions = actions.unsqueeze(-1).expand(-1, -1, self.online_net.n_atoms)
        q_dist = q_dist.gather(2, actions).squeeze(2)
        
        # Project returns onto categorical support
        delta_z = self.online_net.delta_z
        v_min = self.online_net.v_min
        v_max = self.online_net.v_max
        
        returns = returns.clamp(v_min, v_max)
        b = (returns - v_min) / delta_z
        l = b.floor().long()
        u = b.ceil().long()
        
        # Distribute probability of returns
        projected_dist = torch.zeros_like(q_dist)
        offset = torch.linspace(0, (q_dist.size(0) - 1) * q_dist.size(1), q_dist.size(0)).unsqueeze(1).expand(q_dist.size(0), q_dist.size(1)).long()
        projected_dist.view(-1).index_add_(0, (l + offset).view(-1), (q_dist * (u.float() - b)).view(-1))
        projected_dist.view(-1).index_add_(0, (u + offset).view(-1), (q_dist * (b - l.float())).view(-1))
        
        # Compute cross-entropy loss
        loss = -(projected_dist * q_dist.log()).sum(-1)
        
        # Apply importance sampling weights
        loss = loss * weights.unsqueeze(1)
        
        return loss 