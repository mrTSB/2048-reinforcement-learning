import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
from collections import deque
import random

# grab our network architecture from the lstm version
from rdqn_lstm import RDQNLSTMNet

class PrioritizedReplayBuffer:
    def __init__(self, 
                 capacity: int,
                 sequence_length: int = 20,
                 alpha: float = 0.6,
                 beta: float = 0.4,
                 beta_increment: float = 0.001
                 ):
        # basic setup
        self.capacity = capacity
        self.sequence_length = sequence_length
        self.alpha = alpha  # how much to prioritize
        self.beta = beta    # importance sampling weight
        self.beta_increment = beta_increment  # annealing rate
        
        # storage
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0
        
    def push(self, sequence: List[Tuple]):
        """toss a sequence into the buffer"""
        # new sequences get max priority (optimistic!)
        priority = self.max_priority ** self.alpha
        
        # add to buffer
        if len(self.buffer) < self.capacity:
            self.buffer.append(sequence)
        else:
            self.buffer[self.position] = sequence
            
        self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size: int) -> Tuple:
        """grab some sequences based on their priorities"""
        if len(self.buffer) == 0:
            return None
        
        # sample based on priorities
        probs = self.priorities[:len(self.buffer)] / self.priorities[:len(self.buffer)].sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        
        # compute importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()
        
        # slowly increase beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # grab the sequences
        sequences = [self.buffer[idx] for idx in indices]
        
        return sequences, indices, weights
        
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """update priorities based on td errors"""
        priorities = np.abs(priorities)  # we care about magnitude of error
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority ** self.alpha
            self.max_priority = max(self.max_priority, priority)

class SequenceAgent:
    def __init__(self,
                 state_dim: int = 16,
                 action_dim: int = 4,
                 gamma: float = 0.99,
                 sequence_length: int = 20,
                 learning_rate: float = 0.0001,
                 target_update_freq: int = 1000,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"
                 ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.sequence_length = sequence_length
        self.device = device
        
        # networks and optimizer
        self.online_net = RDQNLSTMNet(state_dim, action_dim).to(device)
        self.target_net = RDQNLSTMNet(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=learning_rate)
        
        # now we're using prioritized replay!
        self.replay_buffer = PrioritizedReplayBuffer(100000, sequence_length)
        
        # training variables
        self.target_update_freq = target_update_freq
        self.steps = 0
        
        # sequence collection
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
        # sample sequences from replay buffer
        result = self.replay_buffer.sample(batch_size)
        if result is None:
            return 0.0
            
        sequences, indices, weights = result
        
        # prepare sequences for processing
        states, actions, rewards, next_states, dones = zip(*[zip(*seq) for seq in sequences])
        
        # convert to torch tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        # get current q-distribution
        q_dist, _ = self.online_net(states)
        
        # get next q-distribution from target network
        with torch.no_grad():
            next_q_dist, _ = self.target_net(next_states)
            next_q_values = (next_q_dist * self.target_net.supports).sum(-1)
            next_actions = next_q_values.argmax(-1)
            
        # compute n-step returns
        returns = self._compute_returns(rewards, next_q_values, dones)
        
        # compute loss
        loss = self._compute_categorical_loss(q_dist, actions, returns, weights)
        
        # update priorities in replay buffer
        td_errors = loss.detach().cpu().numpy()
        self.replay_buffer.update_priorities(indices, td_errors)
        
        # optimize
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()
        
        # update target network if needed
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
            
        return loss.mean().item()
        
    def _compute_returns(self,
                        rewards: torch.Tensor,
                        next_q_values: torch.Tensor,
                        dones: torch.Tensor
                        ) -> torch.Tensor:
        returns = torch.zeros_like(rewards)
        future_return = next_q_values * (1 - dones)
        
        # compute returns for each timestep
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
        # get the q-distribution for the actions we took
        actions = actions.unsqueeze(-1).expand(-1, -1, self.online_net.n_atoms)
        q_dist = q_dist.gather(2, actions).squeeze(2)
        
        # project returns onto categorical support
        delta_z = self.online_net.delta_z
        v_min = self.online_net.v_min
        v_max = self.online_net.v_max
        
        returns = returns.clamp(v_min, v_max)
        b = (returns - v_min) / delta_z
        l = b.floor().long()
        u = b.ceil().long()
        
        # distribute probability of returns
        projected_dist = torch.zeros_like(q_dist)
        offset = torch.linspace(0, (q_dist.size(0) - 1) * q_dist.size(1), q_dist.size(0)).unsqueeze(1).expand(q_dist.size(0), q_dist.size(1)).long()
        projected_dist.view(-1).index_add_(0, (l + offset).view(-1), (q_dist * (u.float() - b)).view(-1))
        projected_dist.view(-1).index_add_(0, (u + offset).view(-1), (q_dist * (b - l.float())).view(-1))
        
        # compute cross-entropy loss
        loss = -(projected_dist * q_dist.log()).sum(-1)
        
        # apply importance sampling weights
        loss = loss * weights.unsqueeze(1)
        
        return loss 