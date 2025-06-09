import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from collections import deque
import random

# grab our noisy linear layer from the basic version
from rdqn_basic import NoisyLinear

class RDQNLSTMNet(nn.Module):
    def __init__(self,
                 state_dim: int = 16,  # 4x4 board
                 action_dim: int = 4,   # up, down, left, right
                 n_atoms: int = 51,     # number of atoms for categorical dqn
                 v_min: float = -10.0,  # minimum value for categorical support
                 v_max: float = 10.0,   # maximum value for categorical support
                 hidden_dim: int = 256,  # hidden dimension for lstm
                 ):
        super().__init__()
        
        # store these for later use
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.hidden_dim = hidden_dim
        
        # our trusty mlp backbone
        self.mlp = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # add lstm layer for temporal dependencies
        self.lstm = nn.LSTM(64, hidden_dim, batch_first=True)
        
        # value stream with temporal context
        self.value_net = nn.Sequential(
            NoisyLinear(hidden_dim, hidden_dim),
            nn.ReLU(),
            NoisyLinear(hidden_dim, n_atoms)
        )
        
        # advantage stream with temporal context
        self.advantage_net = nn.Sequential(
            NoisyLinear(hidden_dim, hidden_dim),
            nn.ReLU(),
            NoisyLinear(hidden_dim, action_dim * n_atoms)
        )
        
        # set up categorical distribution support
        self.register_buffer('supports', torch.linspace(v_min, v_max, n_atoms))
        self.delta_z = (v_max - v_min) / (n_atoms - 1)

    def reset_noise(self):
        # reset that noise yo
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()

    def forward(self, 
                state: torch.Tensor,
                hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size = state.size(0)
        seq_length = state.size(1) if len(state.size()) > 2 else 1
        
        # flatten and process through mlp
        x = state.view(-1, self.state_dim)
        x = self.mlp(x)
        x = x.view(batch_size, seq_length, -1)
        
        # run through lstm
        lstm_out, new_hidden = self.lstm(x, hidden)
        
        # reshape for value and advantage streams
        x = lstm_out.reshape(-1, self.hidden_dim)
        
        # get value and advantage distributions
        value = self.value_net(x)
        advantage = self.advantage_net(x)
        
        # reshape advantage
        advantage = advantage.view(-1, self.action_dim, self.n_atoms)
        
        # combine using dueling architecture
        value = value.view(-1, 1, self.n_atoms)
        q_dist = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        # softmax to get probabilities
        q_dist = F.softmax(q_dist, dim=-1)
        
        # reshape back to sequence format if needed
        q_dist = q_dist.view(batch_size, seq_length, self.action_dim, self.n_atoms)
        
        return q_dist, new_hidden

class LSTMAgent:
    def __init__(self,
                 state_dim: int = 16,
                 action_dim: int = 4,
                 gamma: float = 0.99,
                 learning_rate: float = 0.0001,
                 buffer_size: int = 100000,
                 batch_size: int = 32,
                 target_update_freq: int = 1000,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"
                 ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = device
        
        # networks and optimizer
        self.online_net = RDQNLSTMNet(state_dim, action_dim).to(device)
        self.target_net = RDQNLSTMNet(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=learning_rate)
        
        # replay buffer (still basic for now)
        self.buffer = deque(maxlen=buffer_size)
        
        # training variables
        self.target_update_freq = target_update_freq
        self.steps = 0
        
        # lstm state
        self.hidden = None
        
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
        
    def update(self, state, action, reward, next_state, done):
        # store transition
        self.buffer.append((state, action, reward, next_state, done))
        
        # only train if we have enough samples
        if len(self.buffer) < self.batch_size:
            return 0.0
            
        # sample a batch
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # convert to tensors and add sequence dimension
        states = torch.FloatTensor(states).unsqueeze(1).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # get current q-distribution
        q_dist, _ = self.online_net(states)
        q_dist = q_dist.squeeze(1)  # remove sequence dimension
        actions = actions.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, self.online_net.n_atoms)
        q_dist = q_dist.gather(1, actions).squeeze(1)
        
        # get next q-distribution from target network
        with torch.no_grad():
            next_q_dist, _ = self.target_net(next_states)
            next_q_dist = next_q_dist.squeeze(1)  # remove sequence dimension
            next_q_values = (next_q_dist * self.target_net.supports).sum(-1)
            next_actions = next_q_values.argmax(-1)
            next_actions = next_actions.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, self.online_net.n_atoms)
            next_q_dist = next_q_dist.gather(1, next_actions).squeeze(1)
            
            # compute target distribution
            rewards = rewards.unsqueeze(-1)
            dones = dones.unsqueeze(-1)
            supports = self.target_net.supports.unsqueeze(0)
            target_supports = rewards + (1 - dones) * self.gamma * supports
            target_supports = target_supports.clamp(self.target_net.v_min, self.target_net.v_max)
            
            # project onto categorical support
            target_dist = self._categorical_projection(target_supports, next_q_dist)
        
        # compute kl loss
        loss = -(target_dist * q_dist.clamp(min=1e-5).log()).sum(-1).mean()
        
        # optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # update target network if needed
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
            
        return loss.item()
        
    def _categorical_projection(self, supports, probs):
        # project values onto categorical support (same as basic version)
        vmin, vmax = self.online_net.v_min, self.online_net.v_max
        dz = self.online_net.delta_z
        
        tz = supports.clamp(vmin, vmax)
        b = (tz - vmin) / dz
        l = b.floor().long()
        u = b.ceil().long()
        
        # distribute probability
        proj_dist = torch.zeros_like(probs)
        offset = torch.linspace(0, (probs.size(0) - 1) * probs.size(1), probs.size(0)).unsqueeze(1).expand(probs.size(0), probs.size(1)).long().to(self.device)
        
        proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (probs * (u.float() - b)).view(-1))
        proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (probs * (b - l.float())).view(-1))
        
        return proj_dist
        
    def reset_hidden(self):
        # reset lstm state when episode ends
        self.hidden = None 