import gymnasium as gym
import gymnasium_2048
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple, Counter
import random
from tqdm.auto import tqdm

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make("gymnasium_2048/TwentyFortyEight-v0")

# --- Hyperparameters ---
BUFFER_SIZE     = int(1e6)    # replay buffer size
BATCH_SIZE      = 512         # minibatch size
GAMMA           = 0.99        # discount factor
LR              = 1e-4        # learning rate
TAU             = 1e-3        # soft update rate
UPDATE_EVERY    = 4           # how often to learn
EPS_START       = 1.0         # starting epsilon
EPS_END         = 0.05        # final epsilon
EPS_DECAY       = 1e-5        # epsilon decay per step
NUM_EPISODES    = 5000        # max training episodes
MAX_STEPS       = 10000       # max steps per episode
MONO_COEF       = 0.01        # monotonicity bonus weight
ROLLING_WINDOW  = 100         # window for avg max tile
TILE_THRESHOLD  = 2048        # target tile threshold
N_QUANT         = 51          # number of quantiles

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

# --- Helper Functions ---
def decode_board(obs):
    if obs.ndim == 3:
        idxs = np.argmax(obs, axis=-1)
        mask = (obs.sum(axis=-1) == 1)
        return (2 ** idxs) * mask
    return obs


def preprocess(obs):
    board = decode_board(obs).astype(int)
    idxs = np.zeros_like(board, dtype=int)
    nonzero = board > 0
    idxs[nonzero] = np.log2(board[nonzero]).astype(int)
    channels = [(idxs == i).astype(np.float32) for i in range(16)]
    x = np.stack(channels, axis=0)
    return torch.from_numpy(x).unsqueeze(0).to(device)


def monotonicity_bonus(board):
    bonus = 0
    for i in range(4):
        for j in range(3):
            bonus += (board[i, j] <= board[i, j+1]) + (board[j, i] <= board[j+1, i])
    return bonus / 24.0

# --- Quantile Network ---
class QuantileNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(16, 128, kernel_size=2), nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=2), nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(256*2*2, 512), nn.ReLU()
        )
        self.quantiles = nn.Linear(512, 4 * N_QUANT)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        q = self.quantiles(x)
        return q.view(-1, 4, N_QUANT)

# --- Replay Buffer ---
class ReplayBuffer:
    def __init__(self, size):
        self.memory = deque(maxlen=size)

    def add(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.memory)

# --- QR-DQN Agent ---
class QRDQNAgent:
    def __init__(self):
        self.local = QuantileNet().to(device)
        self.target = QuantileNet().to(device)
        self.optimizer = optim.Adam(self.local.parameters(), lr=LR)
        self.memory = ReplayBuffer(BUFFER_SIZE)
        self.epsilon = EPS_START
        self.step_count = 0

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(4)
        with torch.no_grad():
            quantiles = self.local(state)
            q_vals = quantiles.mean(dim=2)
            return int(q_vals.argmax(dim=1).item())

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.step_count += 1
        self.epsilon = max(EPS_END, self.epsilon - EPS_DECAY)
        if self.step_count % UPDATE_EVERY == 0 and len(self.memory) >= BATCH_SIZE:
            self.learn()

    def learn(self):
        trans = self.memory.sample(BATCH_SIZE)
        states      = torch.cat(trans.state)
        actions     = torch.tensor(trans.action, device=device).long()
        rewards     = torch.tensor(trans.reward, device=device).float().unsqueeze(1)
        next_states = torch.cat(trans.next_state)
        dones       = torch.tensor(trans.done, device=device).float().unsqueeze(1)

        q_curr = self.local(states)
        idx    = actions.unsqueeze(-1).unsqueeze(-1).expand(-1,1,N_QUANT)
        q_curr = q_curr.gather(1, idx).squeeze(1)

        with torch.no_grad():
            q_next = self.target(next_states)
            next_mean = q_next.mean(dim=2)
            next_actions = next_mean.argmax(dim=1)
            idx2 = next_actions.unsqueeze(-1).unsqueeze(-1).expand(-1,1,N_QUANT)
            q_next = q_next.gather(1, idx2).squeeze(1)
            target_q = rewards + (1-dones)*GAMMA * q_next

        diff  = target_q.unsqueeze(1) - q_curr.unsqueeze(2)
        huber = torch.where(diff.abs() <= 1, 0.5 * diff.pow(2), diff.abs() - 0.5)
        tau   = (torch.arange(N_QUANT, device=device).float() + 0.5) / N_QUANT
        weight = torch.abs(tau.unsqueeze(0).unsqueeze(2) - (diff < 0).float())
        loss = (weight * huber).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        for t_param, l_param in zip(self.target.parameters(), self.local.parameters()):
            t_param.data.copy_(TAU * l_param.data + (1 - TAU) * t_param.data)

# --- Training Loop ---
def train():
    agent = QRDQNAgent()
    max_tiles = []
    pbar = tqdm(range(1, NUM_EPISODES+1), desc="Training")
    for ep in pbar:
        obs, _ = env.reset()
        state = preprocess(obs)
        board_max = 0
        for step in range(1, MAX_STEPS+1):
            action = agent.act(state)
            next_obs, r, term, trunc, _ = env.step(action)
            done = term or trunc
            shaped_r = r + MONO_COEF * monotonicity_bonus(decode_board(next_obs))
            next_state = preprocess(next_obs)
            board_max = max(board_max, int(decode_board(next_obs).max()))
            agent.step(state, action, shaped_r, next_state, done)
            state = next_state
            if done:
                break
        max_tiles.append(board_max)
        avg_tile = np.mean(max_tiles[-ROLLING_WINDOW:])
        pbar.set_postfix({
            "Episode": ep,
            "Step": step,
            "MaxTile": board_max,
            "AvgMaxTile": f"{avg_tile:.1f}",
            "Epsilon": f"{agent.epsilon:.3f}",
            "Buffer": len(agent.memory)
        })
        if len(max_tiles) >= ROLLING_WINDOW and avg_tile >= TILE_THRESHOLD:
            pbar.write(f"Reached 2048 at episode {ep} (avg max tile {avg_tile:.1f}).")
            break
    return agent

# --- Evaluation ---
def evaluate(agent, episodes=1000):
    tiles = Counter()
    eval_pbar = tqdm(range(1, episodes+1), desc="Evaluating")
    for i in eval_pbar:
        obs, _ = env.reset()
        state = preprocess(obs)
        while True:
            action = agent.act(state)
            obs, _, term, trunc, _ = env.step(action)
            state = preprocess(obs)
            if term or trunc:
                break
        tiles[int(decode_board(obs).max())] += 1
        eval_pbar.set_postfix({"Episode": i})

    print("Evaluation complete:")
    for tile, count in sorted(tiles.items()):
        print(f"  {tile}: {count}")

if __name__ == '__main__':
    trained_agent = train()
    evaluate(trained_agent)
    