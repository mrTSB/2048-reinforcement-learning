import gymnasium as gym
import gymnasium_2048
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from collections import defaultdict, Counter, deque
from tqdm.auto import tqdm

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make("gymnasium_2048/TwentyFortyEight-v0")

# --- Hyperparameters ---
GAMMA = 0.99                   # Discount factor
LAMBDA = 0.97                  # GAE lambda
CLIP_EPS = 0.2                 # PPO clip epsilon
LR = 3e-4                      # Learning rate
PPO_EPOCHS = 6                 # PPO epochs per update
ROLLOUT_STEPS = 4096           # Steps per rollout
MAX_ITERATIONS = 10000         # Max training iterations
ROLLING_AVG_WINDOW = 100       # Window size for rolling average
ROLLING_AVG_THRESHOLD = 20000  # Threshold to stop early
ENTROPY_COEF = 0.01            # Entropy coefficient for loss
STAT_UPDATE_INTERVAL = 1000    # Update stats every 1000 iterations

# Phase thresholds
PHASE_THRESHOLDS = [128, 512]

# Evaluation settings
NUM_EVAL_EPISODES = 1000
MAX_STEPS_PER_EPISODE = 10000

ACTION_MAP = {0: "Up", 1: "Right", 2: "Down", 3: "Left"}

# --- Helper Functions ---
def decode_board(obs):
    """Convert observation (one-hot or raw) to a 4x4 board of tile values."""
    if obs.ndim == 3:
        idxs = np.argmax(obs, axis=-1)
        mask = (obs.sum(axis=-1) == 1)
        return (2 ** idxs) * mask
    return obs


def preprocess(obs):
    """Convert obs to a log2-indexed one-hot tensor of shape [1,16,4,4]."""
    board = decode_board(obs).astype(int)
    idxs = np.zeros_like(board, dtype=int)
    nonzero = board > 0
    idxs[nonzero] = np.log2(board[nonzero]).astype(int)
    channels = [(idxs == i).astype(np.float32) for i in range(16)]
    x = np.stack(channels, axis=0)
    return torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)


def monotonicity_bonus(board):
    """Simple reward: count adjacent monotonic pairs."""
    bonus = 0
    for i in range(4):
        for j in range(3):
            bonus += (board[i, j] <= board[i, j+1]) + (board[j, i] <= board[j+1, i])
    return bonus / 24.0

# --- Actor-Critic with CNN ---
class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=2), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=2), nn.ReLU()
        )
        # conv dims: 4->3->2 => feature size 128*2*2
        self.fc = nn.Sequential(
            nn.Linear(128 * 2 * 2, 64), nn.ReLU()
        )
        self.policy = nn.Linear(64, 4)
        self.value = nn.Linear(64, 1)
        nn.init.constant_(self.value.bias, 1.0)

    def forward(self, x):
        f = self.conv(x).view(x.size(0), -1)
        h = self.fc(f)
        return self.policy(h), self.value(h)

model = ActorCritic().to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)

# --- Phase-Based Action Selection ---
def select_action(obs):
    board = decode_board(obs)
    max_tile = int(board.max())
    x = preprocess(obs)
    logits, value = model(x)

    if max_tile < PHASE_THRESHOLDS[1]:
        dist = Categorical(logits=logits)
        action_tensor = dist.sample()
        logp_tensor = dist.log_prob(action_tensor)
        action = action_tensor.item()
        logp = logp_tensor.squeeze()
    else:
        action = torch.argmax(logits, dim=-1).item()
        logp = torch.tensor(0.0, device=device)

    return action, logp, value.squeeze()

# --- Rollout Collection ---
def collect_rollout():
    states, actions, rewards, dones, log_probs, values = [], [], [], [], [], []
    episode_returns = []
    obs, _ = env.reset()
    raw_return = 0.0

    for _ in range(ROLLOUT_STEPS):
        a, logp, val = select_action(obs)
        next_obs, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated

        shaped_r = r + 0.01 * monotonicity_bonus(decode_board(next_obs))

        states.append(preprocess(obs))
        actions.append(torch.tensor(a, device=device))
        rewards.append(torch.tensor(shaped_r, device=device))
        dones.append(torch.tensor(done, dtype=torch.float32, device=device))
        log_probs.append(logp)
        values.append(val)

        raw_return += r
        if done:
            episode_returns.append(raw_return)
            raw_return = 0.0
            obs, _ = env.reset()
        else:
            obs = next_obs

    if raw_return > 0:
        episode_returns.append(raw_return)

    return states, actions, rewards, dones, log_probs, values, episode_returns

# --- GAE and Return Calculation ---
def compute_advantages(rewards, values, dones):
    advantages, gae = [], 0
    values = values + [torch.tensor(0.0, device=device)]
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + GAMMA * values[t+1] * (1 - dones[t]) - values[t]
        gae = delta + GAMMA * LAMBDA * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    returns = [adv + val for adv, val in zip(advantages, values[:-1])]
    return advantages, returns

# --- Training Loop ---
def train():
    score_hist = deque(maxlen=ROLLING_AVG_WINDOW)
    pbar = tqdm(range(1, MAX_ITERATIONS + 1), desc="Training", unit="iter")

    for it in pbar:
        states, actions, rewards, dones, old_logps, values, ep_returns = collect_rollout()
        rollout_avg = np.mean(ep_returns)

        advs, rets = compute_advantages(rewards, values, dones)
        states = torch.cat(states)
        actions = torch.stack(actions)
        old_logps = torch.stack(old_logps).detach()
        advs = torch.stack(advs).detach()
        rets = torch.stack(rets).detach()
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        for _ in range(PPO_EPOCHS):
            logits, vals = model(states)
            dist = Categorical(logits=logits)
            new_logps = dist.log_prob(actions)
            ratio = (new_logps - old_logps).exp()
            s1 = ratio * advs
            s2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * advs
            policy_loss = -torch.min(s1, s2).mean()
            value_loss = (rets - vals.squeeze()).pow(2).mean()
            entropy = dist.entropy().mean()

            loss = policy_loss + 0.5 * value_loss - ENTROPY_COEF * entropy
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if it % STAT_UPDATE_INTERVAL == 0:
            score_hist.append(rollout_avg)
            rolling_avg = np.mean(score_hist)
            pbar.set_postfix({"RolloutAvg": f"{rollout_avg:.2f}", "RollingAvg": f"{rolling_avg:.2f}"})
            if len(score_hist) == ROLLING_AVG_WINDOW and rolling_avg >= ROLLING_AVG_THRESHOLD:
                pbar.write(f"Early stopping at iteration {it}, rolling avg {rolling_avg:.2f} >= threshold.")
                break

    pbar.close()
    print("Training finished.")

# --- Evaluation Pipeline ---
def evaluate():
    print("Starting evaluation...")
    model.eval()
    scores, tile_cnt, move_cnt = [], Counter(), defaultdict(int)
    for ep in range(1, NUM_EVAL_EPISODES + 1):
        obs, _ = env.reset()
        total = 0.0
        for _ in range(MAX_STEPS_PER_EPISODE):
            a, _, _ = select_action(obs)
            obs, r, term, trunc, _ = env.step(a)
            total += r
            move_cnt[ACTION_MAP[a]] += 1
            if term or trunc:
                break
        scores.append(total)
        tile_cnt[int(decode_board(obs).max())] += 1
        if ep % 100 == 0:
            print(f"Eval progress: {ep}/{NUM_EVAL_EPISODES} episodes")

    print("Evaluation complete.")
    print(f"Avg Score: {np.mean(scores):.2f}, Max Score: {np.max(scores):.2f}")
    print("Tile distribution:")
    for t, c in sorted(tile_cnt.items()):
        print(f"  {t}: {c}")
    print("Move distribution:")
    for m, c in move_cnt.items():
        print(f"  {m}: {c}")

# --- Main Runner ---
if __name__ == '__main__':
    train()
    evaluate()
