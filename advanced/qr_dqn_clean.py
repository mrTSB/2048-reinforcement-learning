import gymnasium as gym
import gymnasium_2048
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple, Counter
import random
from tqdm.auto import tqdm
import csv
import matplotlib.pyplot as plt
import os
import optuna
import argparse
import json

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make("gymnasium_2048/TwentyFortyEight-v0")

# --- Hyperparameters (default values for search) ---
BUFFER_SIZE     = int(1e6)    # replay buffer size
BATCH_SIZE      = 1024         # minibatch size
GAMMA           = 0.9010119619889969       # discount factor
LR              = 0.00013915748598497912       # learning rate
TAU             = 0.009903582546205186       # soft update rate
UPDATE_EVERY    = 4           # how often to learn
EPS_START       = 1.0         # starting epsilon
EPS_END         = 0.05        # final epsilon
EPS_DECAY       = 9.267487159659674e-05       # epsilon decay per step
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
    def __init__(self, hyperparams):
        # Unpack any hyperparameters passed in (or use defaults):
        global BUFFER_SIZE, BATCH_SIZE, GAMMA, LR, TAU, UPDATE_EVERY
        global EPS_START, EPS_END, EPS_DECAY

        BUFFER_SIZE = hyperparams.get("buffer_size", BUFFER_SIZE)
        BATCH_SIZE  = hyperparams.get("batch_size", BATCH_SIZE)
        GAMMA       = hyperparams.get("gamma", GAMMA)
        LR          = hyperparams.get("lr", LR)
        TAU         = hyperparams.get("tau", TAU)
        UPDATE_EVERY= hyperparams.get("update_every", UPDATE_EVERY)
        EPS_START   = hyperparams.get("eps_start", EPS_START)
        EPS_END     = hyperparams.get("eps_end", EPS_END)
        EPS_DECAY   = hyperparams.get("eps_decay", EPS_DECAY)

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

# --- Training Loop (wrapped to return metrics) ---
def train(hyperparams, log_csv_path=None, plot_path=None, early_stop=True):
    """
    Training function that:
      - Accepts a dict `hyperparams` to override defaults
      - Logs per-episode metrics into `log_csv_path` (if provided)
      - Saves a plot of training progress to `plot_path` (if provided)
      - If early_stop=True, uses rolling-average stopping criterion
    Returns:
      - agent          : the trained QRDQNAgent
      - all_metrics    : dict of lists of metrics logged per episode
    """
    agent = QRDQNAgent(hyperparams)

    episode_list     = []
    max_tile_list    = []
    avg_tile_list    = []
    epsilon_list     = []
    buffer_size_list = []

    max_tiles = []
    pbar = tqdm(range(1, hyperparams.get("num_episodes", NUM_EPISODES)+1), desc="Training")

    if log_csv_path is not None:
        os.makedirs(os.path.dirname(log_csv_path), exist_ok=True)
        csv_file = open(log_csv_path, mode='w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "episode", "step", "max_tile", "avg_max_tile",
            "epsilon", "buffer_size"
        ])

    for ep in pbar:
        obs, _ = env.reset()
        state = preprocess(obs)
        board_max = 0

        for step in range(1, hyperparams.get("max_steps", MAX_STEPS)+1):
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

        episode_list.append(ep)
        max_tile_list.append(board_max)
        avg_tile_list.append(avg_tile)
        epsilon_list.append(agent.epsilon)
        buffer_size_list.append(len(agent.memory))

        pbar.set_postfix({
            "Episode": ep,
            "Step": step,
            "MaxTile": board_max,
            "AvgMaxTile": f"{avg_tile:.1f}",
            "Epsilon": f"{agent.epsilon:.3f}",
            "Buffer": len(agent.memory)
        })

        if log_csv_path is not None:
            csv_writer.writerow([
                ep, step, board_max, f"{avg_tile:.3f}",
                f"{agent.epsilon:.4f}", len(agent.memory)
            ])

        if (early_stop
            and ep >= ROLLING_WINDOW
            and avg_tile >= hyperparams.get("tile_threshold", TILE_THRESHOLD)):
            pbar.write(f"Reached {hyperparams.get('tile_threshold', TILE_THRESHOLD)} "
                       f"at episode {ep} (avg max tile {avg_tile:.1f}).")
            break

    if log_csv_path is not None:
        csv_file.close()

    all_metrics = {
        "episode": episode_list,
        "max_tile": max_tile_list,
        "avg_tile": avg_tile_list,
        "epsilon": epsilon_list,
        "buffer_size": buffer_size_list
    }

    if plot_path is not None:
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.figure(figsize=(10,6))
        plt.plot(episode_list, max_tile_list, label="Per-Episode Max Tile", alpha=0.6)
        plt.plot(episode_list, avg_tile_list, label=f"{ROLLING_WINDOW}-Episode Avg Max Tile", color='red')
        plt.xlabel("Episode")
        plt.ylabel("Max Tile")
        plt.title("Training Progress: Max Tile vs. Episode")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

    return agent, all_metrics

# --- Evaluation Function (with average and best score) ---
def evaluate(agent, episodes=1000):
    """
    Plays `episodes` full games with the given agent (in near‐greedy mode),
    and at the end:
      - Prints how many times each max tile was reached.
      - Prints the average game score and the best (highest) game score over all episodes.
    """
    tiles = Counter()
    total_scores = []

    eval_pbar = tqdm(range(1, episodes+1), desc="Evaluating")
    for i in eval_pbar:
        obs, _ = env.reset()
        state = preprocess(obs)
        episode_score = 0.0

        while True:
            action = agent.act(state)
            next_obs, r, term, trunc, _ = env.step(action)
            episode_score += r
            state = preprocess(next_obs)
            if term or trunc:
                final_board = decode_board(next_obs)
                tiles[int(final_board.max())] += 1
                break

        total_scores.append(episode_score)
        eval_pbar.set_postfix({"Episode": i})

    avg_score = np.mean(total_scores)
    best_score = np.max(total_scores)

    print("\nEvaluation complete:")
    print(f"  → Average score over {episodes} episodes: {avg_score:.2f}")
    print(f"  → Best (highest) score seen: {best_score:.2f}\n")

    print("Final‐tile frequencies:")
    for tile, count in sorted(tiles.items()):
        print(f"  {tile}: {count}")

# -----------------------------------------------------------------------------------
# --- HYPERPARAMETER OPTIMIZATION WITH OPTUNA ---
# -----------------------------------------------------------------------------------
def objective(trial):
    # Suggest hyperparameters:
    hyperparams = {
        "lr": trial.suggest_loguniform("lr", 1e-5, 1e-3),
        "batch_size": trial.suggest_categorical("batch_size", [256, 512, 1024]),
        "gamma": trial.suggest_uniform("gamma", 0.90, 0.999),
        "tau": trial.suggest_uniform("tau", 1e-4, 1e-2),
        "eps_decay": trial.suggest_loguniform("eps_decay", 1e-6, 1e-4),
        # Keep others at defaults:
        "buffer_size": int(1e6),
        "update_every": 4,
        "eps_start": 1.0,
        "eps_end": 0.05,
        # Reduced episodes for tuning:
        "num_episodes": 500,
        "max_steps": 10000,
        "tile_threshold": 2048,
    }

    trial_id = trial.number
    log_csv = f"logs/trial_{trial_id}_metrics.csv"
    plot_png = f"logs/trial_{trial_id}_plot.png"

    agent, metrics = train(
        hyperparams,
        log_csv_path=log_csv,
        plot_path=plot_png,
        early_stop=False
    )

    last_avg = np.mean(metrics["max_tile"][-50:])
    return last_avg

# -----------------------------------------------------------------------------------
# --- MAIN ENTRYPOINT FOR FULL TRAINING OR HYPERPARAMETER SEARCH ---
# -----------------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="QR-DQN 2048 Training + Hyperparameter Tuning")
    parser.add_argument("--mode", choices=["train", "tune"], default="train",
                        help="train: run full training with default/best hyperparams; tune: run Optuna search")
    parser.add_argument("--n_trials", type=int, default=20,
                        help="Number of Optuna trials if mode=tune")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Directory to save logs, models, and plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == "tune":
        print(f"Starting hyperparameter tuning with {args.n_trials} trials...")
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=args.n_trials)

        print("=== Study statistics ===")
        print(f"  Number of finished trials: {len(study.trials)}")
        print(f"  Best trial index: {study.best_trial.number}")
        print(f"  Best trial value (avg max tile): {study.best_trial.value:.2f}")
        print("  Best hyperparameters:")
        for key, val in study.best_trial.params.items():
            print(f"    {key}: {val}")

        best_params_path = os.path.join(args.output_dir, "best_hyperparams.json")
        with open(best_params_path, "w") as f:
            json.dump(study.best_trial.params, f, indent=2)
        print(f"Best hyperparameters saved to {best_params_path}")

    else:
        print("Starting full training with default hyperparameters...")
        hyperparams = {
            "buffer_size": BUFFER_SIZE,
            "batch_size": BATCH_SIZE,
            "gamma": GAMMA,
            "lr": LR,
            "tau": TAU,
            "update_every": UPDATE_EVERY,
            "eps_start": EPS_START,
            "eps_end": EPS_END,
            "eps_decay": EPS_DECAY,
            "num_episodes": NUM_EPISODES,
            "max_steps": MAX_STEPS,
            "tile_threshold": TILE_THRESHOLD,
        }

        log_csv = os.path.join(args.output_dir, "training_metrics.csv")
        plot_png = os.path.join(args.output_dir, "training_plot.png")
        model_path = os.path.join(args.output_dir, "qrdqn_2048_model.pth")

        agent, metrics = train(
            hyperparams,
            log_csv_path=log_csv,
            plot_path=plot_png,
            early_stop=True
        )

        torch.save(agent.local.state_dict(), model_path)
        print(f"Trained model saved to {model_path}")

        print("\n=== Evaluation of the trained agent ===")
        evaluate(agent)
