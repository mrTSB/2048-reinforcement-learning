import gymnasium as gym
import gymnasium_2048
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

# --- Hyperparameters ---
COLLECT_EPISODES = 500      # Number of games to collect expert data
TRAIN_BATCH_SIZE = 256      # Batch size for training
TRAIN_EPOCHS     = 10       # Number of epochs
LEARNING_RATE    = 1e-3     # Learning rate
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Helper Functions ---
def decode_board(obs):
    if obs.ndim == 3:
        idxs = np.argmax(obs, axis=-1)
        mask = (obs.sum(axis=-1) == 1)
        return (2 ** idxs) * mask
    return obs

# One-hot + log2 indexing for network input
def preprocess(obs):
    board = decode_board(obs).astype(int)
    idxs = np.zeros_like(board, dtype=int)
    nonzero = board > 0
    idxs[nonzero] = np.log2(board[nonzero]).astype(int)
    channels = [(idxs == i).astype(np.float32) for i in range(16)]
    x = np.stack(channels, axis=0)
    return x  # shape [16,4,4]

# --- Heuristic Expectimax Expert ---
def heuristic(board):
    empty = np.count_nonzero(board == 0)
    mono = 0
    for i in range(4):
        for j in range(3):
            mono -= abs(board[i, j] - board[i, j+1])
            mono -= abs(board[j, i] - board[j+1, i])
    max_tile = board.max()
    corner = 0
    if board[0, 0] == max_tile or board[0, 3] == max_tile or board[3, 0] == max_tile or board[3, 3] == max_tile:
        corner = np.log2(max_tile)
    return empty + 0.1 * mono + corner

def get_children(board):
    empties = list(zip(*np.where(board == 0)))
    for (i, j) in empties:
        for val, p in [(2, 0.9), (4, 0.1)]:
            new = board.copy()
            new[i, j] = val
            yield new, p / len(empties)

def simulate_move(board, action):
    moved = False
    rotated = np.rot90(board, -action)
    new_board = np.zeros_like(board)
    for i, row in enumerate(rotated):
        tiles = row[row > 0]
        merged = []
        skip = False
        for k in range(len(tiles)):
            if skip:
                skip = False
                continue
            if k + 1 < len(tiles) and tiles[k] == tiles[k+1]:
                merged.append(tiles[k] * 2)
                skip = True
                moved = True
            else:
                merged.append(tiles[k])
        if len(merged) != len(tiles):
            moved = True
        new_board[i, :len(merged)] = merged
    return np.rot90(new_board, action), moved

def expectimax(board, depth, player):
    if depth == 0:
        return heuristic(board)
    if player:
        best = -np.inf
        for a in range(4):
            new_board, moved = simulate_move(board, a)
            if not moved:
                continue
            val = expectimax(new_board, depth - 1, False)
            best = max(best, val)
        return best if best != -np.inf else heuristic(board)
    else:
        total = 0
        for child, prob in get_children(board):
            total += prob * expectimax(child, depth - 1, True)
        return total

def expert_move(board):
    best_val = -np.inf
    best_act = 0
    for a in range(4):
        new_board, moved = simulate_move(board, a)
        if not moved:
            continue
        val = expectimax(new_board, 2, False)
        if val > best_val:
            best_val = val
            best_act = a
    return best_act

# --- Data Collection ---
def collect_data(episodes=COLLECT_EPISODES):
    env = gym.make("gymnasium_2048/TwentyFortyEight-v0")
    states, actions = [], []
    for _ in tqdm(range(episodes), desc="Collecting Expert Data", unit="episode"):
        obs, _ = env.reset()
        done = False
        while not done:
            board = decode_board(obs)
            s = preprocess(obs)
            a = expert_move(board)
            obs, _, term, trunc, _ = env.step(a)
            done = term or trunc
            states.append(s)
            actions.append(a)
    return np.array(states), np.array(actions)

# --- Policy Network ---
class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(16, 64, 2), nn.ReLU(),
            nn.Conv2d(64, 128, 2), nn.ReLU(),
            nn.Flatten(), nn.Linear(128*2*2, 64), nn.ReLU(),
            nn.Linear(64, 4)
        )
    def forward(self, x):
        return self.net(x)

# --- Training ---
def train_model(states, actions):
    dataset = TensorDataset(
        torch.tensor(states, dtype=torch.float32),
        torch.tensor(actions, dtype=torch.long)
    )
    loader = DataLoader(dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    model = PolicyNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(1, TRAIN_EPOCHS + 1):
        total, correct = 0, 0
        for xb, yb in tqdm(loader, desc=f"Epoch {epoch}", unit="batch", leave=False):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            preds = logits.argmax(dim=1)
            total += yb.size(0)
            correct += (preds == yb).sum().item()
        acc = 100 * correct / total
        print(f"Epoch {epoch}: Accuracy {acc:.2f}%")
    return model

# --- Evaluation ---
def play_with_model(model, n_games=100):
    env = gym.make("gymnasium_2048/TwentyFortyEight-v0")
    model.eval()
    scores = []
    for _ in tqdm(range(n_games), desc="Playing with Model", unit="game"):
        obs, _ = env.reset()
        done = False
        score = 0
        while not done:
            state_tensor = torch.tensor(
                preprocess(obs), dtype=torch.float32
            ).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                action = model(state_tensor).argmax(dim=1).item()
            obs, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            score += reward
        scores.append(score)
    print(f"Model Avg Score: {np.mean(scores):.2f}, Max: {np.max(scores)}")

# --- Main ---
if __name__ == '__main__':
    states, actions = collect_data()
    policy = train_model(states, actions)
    play_with_model(policy)
