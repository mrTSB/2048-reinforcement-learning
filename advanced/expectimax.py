import gymnasium as gym
import gymnasium_2048
import numpy as np
from tqdm.auto import tqdm

# --- Heuristic-Based Expectimax Agent ---
def decode_board(obs):
    if obs.ndim == 3:
        idxs = np.argmax(obs, axis=-1)
        mask = (obs.sum(axis=-1) == 1)
        return (2 ** idxs) * mask
    return obs

# Heuristic combining empty cells, monotonicity, and max-tile-in-corner

def heuristic(board):
    empty = np.count_nonzero(board == 0)
    # monotonicity: reward smooth gradients in rows and cols
    mono = 0
    for i in range(4):
        for j in range(3):
            mono += -abs(board[i, j] - board[i, j+1])
            mono += -abs(board[j, i] - board[j+1, i])
    # encourage max tile in corner
    max_tile = board.max()
    corner_bonus = 0
    if board[0, 0] == max_tile or board[0, 3] == max_tile or board[3, 0] == max_tile or board[3, 3] == max_tile:
        corner_bonus = np.log2(max_tile)
    return empty + 0.1 * mono + corner_bonus

# Generate all possible successor boards and probabilities for a random tile

def get_children_chance(board):
    children = []
    empties = list(zip(*np.where(board == 0)))
    for (i, j) in empties:
        for val, p in [(2, 0.9), (4, 0.1)]:
            new = board.copy()
            new[i, j] = val
            children.append((new, p / len(empties)))
    return children

# Expectimax search

def expectimax(board, depth, is_player):
    if depth == 0 or not board.any():
        return heuristic(board)
    if is_player:
        best = -np.inf
        for action in range(4):
            env_board = board.copy()
            # simulate move
            new_board, moved = simulate_move(env_board, action)
            if not moved:
                continue
            val = expectimax(new_board, depth - 1, False)
            if val > best:
                best = val
        return best if best != -np.inf else heuristic(board)
    else:
        # chance node
        val = 0
        for child, prob in get_children_chance(board):
            val += prob * expectimax(child, depth - 1, True)
        return val

# Slide and merge logic

def simulate_move(board, action):
    moved = False
    rotated = np.rot90(board, -action)
    new = np.zeros_like(board)
    for i in range(4):
        line = rotated[i][rotated[i] > 0]
        merged = []
        skip = False
        for j in range(len(line)):
            if skip:
                skip = False
                continue
            if j + 1 < len(line) and line[j] == line[j+1]:
                merged.append(line[j] * 2)
                skip = True
                moved = True
            else:
                merged.append(line[j])
        if len(merged) != len(line): moved = True
        new[i, :len(merged)] = merged
    new = np.rot90(new, action)
    return new, moved

# Choose best move via expectimax

def select_move(board, depth=2):
    best_action = 0
    best_val = -np.inf
    for action in range(4):
        new_board, moved = simulate_move(board.copy(), action)
        if not moved:
            continue
        val = expectimax(new_board, depth - 1, False)
        if val > best_val:
            best_val = val
            best_action = action
    return best_action

# --- Main Play Loop ---
def play_games(n_games=100):
    env = gym.make("gymnasium_2048/TwentyFortyEight-v0")
    scores = []
    for _ in tqdm(range(n_games), desc="Playing"):
        obs, _ = env.reset()
        board = decode_board(obs)
        score = 0
        done = False
        while not done:
            action = select_move(board, depth=2)
            obs, r, term, trunc, _ = env.step(action)
            done = term or trunc
            board = decode_board(obs)
            score += r
        scores.append(score)
    print(f"Average score over {n_games} games: {np.mean(scores):.2f}")
    print(f"Max score: {np.max(scores)}")

if __name__ == '__main__':
    play_games(100)
