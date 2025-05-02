import gymnasium as gym
import numpy as np
from gymnasium import spaces

class Game2048Env(gym.Env):
    def __init__(self):
        super().__init__()
        self.size = 4
        self.board = None
        
        # Action space: 0=left, 1=right, 2=up, 3=down
        self.action_space = spaces.Discrete(4)
        
        # Observation space: 4x4 board with values up to 2048
        self.observation_space = spaces.Box(
            low=0, high=2048, shape=(4, 4), dtype=np.int32
        )
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.board = np.zeros((self.size, self.size), dtype=np.int32)
        # Add two initial tiles
        self._add_new_tile()
        self._add_new_tile()
        return self.board, {}

    def _add_new_tile(self):
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            x, y = self.np_random.choice(len(empty_cells))
            value = 2 if self.np_random.random() < 0.9 else 4
            self.board[empty_cells[x]] = value

    def _merge(self, line):
        # Remove zeros and merge identical numbers
        non_zero = line[line != 0]
        merged = []
        i = 0
        score = 0
        while i < len(non_zero):
            if i + 1 < len(non_zero) and non_zero[i] == non_zero[i + 1]:
                merged.append(non_zero[i] * 2)
                score += non_zero[i] * 2
                i += 2
            else:
                merged.append(non_zero[i])
                i += 1
        # Pad with zeros
        merged.extend([0] * (self.size - len(merged)))
        return np.array(merged), score

    def _move(self, action):
        original_board = self.board.copy()
        score = 0
        
        # Rotate board to make all moves like left move
        if action == 1:  # right
            self.board = np.flip(self.board, axis=1)
        elif action == 2:  # up
            self.board = self.board.T
        elif action == 3:  # down
            self.board = np.flip(self.board.T, axis=0)
            
        # Merge each row
        for i in range(self.size):
            self.board[i], move_score = self._merge(self.board[i])
            score += move_score
            
        # Rotate back
        if action == 1:  # right
            self.board = np.flip(self.board, axis=1)
        elif action == 2:  # up
            self.board = self.board.T
        elif action == 3:  # down
            self.board = np.flip(self.board.T, axis=0)
            
        # Check if board changed
        if not np.array_equal(original_board, self.board):
            self._add_new_tile()
            
        return score

    def step(self, action):
        if not 0 <= action <= 3:
            raise ValueError("Invalid action")
            
        score = self._move(action)
        
        # Check game over
        done = True
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == 0:
                    done = False
                    break
                if i < self.size - 1 and self.board[i][j] == self.board[i + 1][j]:
                    done = False
                    break
                if j < self.size - 1 and self.board[i][j] == self.board[i][j + 1]:
                    done = False
                    break
            if not done:
                break
                
        return self.board, score, done, False, {}

    def render(self):
        print(self.board)