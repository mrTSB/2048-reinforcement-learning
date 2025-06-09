# 2048 Reinforcement Learning

This repository contains the final project for CS224r, focused on deep reinforcement learning approaches for the game 2048. The project implements and compares advanced RL algorithms, including distributional and recurrent DQN variants, quantile regression DQN, PPO, and more.

## Overview

- **Project Goal:**  
  To explore and benchmark advanced deep reinforcement learning algorithms on the 2048 game environment, with a focus on distributional RL, recurrent architectures, and quantile-based methods.
- **Technologies Used:**  
  - Python 3.12+
  - PyTorch
  - Gymnasium & gymnasium_2048
  - Jupyter Notebooks
  - NumPy, TQDM, Matplotlib

## Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/cs224r_final_project.git
   cd cs224r_final_project
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **(Optional) Set up a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   ```

## Project Structure

- `main.py` — Simple entry point (prints a hello message).
- `rdqn_r2d2/` — Main directory for advanced DQN variants:
  - `rdqn_r2d2.py` — Recurrent Distributional DQN (R2D2) with noisy layers and prioritized replay.
  - `rdqn_seq.py`, `rdqn_lstm.py`, `rdqn_basic.py` — Other DQN variants and ablations.
  - `rdqn_r2d2_training.ipynb` — Notebook for training and evaluating the R2D2 agent on 2048.
- `baselines/` — Baseline RL implementations:
  - `ppo_baseline.ipynb` — PPO agent for 2048.
  - `dqn_baseline.ipynb` — DQN baseline for 2048.
  - `game_2048.py` — Custom 2048 Gymnasium environment implementation.
- `advanced/` — Advanced and experimental agents:
  - `QR-DQN.py`, `qr_dqn_clean.py`, `eval_qr_dqn.py` — Quantile Regression DQN and evaluation scripts.
  - `cnn.py`, `expectimax.py` — CNN-based and expectimax agents.
  - `tanvir_scratch.py` — Experimental agent.
- `biases/` — Additional experiments (e.g., dueling DQN).
- `requirements.txt` — Python dependencies.
- `pyproject.toml` — Project metadata.
- `README.md` — Project documentation.

## Usage

### Training the R2D2 Agent

1. Open `rdqn_r2d2/rdqn_r2d2_training.ipynb` in Jupyter Notebook or VSCode.
2. Run all cells to train the agent on the 2048 environment.
3. The notebook includes code for training, evaluation, and plotting results.

### Running Baselines

- **PPO Baseline:**  
  Open `baselines/ppo_baseline.ipynb` and run the notebook to train and evaluate a PPO agent.
- **DQN Baseline:**  
  Open `baselines/dqn_baseline.ipynb` for a standard DQN implementation.

### Custom 2048 Environment

- The project uses both the `gymnasium_2048` environment and a custom implementation in `baselines/game_2048.py`.

### Advanced Agents

- For quantile-based DQN, see `advanced/QR-DQN.py` and related files.

## Notebooks

- All major experiments and training loops are provided as Jupyter notebooks for easy reproducibility and visualization.

## Authors

- Tanvir Bhathal, Prady Saligram, Robby Manihani

---

### Acknowledgements

- The project builds on open-source libraries such as Gymnasium, PyTorch, and others.
- 2048 environment adapted from [gymnasium_2048](https://github.com/Farama-Foundation/Gymnasium-2048).

---

**Feel free to open issues or pull requests for questions, improvements, or contributions!**
