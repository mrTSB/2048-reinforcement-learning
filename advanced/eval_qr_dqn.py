import gymnasium as gym
import gymnasium_2048
import numpy as np
import torch
import torch.nn as nn
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
import json
import argparse
import os
from tqdm.auto import tqdm
import seaborn as sns

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Constants
N_QUANT = 51  # number of quantiles (must match training)

# --- Helper Functions (copied from training script) ---
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

# --- Quantile Network (copied from training script) ---
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

# --- Agent wrapper for evaluation ---
class EvaluationAgent:
    def __init__(self, model_path):
        self.model = QuantileNet().to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        
    def act(self, state, epsilon=0.01):
        """Act with low epsilon for near-greedy behavior"""
        if np.random.random() < epsilon:
            return np.random.randint(4)
        
        with torch.no_grad():
            quantiles = self.model(state)
            q_vals = quantiles.mean(dim=2)
            return int(q_vals.argmax(dim=1).item())

def evaluate_comprehensive(agent, episodes=1000, save_individual_games=False):
    """
    Comprehensive evaluation that tracks detailed metrics
    """
    env = gym.make("gymnasium_2048/TwentyFortyEight-v0")
    
    # Tracking variables
    tiles_reached = Counter()
    scores = []
    game_lengths = []
    boards_final = []
    individual_games = []
    
    eval_pbar = tqdm(range(episodes), desc="Evaluating")
    
    for game_idx in eval_pbar:
        obs, _ = env.reset()
        state = preprocess(obs)
        episode_score = 0.0
        steps = 0
        
        game_history = {
            'game_id': game_idx,
            'moves': [],
            'scores': [],
            'max_tiles': []
        }
        
        while True:
            action = agent.act(state)
            next_obs, reward, term, trunc, _ = env.step(action)
            episode_score += reward
            steps += 1
            
            current_board = decode_board(next_obs)
            current_max_tile = int(current_board.max())
            
            if save_individual_games:
                game_history['moves'].append(action)
                game_history['scores'].append(episode_score)
                game_history['max_tiles'].append(current_max_tile)
            
            state = preprocess(next_obs)
            
            if term or trunc:
                final_board = decode_board(next_obs)
                max_tile = int(final_board.max())
                tiles_reached[max_tile] += 1
                scores.append(episode_score)
                game_lengths.append(steps)
                boards_final.append(final_board.tolist())
                
                if save_individual_games:
                    game_history['final_score'] = episode_score
                    game_history['final_max_tile'] = max_tile
                    game_history['steps'] = steps
                    individual_games.append(game_history)
                
                break
        
        eval_pbar.set_postfix({
            'Game': game_idx + 1,
            'Score': f'{episode_score:.0f}',
            'MaxTile': max_tile,
            'AvgScore': f'{np.mean(scores):.1f}'
        })
    
    env.close()
    
    # Compile results
    results = {
        'summary': {
            'total_games': episodes,
            'avg_score': float(np.mean(scores)),
            'std_score': float(np.std(scores)),
            'min_score': float(np.min(scores)),
            'max_score': float(np.max(scores)),
            'median_score': float(np.median(scores)),
            'avg_game_length': float(np.mean(game_lengths)),
            'std_game_length': float(np.std(game_lengths)),
        },
        'tile_frequencies': dict(tiles_reached),
        'score_percentiles': {
            '25th': float(np.percentile(scores, 25)),
            '50th': float(np.percentile(scores, 50)),
            '75th': float(np.percentile(scores, 75)),
            '90th': float(np.percentile(scores, 90)),
            '95th': float(np.percentile(scores, 95)),
            '99th': float(np.percentile(scores, 99)),
        },
        'detailed_data': {
            'scores': scores,
            'game_lengths': game_lengths,
            'final_boards': boards_final
        }
    }
    
    if save_individual_games:
        results['individual_games'] = individual_games
    
    return results

def create_visualizations(results, output_dir):
    """Create and save comprehensive visualizations"""
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Score distribution
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    scores = results['detailed_data']['scores']
    plt.hist(scores, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(results['summary']['avg_score'], color='red', linestyle='--', 
                label=f"Mean: {results['summary']['avg_score']:.1f}")
    plt.axvline(results['summary']['median_score'], color='orange', linestyle='--',
                label=f"Median: {results['summary']['median_score']:.1f}")
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.title('Score Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Tile frequency
    plt.subplot(2, 2, 2)
    tiles = sorted(results['tile_frequencies'].keys())
    frequencies = [results['tile_frequencies'][tile] for tile in tiles]
    colors = plt.cm.viridis(np.linspace(0, 1, len(tiles)))
    
    bars = plt.bar([str(tile) for tile in tiles], frequencies, color=colors)
    plt.xlabel('Maximum Tile Reached')
    plt.ylabel('Frequency')
    plt.title('Maximum Tile Frequency')
    plt.xticks(rotation=45)
    
    # Add percentage labels on bars
    total_games = results['summary']['total_games']
    for bar, freq in zip(bars, frequencies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(frequencies),
                f'{100*freq/total_games:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # 3. Game length distribution
    plt.subplot(2, 2, 3)
    game_lengths = results['detailed_data']['game_lengths']
    plt.hist(game_lengths, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    plt.axvline(results['summary']['avg_game_length'], color='red', linestyle='--',
                label=f"Mean: {results['summary']['avg_game_length']:.1f}")
    plt.xlabel('Game Length (Steps)')
    plt.ylabel('Frequency')
    plt.title('Game Length Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Score vs Game Length scatter
    plt.subplot(2, 2, 4)
    plt.scatter(game_lengths, scores, alpha=0.6, s=20)
    plt.xlabel('Game Length (Steps)')
    plt.ylabel('Score')
    plt.title('Score vs Game Length')
    plt.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(game_lengths, scores, 1)
    p = np.poly1d(z)
    plt.plot(sorted(game_lengths), p(sorted(game_lengths)), "r--", alpha=0.8, linewidth=2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'evaluation_overview.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Detailed tile analysis
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    # Cumulative tile distribution
    tiles = sorted(results['tile_frequencies'].keys())
    frequencies = [results['tile_frequencies'][tile] for tile in tiles]
    cumulative_freq = np.cumsum(frequencies)
    cumulative_pct = 100 * cumulative_freq / total_games
    
    plt.plot([str(tile) for tile in tiles], cumulative_pct, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Maximum Tile Reached')
    plt.ylabel('Cumulative Percentage')
    plt.title('Cumulative Tile Achievement')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Add percentage labels
    for i, (tile, pct) in enumerate(zip(tiles, cumulative_pct)):
        plt.annotate(f'{pct:.1f}%', (i, pct), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=9)
    
    # 6. Box plot of scores by max tile
    plt.subplot(1, 2, 2)
    
    # Group scores by max tile achieved
    tile_score_groups = {}
    for i, score in enumerate(scores):
        final_board = results['detailed_data']['final_boards'][i]
        max_tile = int(np.max(final_board))
        if max_tile not in tile_score_groups:
            tile_score_groups[max_tile] = []
        tile_score_groups[max_tile].append(score)
    
    # Create box plot
    box_data = []
    box_labels = []
    for tile in sorted(tile_score_groups.keys()):
        if len(tile_score_groups[tile]) >= 3:  # Only include if we have enough data points
            box_data.append(tile_score_groups[tile])
            box_labels.append(str(tile))
    
    if box_data:
        plt.boxplot(box_data, labels=box_labels)
        plt.xlabel('Maximum Tile Reached')
        plt.ylabel('Score')
        plt.title('Score Distribution by Max Tile')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tile_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

def save_results(results, output_dir, format_types=['json', 'csv']):
    """Save results in multiple formats"""
    
    if 'json' in format_types:
        # Save complete results as JSON
        json_path = os.path.join(output_dir, 'evaluation_results.json')
        with open(json_path, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            json_results = json.loads(json.dumps(results, default=str))
            json.dump(json_results, f, indent=2)
        print(f"Results saved to {json_path}")
    
    if 'csv' in format_types:
        # Save summary as CSV
        summary_df = pd.DataFrame([results['summary']])
        summary_path = os.path.join(output_dir, 'evaluation_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        
        # Save detailed scores and game data
        detailed_df = pd.DataFrame({
            'game_id': range(len(results['detailed_data']['scores'])),
            'score': results['detailed_data']['scores'],
            'game_length': results['detailed_data']['game_lengths'],
            'max_tile': [int(np.max(board)) for board in results['detailed_data']['final_boards']]
        })
        detailed_path = os.path.join(output_dir, 'evaluation_detailed.csv')
        detailed_df.to_csv(detailed_path, index=False)
        
        # Save tile frequencies
        tile_freq_df = pd.DataFrame([
            {'tile': tile, 'frequency': freq, 'percentage': 100*freq/results['summary']['total_games']}
            for tile, freq in results['tile_frequencies'].items()
        ])
        tile_freq_path = os.path.join(output_dir, 'tile_frequencies.csv')
        tile_freq_df.to_csv(tile_freq_path, index=False)
        
        print(f"CSV files saved to {output_dir}")

def print_summary(results):
    """Print a nice summary of evaluation results"""
    summary = results['summary']
    
    print("\n" + "="*60)
    print("🎮 QR-DQN 2048 EVALUATION RESULTS")
    print("="*60)
    
    print(f"\n📊 GAME STATISTICS ({summary['total_games']} games)")
    print(f"   Average Score:    {summary['avg_score']:8.1f} ± {summary['std_score']:.1f}")
    print(f"   Median Score:     {summary['median_score']:8.1f}")
    print(f"   Best Score:       {summary['max_score']:8.1f}")
    print(f"   Worst Score:      {summary['min_score']:8.1f}")
    print(f"   Average Length:   {summary['avg_game_length']:8.1f} ± {summary['std_game_length']:.1f} steps")
    
    print(f"\n🎯 SCORE PERCENTILES")
    percs = results['score_percentiles']
    print(f"   25th percentile:  {percs['25th']:8.1f}")
    print(f"   50th percentile:  {percs['50th']:8.1f}")
    print(f"   75th percentile:  {percs['75th']:8.1f}")
    print(f"   90th percentile:  {percs['90th']:8.1f}")
    print(f"   95th percentile:  {percs['95th']:8.1f}")
    print(f"   99th percentile:  {percs['99th']:8.1f}")
    
    print(f"\n🏆 TILE ACHIEVEMENT RATES")
    total_games = summary['total_games']
    tiles = sorted(results['tile_frequencies'].keys(), reverse=True)
    
    for tile in tiles:
        count = results['tile_frequencies'][tile]
        percentage = 100 * count / total_games
        print(f"   {tile:>4}: {count:5d} games ({percentage:5.1f}%)")
    
    print("\n" + "="*60)

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained QR-DQN 2048 model")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained model (.pth file)")
    parser.add_argument("--episodes", type=int, default=1000,
                        help="Number of evaluation episodes")
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                        help="Directory to save results and plots")
    parser.add_argument("--save_individual_games", action="store_true",
                        help="Save detailed data for individual games (increases file size)")
    parser.add_argument("--epsilon", type=float, default=0.01,
                        help="Epsilon for evaluation (0.0 = fully greedy)")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading model from {args.model_path}...")
    try:
        agent = EvaluationAgent(args.model_path)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    print(f"\nStarting evaluation with {args.episodes} episodes...")
    print(f"Using epsilon = {args.epsilon} (0.0 = fully greedy)")
    
    # Run evaluation
    results = evaluate_comprehensive(
        agent, 
        episodes=args.episodes,
        save_individual_games=args.save_individual_games
    )
    
    # Print summary
    print_summary(results)
    
    # Save results
    print(f"\nSaving results to {args.output_dir}...")
    save_results(results, args.output_dir)
    
    # Create visualizations
    print("Creating visualizations...")
    create_visualizations(results, args.output_dir)
    
    print(f"\n✅ Evaluation complete! Check {args.output_dir} for results and plots.")

if __name__ == "__main__":
    main() 