"""
Synapse Benchmark - Trajectory Rollout Script
Generate trajectories from trained policies for downstream tasks
"""

import argparse
import yaml
import os
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path
import re

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from core import *
from utils_latentPolicy_sac_lstm_zt_zt1_early_termination import *

import tensorflow_probability as tfp


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_environment(config):
    """Setup GPU and random seeds."""
    os.environ["CUDA_VISIBLE_DEVICES"] = config['hardware']['cuda_visible_devices']
    
    tf_config = tf.ConfigProto(log_device_placement=False)
    tf_config.gpu_options.allow_growth = config['hardware']['gpu_memory_allow_growth']
    
    # Set random seeds
    seed = config['training']['random_seed']
    np.random.seed(seed)
    tf.set_random_seed(seed)
    
    return tf_config


def find_best_checkpoint(results_dir, env_name):
    """
    Find the best checkpoint based on MAE and RANK metrics.
    
    Args:
        results_dir: Directory containing result .txt files
        env_name: Environment name to filter results
    
    Returns:
        tuple: (best_file, best_mae, best_rank)
    """
    best_file = None
    best_mae = float("inf")
    best_rank = -float("inf")
    
    pattern = re.compile(r"MAE:\s*([\d.]+)\s*\|\s*RANK:\s*([\d.]+)")
    
    if not os.path.exists(results_dir):
        print(f"Warning: Results directory not found: {results_dir}")
        return None, None, None
    
    for fname in os.listdir(results_dir):
        if fname.endswith(".txt") and env_name in fname:
            path = os.path.join(results_dir, fname)
            
            # Read the last non-empty line
            with open(path, "r") as f:
                lines = [ln.strip() for ln in f.readlines() if ln.strip()]
                if not lines:
                    continue
                last_line = lines[-1]
            
            # Extract MAE and RANK from the line
            m = pattern.search(last_line)
            if not m:
                continue
            
            mae = float(m.group(1))
            rank = float(m.group(2))
            
            # Track best by MAE (lower) then RANK (higher)
            if (mae < best_mae) or (mae == best_mae and rank > best_rank):
                best_mae = mae
                best_rank = rank
                best_file = fname
    
    if best_file:
        print(f"Best checkpoint found: {best_file}")
        print(f"  MAE: {best_mae:.2f}")
        print(f"  RANK: {best_rank:.4f}")
    
    return best_file, best_mae, best_rank


def rollout_trajectories(args):
    """Main rollout function."""
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.env_name:
        config['environment']['name'] = args.env_name
    if args.raw_data_path:
        config['data']['raw_data_path'] = args.raw_data_path
    if args.checkpoint_dir:
        config['data']['checkpoint_dir'] = args.checkpoint_dir
    if args.output_dir:
        config['data']['output_dir'] = args.output_dir
    if args.target_policy_dir:
        args.target_policy_dir = args.target_policy_dir
    if args.seed is not None:
        config['training']['random_seed'] = args.seed
    
    # Setup environment
    tf_config = setup_environment(config)
    
    # Extract parameters
    env_name = config['environment']['name']
    seed = config['training']['random_seed']
    
    print(f"Rolling out trajectories for {env_name}")
    print(f"Seed: {seed}")
    
    # Construct file paths
    raw_data_path = config['data']['raw_data_path'].format(ENV_NAME=env_name)
    checkpoint_dir = config['data']['checkpoint_dir'].format(ENV_NAME=env_name)
    output_dir = config['data']['output_dir']
    results_dir = config['data']['results_dir']
    
    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print(f"Loading data from: {raw_data_path}")
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(f"Data file not found: {raw_data_path}")
    
    raw_data = np.load(raw_data_path, allow_pickle=True).item()
    
    # Find best checkpoint if auto mode
    if args.auto_find_best:
        best_file, best_mae, best_rank = find_best_checkpoint(results_dir, env_name)
        if best_file is None:
            raise ValueError("No valid checkpoint found in results directory")
        # Extract checkpoint path from result filename
        checkpoint_name = best_file.replace('.txt', '')
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    else:
        if args.checkpoint_path is None:
            raise ValueError("Must provide --checkpoint_path or use --auto_find_best")
        checkpoint_path = args.checkpoint_path
    
    print(f"Using checkpoint: {checkpoint_path}")
    
    # Target policy directory
    target_policy_dir = args.target_policy_dir or f"checkpoints/target_policies/{env_name}"
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Initialize TensorFlow session
    with tf.Session(config=tf_config) as sess:
        # TODO: Add your rollout logic here
        # This is where you would:
        # 1. Load the trained OPE model
        # 2. Load target policies
        # 3. Roll out trajectories
        # 4. Save trajectory data for downstream tasks
        
        print("Rollout started...")
        print("Note: Complete rollout implementation depends on your specific environment class")
        
        # Policy parameters to evaluate (example)
        policy_params = args.policy_params or ['param3_distilled', 'param3', 'param2_distilled', 'param4']
        
        for param in policy_params:
            print(f"Rolling out policy: {param}")
            # Load policy from target_policy_dir/param/
            # Generate trajectories
            # Save to output_dir
        
        print(f"Trajectories saved to: {output_dir}")
    
    print("Rollout completed!")


def main():
    parser = argparse.ArgumentParser(description='Rollout trajectories from trained policies')
    
    # Configuration file
    parser.add_argument('--config', type=str, default='configs/config_template.yaml',
                        help='Path to configuration YAML file')
    
    # Override options
    parser.add_argument('--env_name', type=str, default=None,
                        help='Environment name (e.g., RZCH_clinical)')
    parser.add_argument('--raw_data_path', type=str, default=None,
                        help='Path to raw data .npy file')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Directory containing trained checkpoints')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Specific checkpoint path to use')
    parser.add_argument('--target_policy_dir', type=str, default=None,
                        help='Directory containing target policies')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save rolled out trajectories')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed')
    parser.add_argument('--auto_find_best', action='store_true',
                        help='Automatically find best checkpoint from results')
    parser.add_argument('--policy_params', nargs='+', default=None,
                        help='List of policy parameters to evaluate')
    
    args = parser.parse_args()
    
    rollout_trajectories(args)


if __name__ == '__main__':
    main()
