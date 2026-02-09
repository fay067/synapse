"""
Synapse Benchmark - Policy Learning Script
Train offline RL policies using various algorithms (CQL, etc.)
"""

import argparse
import yaml
import os
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from core import *
from utils_latentPolicy_sac_lstm_zt_zt1_early_termination import *

import tensorflow_probability as tfp
from d3rlpy.algos import CQL, CQLConfig
from d3rlpy.models.optimizers import AdamFactory
from d3rlpy.datasets import MDPDataset


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


def train_policy(args):
    """Main training function."""
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.env_name:
        config['environment']['name'] = args.env_name
    if args.raw_data_path:
        config['data']['raw_data_path'] = args.raw_data_path
    if args.checkpoint_dir:
        config['data']['checkpoint_dir'] = args.checkpoint_dir
    if args.results_dir:
        config['data']['results_dir'] = args.results_dir
    if args.seed is not None:
        config['training']['random_seed'] = args.seed
    if args.max_episodes is not None:
        config['training']['max_episodes'] = args.max_episodes
    if args.beta is not None:
        config['training']['beta'] = args.beta
    
    # Setup environment
    tf_config = setup_environment(config)
    
    # Extract parameters
    env_name = config['environment']['name']
    max_episodes = config['training']['max_episodes']
    gamma = config['training']['gamma']
    seed = config['training']['random_seed']
    ope_lr = config['training']['ope_learning_rate']
    ope_ds = config['training']['ope_ds']
    ope_dr = config['training']['ope_dr']
    code_size = config['training']['code_size']
    beta = config['training']['beta']
    
    print(f"Training policy for {env_name}")
    print(f"Configuration: max_episodes={max_episodes}, seed={seed}, beta={beta}")
    
    # Construct file paths
    raw_data_path = config['data']['raw_data_path'].format(ENV_NAME=env_name)
    checkpoint_dir = config['data']['checkpoint_dir'].format(ENV_NAME=env_name)
    results_dir = config['data']['results_dir']
    
    # Create directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Load data
    print(f"Loading data from: {raw_data_path}")
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(f"Data file not found: {raw_data_path}")
    
    raw_data = np.load(raw_data_path, allow_pickle=True).item()
    
    # Environment setup
    window = config['environment']['window_size']
    env_state_dim = window
    env_action_dim = config['environment']['action_dim']
    env_action_bound = config['environment']['action_bound']
    
    # Create file appendix for saving
    file_appendix = (
        f"OPE_SAC_latentPolicy_lstm_zt_zt1_d4rlOnly_{env_name}_{max_episodes}epi_"
        f"repeat{config['training']['repeat']}_{ope_lr}_{ope_ds}_{ope_dr}_{code_size}_{beta}_{seed}"
    )
    
    ope_path = os.path.join(checkpoint_dir, file_appendix)
    os.makedirs(ope_path, exist_ok=True)
    
    print(f"Checkpoint path: {ope_path}")
    
    # Initialize TensorFlow session
    with tf.Session(config=tf_config) as sess:
        # TODO: Add your training logic here
        # This is where you would:
        # 1. Initialize your environment
        # 2. Load the OPE model
        # 3. Create the policy learning algorithm (CQL, etc.)
        # 4. Train the policy
        # 5. Evaluate and save results
        
        print("Training started...")
        print("Note: Complete training implementation depends on your specific environment class")
        print(f"Results will be saved to: {results_dir}")
        
        # Placeholder for actual training loop
        # See the original notebook for the complete implementation
        
    print("Training completed!")


def main():
    parser = argparse.ArgumentParser(description='Train offline RL policies on Synapse benchmark')
    
    # Configuration file
    parser.add_argument('--config', type=str, default='configs/config_template.yaml',
                        help='Path to configuration YAML file')
    
    # Override options
    parser.add_argument('--env_name', type=str, default=None,
                        help='Environment name (e.g., RZCH_clinical)')
    parser.add_argument('--raw_data_path', type=str, default=None,
                        help='Path to raw data .npy file')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Directory to save checkpoints')
    parser.add_argument('--results_dir', type=str, default=None,
                        help='Directory to save results')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed')
    parser.add_argument('--max_episodes', type=int, default=None,
                        help='Maximum number of episodes')
    parser.add_argument('--beta', type=float, default=None,
                        help='Beta hyperparameter')
    
    args = parser.parse_args()
    
    train_policy(args)


if __name__ == '__main__':
    main()
