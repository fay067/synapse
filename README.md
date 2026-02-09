# SYNAPSE: Simulation Benchmark of Neuro-Adaptive Patient-Specific Evaluation for Episodic Decision-Making

[![Paper]
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

Synapse is a virtual environment benchmark designed for offline reinforcement learning, off-policy evaluation, and policy learning research. This repository currently provides tools to:

1. **Roll out trajectories** from trained policies for downstream tasks (prediction, off-policy evaluation, offline policy learning, transfer learning, etc.)
2. **Learn and evaluate** offline/online RL policies with real-time performance monitoring

The benchmark includes five virtual cohorts with different characteristics, enabling comprehensive evaluation of RL algorithms in clinical or similar sequential decision-making settings.

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Data Preparation](#data-preparation)
- [Usage](#usage)
  - [Training Policies](#training-policies)
  - [Rolling Out Trajectories](#rolling-out-trajectories)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Citation](#citation)
- [License](#license)

## 🔧 Installation

### Prerequisites

- Python 3.7 or 3.8
- CUDA-compatible GPU (recommended)
- CUDA 11.0+ and cuDNN (for TensorFlow GPU support)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/[YOUR-USERNAME]/synapse-benchmark.git
cd synapse-benchmark
```

2. **Create a virtual environment**
```bash
conda create -n synapse python=3.8
conda activate synapse
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Install the package**
```bash
pip install -e .
```

## 🚀 Quick Start

### 1. Download Data and Checkpoints

**Data Access for Academic Research**

The Synapse benchmark data is available for academic and research purposes through a request and approval process.

**To request checkpoint access:**

1. Download and complete the [Data Access Request Form](DATA_ACCESS_REQUEST.md)
2. Include your information:
   - Full name
   - Institution and department
   - Supervisor's name and contact (required for students/postdocs)
   - Research purpose
3. Email the completed form to: ge.gao.hai@gmail.com
4. You will receive download links within 5-7 business days upon approval

**Important:** 
- Data is for academic/research use only
- Do not redistribute data files
- Properly cite the Synapse benchmark in publications

**Checkpoint files:** Pre-trained checkpoints are available through the same request process (optional).

### 2. Run a Quick Test

Train a CQL policy on a virtual cohort:
```bash
python scripts/train_policy.py \
    --config configs/config_template.yaml \
    --env_name RZCH_clinical \
    --max_episodes 50
```

## 📊 Data Preparation

### Required Data Files

**TODO: Please specify the exact data files needed:**

1. **Raw trajectory data**: `data/raw/d4rl_typed_data_splited_into_trajectories_{ENV_NAME}.npy`
   - Format: [Please describe the data format]
   - Expected keys in the dictionary: [List the keys]
   - Data source: [Where to obtain this data]

2. **Pre-trained checkpoints** (optional):
   - Location: `checkpoints/{ENV_NAME}/`
   - Format: TensorFlow checkpoint files (`.ckpt`, `.meta`, `.index`)

3. **Target policies** (for rollout):
   - Location: `checkpoints/target_policies/{ENV_NAME}/`
   - Available policy parameters: [List available parameters]

### Data Format

The trajectory data should be stored as a NumPy `.npy` file containing a dictionary with the following structure:

```python
{
    'observations': np.array,  # Shape: [N, obs_dim]
    'next_observations': np.array,  # Shape: [N, obs_dim]
    'actions': np.array,  # Shape: [N, action_dim]
    'rewards': np.array,  # Shape: [N,]
    'terminals': np.array,  # Shape: [N,] (boolean or 0/1)
    # ... other keys as needed
}
```

## 📖 Usage

### Training Policies

Train an offline RL policy using the command-line interface:

```bash
python scripts/train_policy.py \
    --config configs/config_template.yaml \
    --env_name RZCH_clinical \
    --raw_data_path data/raw/your_data_file.npy \
    --checkpoint_dir checkpoints/RZCH_clinical \
    --results_dir results \
    --seed 2599 \
    --max_episodes 300 \
    --beta 0.1
```

#### Arguments

- `--config`: Path to YAML configuration file (default: `configs/config_template.yaml`)
- `--env_name`: Environment/cohort name (options: `RZCH_clinical`, `E395_clinical`, etc.)
- `--raw_data_path`: Path to raw trajectory data `.npy` file
- `--checkpoint_dir`: Directory to save model checkpoints
- `--results_dir`: Directory to save training results
- `--seed`: Random seed for reproducibility
- `--max_episodes`: Maximum number of training episodes
- `--beta`: Beta hyperparameter for the algorithm

### Rolling Out Trajectories

Generate trajectories from trained policies:

```bash
python scripts/rollout_trajectories.py \
    --config configs/config_template.yaml \
    --env_name RZCH_clinical \
    --checkpoint_path checkpoints/RZCH_clinical/your_checkpoint \
    --target_policy_dir checkpoints/target_policies/RZCH_clinical \
    --output_dir outputs \
    --policy_params param3 param4
```

Or automatically find and use the best checkpoint:

```bash
python scripts/rollout_trajectories.py \
    --config configs/config_template.yaml \
    --env_name RZCH_clinical \
    --auto_find_best \
    --target_policy_dir checkpoints/target_policies/RZCH_clinical \
    --output_dir outputs
```

#### Arguments

- `--config`: Path to YAML configuration file
- `--env_name`: Environment/cohort name
- `--checkpoint_path`: Specific checkpoint to use for rollout
- `--auto_find_best`: Automatically find the best checkpoint based on MAE and RANK metrics
- `--target_policy_dir`: Directory containing target policies to evaluate
- `--output_dir`: Directory to save generated trajectories
- `--policy_params`: List of policy parameters to evaluate (e.g., `param3 param4`)

## ⚙️ Configuration

### YAML Configuration File

The `configs/config_template.yaml` file contains all hyperparameters and paths. Key sections:

```yaml
environment:
  name: "RZCH_clinical"  # Environment name
  window_size: 10  # Observation window
  
data:
  raw_data_path: "data/raw/..."  # Data file path template
  checkpoint_dir: "checkpoints/{ENV_NAME}"  # Checkpoint directory
  
training:
  gamma: 0.995  # Discount factor
  max_episodes: 300  # Training episodes
  random_seed: 2599  # Random seed
  
policy_learning:
  algorithm: "CQL"  # RL algorithm
  batch_size: 256
  # ... other hyperparameters
```

You can override any configuration parameter using command-line arguments.

## 📁 Project Structure

```
synapse-benchmark/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── setup.py                  # Package setup
├── .gitignore               # Git ignore rules
│
├── configs/                  # Configuration files
│   └── config_template.yaml  # Default configuration template
│
├── src/                      # Source code
│   ├── core.py              # Core RL components (policies, critics)
│   ├── utils.py             # Utility functions
│   ├── vlm.py               # VLM-related utilities
│   └── utils_latentPolicy_sac_lstm_zt_zt1*.py  # Policy utilities
│
├── scripts/                  # Executable scripts
│   ├── train_policy.py      # Policy training script
│   └── rollout_trajectories.py  # Trajectory rollout script
│
├── notebooks/               # Jupyter notebooks (original versions)
│   ├── POLICY_LEARNING_*.ipynb
│   └── ROLLOUT_TRAJ_*.ipynb
│
├── data/                    # Data directory
│   ├── raw/                # Raw trajectory data (.npy files)
│   └── processed/          # Processed data
│
├── checkpoints/            # Model checkpoints
│   ├── RZCH_clinical/     # RZCH cohort checkpoints
│   ├── target_policies/   # Target policies for evaluation
│   └── ...                # Other cohort checkpoints
│
├── results/               # Training results and logs
└── outputs/              # Generated trajectories and outputs
```

## 🏥 Available Virtual Cohorts

The benchmark includes five virtual cohorts

**TODO: Please provide brief descriptions of each cohort's characteristics**

## 🔬 Supported RL Algorithms

Currently supported:
- **CQL** (Conservative Q-Learning)
- **SAC** (Soft Actor-Critic)
- **PPO** (Proximal Policy Optimization)
- **A2C** (Advantage Actor-Critic) 
- **DDPG** 

The codebase is designed with Gym API compatibility, making it easy to integrate additional RL algorithms. 

## 📝 Examples

### Example 1: Train CQL on RZCH virtual cohort with custom hyperparameters

```bash
python scripts/train_policy.py \
    --config configs/config_template.yaml \
    --env_name RZCH_clinical \
    --max_episodes 500 \
    --beta 0.2 \
    --seed 42
```

### Example 2: Rollout trajectories for multiple policy parameters

```bash
python scripts/rollout_trajectories.py \
    --config configs/config_template.yaml \
    --env_name RZCH_clinical \
    --auto_find_best \
    --policy_params param3_distilled param3 param2_distilled param4
```

### Example 3: Train on a different cohort

```bash
python scripts/train_policy.py \
    --config configs/config_template.yaml \
    --env_name E395_clinical \
    --raw_data_path data/raw/d4rl_typed_data_splited_into_trajectories_E395_clinical.npy
```

## 📄 License

This code is licensed under the MIT License - see the LICENSE file for details.

## 📚 Citation

If you use this benchmark in your research, please cite:

```bibtex
@inproceedings{synapse2025,
  title={SYNAPSE: Simulation Benchmark of Neuro-Adaptive Patient-Specific Evaluation for Episodic Decision-Making},
  author={[Ge Gao, Qitong Gao, Miroslav Pajic, Emma Brunskill]},
  booktitle={[TODO: Add venue]},
  year={2025}
}
```

##  Acknowledgments

PLACEHOLDER

