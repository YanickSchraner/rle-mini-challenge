# RLE Mini-Challenge

**English** | [Deutsch](README.de.md)

The goal of this mini-challenge is to train a deep reinforcement learning agent that achieves the highest possible score in the Atari game "Space Invaders".

This repository contains a template on which you can implement your solution, as well as example implementations of DQN and PPO agents.

## Atari Space Invaders Environment

![](https://ale.farama.org/_images/space_invaders.gif)


Gymnasium documentation: [https://gymnasium.farama.org/](https://gymnasium.farama.org/)

Space Invaders documentation: [https://ale.farama.org/environments/space_invaders/](https://ale.farama.org/environments/space_invaders/)

### Official environment and evaluation

All agents are trained and evaluated on the same environment, defined in `utils/env.py` (`make_env`). Do not change this file.

- `ALE/SpaceInvaders-v5` with `frameskip=1`: every step advances the game by exactly one frame. Frame skipping is part of your agent's preprocessing (e.g. `gymnasium.wrappers.AtariPreprocessing(env, frame_skip=4)`).
- Sticky actions (`repeat_action_probability=0.25`): with a probability of 25% the previous action is repeated instead of the selected one.
- An episode is a full game (all lives) and lasts at most 108'000 frames (30 minutes of play).

Your submission is scored with `utils.env.evaluate`: the mean score (unclipped game points) over 100 episodes with seed 1234. You provide a policy (observation → action) and the preprocessing wrappers your agent was trained with. Wrappers that end an episode early (e.g. on life loss) are not allowed during evaluation. All example scripts use this function for their evaluation.

```python
import utils.env

scores = utils.env.evaluate(policy, wrap_env=preprocess)
```


## Installation

This project uses [Astral UV](https://docs.astral.sh/uv/) as its package and environment manager, which is considerably faster than pip and other tools.

### 1. Install Astral UV

UV is a modern Python package manager that replaces pip, venv, poetry and other tools. It is 10-100x faster than traditional solutions.

For Windows:
```
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

For macOS and Linux:
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone the project repository

```
git clone https://github.com/YanickSchraner/rle-mini-challenge.git
cd rle-mini-challenge
```

A matching virtual environment is created and activated automatically as soon as you run `uv run python <script>.py`. The project supports Python 3.11 to 3.14 and uses Python 3.13 by default (see `.python-version`). UV downloads it automatically if needed.

## Tensorboard
Individual experiments can be logged to Tensorboard.
To visualize them:
```
uv run tensorboard --logdir runs
```

## Start training:

PPO Clean RL
```
uv run python ppo_clean_rl.py
```

DQN Clean RL
```
uv run python dqn_clean_rl.py
```

DQN Yanick
```
uv run python dqn_example.py
```

After training, every script evaluates the agent with the official evaluation (see above).

### Parameters:

#### DQN Clean RL Parameters:
- `--exp_name`: Experiment name (default: 'dqn_clean_rl')
- `--seed`: Random seed (default: 1)
- `--cuda`: Enable CUDA (default: True)
- `--capture_video`: Record videos to the `videos` folder (default: True)
- `--total_timesteps`: Total timesteps for training (default: 10000000)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--buffer_size`: Replay buffer size (default: 100000, needs ~2.8 GB of RAM)
- `--gamma`: Discount factor (default: 0.99)
- `--batch_size`: Batch size (default: 32)
- `--start_e`: Initial exploration epsilon (default: 1)
- `--end_e`: Final exploration epsilon (default: 0.01)
- `--exploration_fraction`: Fraction of timesteps for epsilon decay (default: 0.10)
- `--learning_starts`: Timesteps before learning starts (default: 80000)
- `--train_frequency`: Training frequency (default: 4)
- `--eval_checkpoint`: Path to checkpoint for evaluation only
- `--eval_episodes`: Number of evaluation episodes (default: 100)
- `--eval_epsilon`: Exploration epsilon during evaluation (default: 0.05)

#### PPO Clean RL Parameters:
- `--exp_name`: Experiment name (default: 'ppo_clean_rl')
- `--seed`: Random seed (default: 1)
- `--cuda`: Enable CUDA (default: True)
- `--capture_video`: Record videos to the `videos` folder (default: True)
- `--total_timesteps`: Total timesteps for training (default: 10000000)
- `--learning_rate`: Learning rate (default: 2.5e-4)
- `--num_envs`: Number of parallel environments (default: 16)
- `--num_steps`: Steps per environment per rollout (default: 128)
- `--anneal_lr`: Enable learning rate annealing (default: True)
- `--gamma`: Discount factor (default: 0.99)
- `--gae_lambda`: GAE lambda parameter (default: 0.95)
- `--num_minibatches`: Number of minibatches (default: 4)
- `--update_epochs`: Number of update epochs (default: 4)
- `--clip_coef`: PPO clip coefficient (default: 0.1)
- `--ent_coef`: Entropy coefficient (default: 0.01)
- `--vf_coef`: Value function coefficient (default: 0.5)
- `--eval_checkpoint`: Path to checkpoint for evaluation only
- `--eval_episodes`: Number of evaluation episodes (default: 100)

Boolean parameters of the CleanRL scripts are switched off with a `--no-` prefix, e.g. `--no-cuda` or `--no-capture-video`.

#### DQN Yanick Parameters:
All step counts are environment steps, summed over all parallel environments.
- `--mode`: 'train' or 'eval' mode (default: 'train')
- `--logdir`: Directory for logs (default: './runs')
- `--run_name`: Run name (default: current datetime)
- `--cuda`: Use CUDA if available (default: True, disable with `--nocuda`)
- `--seed`: Random seed (default: 42)
- `--gamma`: Discount factor (default: 0.99)
- `--batch_size`: Batch size (default: 32)
- `--learning_rate`: Learning rate (default: 2.5e-4)
- `--num_envs`: Number of parallel environments (default: 16)
- `--total_steps`: Total environment steps (default: 10000000)
- `--warmup_steps`: Steps to fill replay buffer before training (default: 80000)
- `--buffer_size`: Replay buffer size (default: 100000, needs ~1.4 GB of RAM)
- `--exploration_epsilon_initial`: Initial exploration epsilon (default: 1.0)
- `--exploration_epsilon_final`: Final exploration epsilon (default: 0.1)
- `--exploration_fraction`: Fraction of training for epsilon decay (default: 0.1)
- `--train_freq`: One gradient step every `train_freq` environment steps (default: 4)
- `--target_network_update_freq`: Target network update frequency (default: 1000)
- `--checkpoint_freq`: Checkpoint frequency (default: 1000000)
- `--eval_checkpoint`: Path to checkpoint for evaluation mode
- `--eval_num_episodes`: Number of evaluation episodes (default: 100)
- `--eval_epsilon`: Exploration epsilon during evaluation (default: 0.05)
- `--eval_render`: Show the game in a window during evaluation (default: False)

### Example: setting parameters

To change parameters when running scripts with `uv`, simply append them after the script name. For example, to train for fewer timesteps:

```
# DQN Clean RL with 1 million timesteps instead of 10 million
uv run python dqn_clean_rl.py --total_timesteps 1000000

# PPO Clean RL with 2 million timesteps
uv run python ppo_clean_rl.py --total_timesteps 2000000

# DQN Yanick with 500,000 timesteps (uses --total_steps instead of --total_timesteps)
uv run python dqn_example.py --total_steps 500000
```

You can combine multiple parameters in the same command:

```
uv run python dqn_clean_rl.py --total_timesteps 1000000 --learning_rate 5e-4 --seed 42
```

### Weights & Biases (wandb) Tracking

The CleanRL implementations (`ppo_clean_rl.py` and `dqn_clean_rl.py`) support tracking with Weights & Biases (wandb), a tool for ML experiment tracking.

#### Setting up wandb

1. **Create an account**: Visit [wandb.ai](https://wandb.ai) and create a free account.

2. **Copy your API key**: After signing in, you can find your API key at [wandb.ai/authorize](https://wandb.ai/authorize).

3. **Install wandb and log in**:
```
uv run wandb login
```
During login you will be prompted to enter your API key.

#### Enabling wandb tracking

To enable wandb tracking with the CleanRL implementations:

```
# DQN with wandb tracking
uv run python dqn_clean_rl.py --track --wandb_project_name "space-invaders-dqn"

# PPO with wandb tracking and your own team/entity
uv run python ppo_clean_rl.py --track --wandb_project_name "space-invaders-ppo" --wandb_entity "my-team"
```

#### Available wandb parameters:

- `--track`: Enables wandb tracking (default: False)
- `--wandb_project_name`: Name of the wandb project (default: "cleanRL")
- `--wandb_entity`: Entity/team for the wandb project (optional)

With wandb you can:
- Track training metrics in real time
- Visualize hyperparameters and their effects
- Compare and share experiments
- Analyze model performance over time

After training, you can view your results at [wandb.ai/home](https://wandb.ai/home).

## Evaluation

Evaluate a saved checkpoint without training:

PPO Clean RL:
```
uv run python ppo_clean_rl.py --eval_checkpoint "runs/{run_name}/ppo_clean_rl.cleanrl_model"
```

DQN Clean RL:
```
uv run python dqn_clean_rl.py --eval_checkpoint "runs/{run_name}/dqn_clean_rl.cleanrl_model"
```

DQN Yanick
```
uv run python dqn_example.py --mode eval --eval_checkpoint "runs/{run_name}/checkpoint-last.pt"
```

## Project structure

### pyproject.toml

This file defines all project dependencies and the supported Python versions (3.11 to 3.14). It also contains special configuration for PyTorch with CUDA support on Windows and Linux.

### run.py

Template for implementing your solution. It runs as is with a random agent, the `TODO` comments mark where your agent goes.

However, you are free to use this template or follow your own approach. Either way, evaluate your agent with `utils.env.evaluate`.

### dqn_example.py

Example implementation of a simple DQN agent. Can be used and extended for the mini-challenge.

### dqn_clean_rl.py

Example DQN implementation from [CleanRL](https://docs.cleanrl.dev/). Can be used and extended for the mini-challenge.

### ppo_clean_rl.py

Example PPO implementation from [CleanRL](https://docs.cleanrl.dev/). Can be used and extended for the mini-challenge.


### utils/env.py

Contains the official environment (`make_env`) and evaluation (`evaluate`).

### utils/utils.py

Contains useful functions and classes.

## Troubleshooting

### PyTorch CUDA issues

If you run into problems with CUDA support, check:

1. Whether a compatible NVIDIA graphics card is installed
2. Whether the latest NVIDIA drivers are installed (PyTorch is installed with CUDA 12.8 and needs driver version 570 or newer)

You can use the following command to test whether PyTorch detects CUDA:

```
uv run python -c "import torch; print(torch.cuda.is_available())"
```

On macOS, PyTorch runs on the CPU.

### Atari ROMs

The Atari ROMs are included in the `ale-py` package, no separate download or license step is needed.
