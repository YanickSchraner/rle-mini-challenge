# RLE Mini-Challenge

Ziel dieser Mini-Challenge ist es einen Deep Reinforcemen Learning Agenten zu trainieren, der einen möglichst hohen Score im Atari Spiel "Space Invaders" erreicht.

In diesem Repository findet ihr ein Template, auf dem ihr eure Lösung implementieren könnt, sowie eine Beispiel-Implementation eines einfachen DQN Agenten.

## Atari Space Invaders Environment

![](https://ale.farama.org/_images/space_invaders.gif)


Gym Dokumentation: [https://gymnasium.farama.org/](https://gymnasium.farama.org/)

Gym Space Invaders Dokumentation: [https://ale.farama.org/environments/space_invaders/](https://ale.farama.org/environments/space_invaders/)


## Installation

Dieses Projekt verwendet [Astral UV](https://docs.astral.sh/uv/) als Paket- und Umgebungsmanager, der wesentlich schneller als pip und andere Tools ist.

### 1. Astral UV installieren

UV ist ein moderner Python-Paketmanager, der pip, venv, poetry und andere Tools ersetzt. Er ist 10-100x schneller als traditionelle Lösungen.

Für Windows:
```
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Für macOS und Linux:
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Projekt-Repository klonen

```
git clone https://github.com/YanickSchraner/rle-mini-challenge.git
cd rle-mini-challenge
```

Ein passendes virtual environment wird automatisch erstellt und aktiviert sobald Sie `uv run python <script>.py` ausführen

## Tensorboard
Einzelne Experimente können in Tensorboard gelogged werden.
So können diese visualisiert werden:
```
uv run tensorboard --logdir runs
```

## Training starten:

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

### Parameters:

#### DQN Clean RL Parameters:
- `--exp_name`: Experiment name (default: 'dqn_clean_rl')
- `--seed`: Random seed (default: 1)
- `--cuda`: Enable CUDA (default: True)
- `--total_timesteps`: Total timesteps for training (default: 10000000)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--buffer_size`: Replay buffer size (default: 1000000)
- `--gamma`: Discount factor (default: 0.99)
- `--batch_size`: Batch size (default: 32)
- `--start_e`: Initial exploration epsilon (default: 1)
- `--end_e`: Final exploration epsilon (default: 0.01)
- `--exploration_fraction`: Fraction of timesteps for epsilon decay (default: 0.10)
- `--learning_starts`: Timesteps before learning starts (default: 80000)
- `--train_frequency`: Training frequency (default: 4)
- `--eval_checkpoint`: Path to checkpoint for evaluation only

#### PPO Clean RL Parameters:
- `--exp_name`: Experiment name (default: 'ppo_clean_rl')
- `--seed`: Random seed (default: 1)
- `--cuda`: Enable CUDA (default: True)
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

#### DQN Yanick Parameters:
- `--mode`: 'train' or 'eval' mode (default: 'train')
- `--logdir`: Directory for logs (default: './runs')
- `--run_name`: Run name (default: current datetime)
- `--cuda`: Enable CUDA (default: True)
- `--seed`: Random seed (default: 42)
- `--gamma`: Discount factor (default: 0.99)
- `--batch_size`: Batch size (default: 32)
- `--learning_rate`: Learning rate (default: 2.5e-4)
- `--num_envs`: Number of parallel environments (default: 16)
- `--total_steps`: Total training steps (default: 10000000)
- `--warmup_steps`: Steps to fill replay buffer before training (default: 80000)
- `--buffer_size`: Replay buffer size (default: 100000)
- `--exploration_epsilon_initial`: Initial exploration epsilon (default: 1.0)
- `--exploration_epsilon_final`: Final exploration epsilon (default: 0.1)
- `--exploration_fraction`: Fraction of training for epsilon decay (default: 0.1)
- `--train_freq`: Training frequency (default: 4)
- `--eval_checkpoint`: Path to checkpoint for evaluation mode

### Beispiel: Parameter setzen

Um Parameter beim Ausführen von Skripten mit `uv` zu ändern, fügen Sie sie einfach nach dem Skriptnamen an. Zum Beispiel, um weniger Zeitschritte zu trainieren:

```
# DQN Clean RL mit 1 Millionen Zeitschritte statt 10 Millionen
uv run python dqn_clean_rl.py --total_timesteps 1000000

# PPO Clean RL mit 2 Millionen Zeitschritte
uv run python ppo_clean_rl.py --total_timesteps 2000000 

# DQN Yanick mit 500,000 Zeitschritte (uses --total_steps instead of --total_timesteps)
uv run python dqn_example.py --total_steps 500000
```

Sie können mehrere Parameter in derselben Befehlszeile kombinieren:

```
uv run python dqn_clean_rl.py --total_timesteps 1000000 --learning_rate 5e-4 --seed 42
```

## Evaluation

PPO Clean RL:
```
uv run python ppo_clean_rl.py --eval-checkpoint "runs/{run_name}/{args.exp_name}.cleanrl_model"
```

DQN Clean RL:
```
uv run python dqn_clean_rl.py --eval-checkpoint "runs/{run_name}/{args.exp_name}.cleanrl_model"
```

DQN Yanick
```
uv run python dqn_example.py --eval_checkpoint PATH_TO_CHECKPOINT
```

## Projektstruktur

### pyproject.toml

Diese Datei definiert alle Projektabhängigkeiten und die Python-Version (3.10). Sie enthält auch spezielle Konfigurationen für PyTorch mit CUDA-Unterstützung auf Windows.

### run.py

Template für das Implementieren der Lösung.

Es steht euch jedoch offen, ob ihr dieses Template verwendet oder einen eigenen Ansatz verfolgt.

### dqn_example.py

Beispiel-Implementation eines einfachen DQN agent. Kann für die mini-challenge verwendet und erweitert werden.

### dqn_clen_rl.py

Beispiel-Implementation DQN von [CleanRL](https://docs.cleanrl.dev/). Kann für die mini-challenge verwendet und erweitert werden.

### ppo_clen_rl.py

Beispiel-Implementation PPO von [CleanRL](https://docs.cleanrl.dev/). Kann für die mini-challenge verwendet und erweitert werden.


### utils/env.py

Beinhaltet die `make_env` Funktion, zum erstellen einer Environment-Instanz.

### utils/utils.py

Beinhaltet nützliche Funktionen und Klassen.

## Fehlerbehebung

### PyTorch CUDA-Probleme

Falls Probleme mit der CUDA-Unterstützung auftreten, überprüfen Sie:

1. Ob eine kompatible NVIDIA-Grafikkarte installiert ist
2. Ob die neuesten NVIDIA-Treiber installiert sind
3. Ob CUDA Toolkit installiert ist

Mit dem folgenden Befehl können Sie testen, ob PyTorch CUDA erkennt:

```
uv run python -c "import torch; print(torch.cuda.is_available())"
```

### Atari ROM-Probleme

Wenn Fehler bezüglich Atari ROMs auftreten, stellen Sie sicher, dass Sie beim ersten Ausführen die Lizenzvereinbarung akzeptiert haben.
