# RLE Mini-Challenge

[English](README.md) | **Deutsch**

Ziel dieser Mini-Challenge ist es, einen Deep Reinforcement Learning Agenten zu trainieren, der einen möglichst hohen Score im Atari Spiel "Space Invaders" erreicht.

In diesem Repository findet ihr ein Template, auf dem ihr eure Lösung implementieren könnt, sowie Beispiel-Implementationen von DQN und PPO Agenten.

## Atari Space Invaders Environment

![](https://ale.farama.org/_images/space_invaders.gif)


Gymnasium Dokumentation: [https://gymnasium.farama.org/](https://gymnasium.farama.org/)

Space Invaders Dokumentation: [https://ale.farama.org/environments/space_invaders/](https://ale.farama.org/environments/space_invaders/)

### Offizielles Environment und Evaluation

Alle Agenten werden auf demselben Environment trainiert und evaluiert, definiert in `utils/env.py` (`make_env`). Diese Datei darf nicht verändert werden.

- `ALE/SpaceInvaders-v5` mit `frameskip=1`: jeder Schritt bewegt das Spiel um genau einen Frame weiter. Frame Skipping ist Teil der Vorverarbeitung eures Agenten (z.B. `gymnasium.wrappers.AtariPreprocessing(env, frame_skip=4)`).
- Sticky Actions (`repeat_action_probability=0.25`): mit einer Wahrscheinlichkeit von 25% wird die vorherige Aktion anstelle der gewählten wiederholt.
- Eine Episode ist ein ganzes Spiel (alle Leben) und dauert höchstens 108'000 Frames (30 Minuten Spielzeit).

Eure Abgabe wird mit `utils.env.evaluate` bewertet: der mittlere Score (ungeclippte Spielpunkte) über 100 Episoden mit Seed 1234. Ihr übergebt eine Policy (Observation → Aktion) und die Wrapper zur Vorverarbeitung, mit denen euer Agent trainiert wurde. Wrapper, die eine Episode frühzeitig beenden (z.B. bei Verlust eines Lebens), sind bei der Evaluation nicht erlaubt. Alle Beispiel-Skripte verwenden diese Funktion für ihre Evaluation.

```python
import utils.env

scores = utils.env.evaluate(policy, wrap_env=preprocess)
```


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

Ein passendes virtual environment wird automatisch erstellt und aktiviert, sobald Sie `uv run python <script>.py` ausführen. Das Projekt unterstützt Python 3.11 bis 3.14 und verwendet standardmässig Python 3.13 (siehe `.python-version`). UV lädt es bei Bedarf automatisch herunter.

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

Nach dem Training evaluiert jedes Skript den Agenten mit der offiziellen Evaluation (siehe oben).

### Parameters:

#### DQN Clean RL Parameters:
- `--exp_name`: Experiment name (default: 'dqn_clean_rl')
- `--seed`: Random seed (default: 1)
- `--cuda`: Enable CUDA (default: True)
- `--capture_video`: Record videos to the `videos` folder (default: True)
- `--total_timesteps`: Total timesteps for training (default: 10000000)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--buffer_size`: Replay buffer size (default: 100000, benötigt ~2.8 GB RAM)
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

Boolesche Parameter der CleanRL-Skripte werden mit dem Präfix `--no-` ausgeschaltet, z.B. `--no-cuda` oder `--no-capture-video`.

#### DQN Yanick Parameters:
Alle Schrittzahlen sind Environment-Schritte, summiert über alle parallelen Environments.
- `--mode`: 'train' or 'eval' mode (default: 'train')
- `--logdir`: Directory for logs (default: './runs')
- `--run_name`: Run name (default: current datetime)
- `--cuda`: Use CUDA if available (default: True, mit `--nocuda` ausschalten)
- `--seed`: Random seed (default: 42)
- `--gamma`: Discount factor (default: 0.99)
- `--batch_size`: Batch size (default: 32)
- `--learning_rate`: Learning rate (default: 2.5e-4)
- `--num_envs`: Number of parallel environments (default: 16)
- `--total_steps`: Total environment steps (default: 10000000)
- `--warmup_steps`: Steps to fill replay buffer before training (default: 80000)
- `--buffer_size`: Replay buffer size (default: 100000, benötigt ~1.4 GB RAM)
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

### Beispiel: Parameter setzen

Um Parameter beim Ausführen von Skripten mit `uv` zu ändern, fügen Sie sie einfach nach dem Skriptnamen an. Zum Beispiel, um weniger Zeitschritte zu trainieren:

```
# DQN Clean RL mit 1 Millionen Zeitschritte statt 10 Millionen
uv run python dqn_clean_rl.py --total_timesteps 1000000

# PPO Clean RL mit 2 Millionen Zeitschritte
uv run python ppo_clean_rl.py --total_timesteps 2000000

# DQN Yanick mit 500,000 Zeitschritte (verwendet --total_steps statt --total_timesteps)
uv run python dqn_example.py --total_steps 500000
```

Sie können mehrere Parameter in derselben Befehlszeile kombinieren:

```
uv run python dqn_clean_rl.py --total_timesteps 1000000 --learning_rate 5e-4 --seed 42
```

### Weights & Biases (wandb) Tracking

Die CleanRL-Implementierungen (`ppo_clean_rl.py` und `dqn_clean_rl.py`) unterstützen das Tracking mit Weights & Biases (wandb), einem Tool für ML Experiment Tracking.

#### Einrichtung von wandb

1. **Konto erstellen**: Besuchen Sie [wandb.ai](https://wandb.ai) und erstellen Sie ein kostenloses Konto.

2. **API-Schlüssel kopieren**: Nach der Anmeldung können Sie Ihren API-Schlüssel unter [wandb.ai/authorize](https://wandb.ai/authorize) finden.

3. **wandb installieren und anmelden**:
```
uv run wandb login
```
Bei der Anmeldung werden Sie aufgefordert, Ihren API-Schlüssel einzugeben.

#### Aktivieren des wandb-Trackings

Um wandb-Tracking mit den CleanRL-Implementierungen zu aktivieren:

```
# DQN mit wandb-Tracking
uv run python dqn_clean_rl.py --track --wandb_project_name "space-invaders-dqn"

# PPO mit wandb-Tracking und eigenem Team/Entity
uv run python ppo_clean_rl.py --track --wandb_project_name "space-invaders-ppo" --wandb_entity "mein-team"
```

#### Verfügbare wandb-Parameter:

- `--track`: Aktiviert das wandb-Tracking (default: False)
- `--wandb_project_name`: Name des wandb-Projekts (default: "cleanRL")
- `--wandb_entity`: Entity/Team für das wandb-Projekt (optional)

Mit wandb können Sie:
- Trainingsmetriken in Echtzeit verfolgen
- Hyperparameter und deren Auswirkungen visualisieren
- Experimente vergleichen und teilen
- Modellleistung über Zeit analysieren

Nach dem Training können Sie Ihre Ergebnisse unter [wandb.ai/home](https://wandb.ai/home) einsehen.

## Evaluation

Einen gespeicherten Checkpoint ohne Training evaluieren:

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

## Projektstruktur

### pyproject.toml

Diese Datei definiert alle Projektabhängigkeiten und die unterstützten Python-Versionen (3.11 bis 3.14). Sie enthält auch spezielle Konfigurationen für PyTorch mit CUDA-Unterstützung auf Windows und Linux.

### run.py

Template für das Implementieren der Lösung. Es läuft unverändert mit einem zufälligen Agenten, die `TODO` Kommentare markieren, wo euer Agent hingehört.

Es steht euch jedoch offen, ob ihr dieses Template verwendet oder einen eigenen Ansatz verfolgt. Evaluiert euren Agenten in jedem Fall mit `utils.env.evaluate`.

### dqn_example.py

Beispiel-Implementation eines einfachen DQN Agenten. Kann für die Mini-Challenge verwendet und erweitert werden.

### dqn_clean_rl.py

Beispiel-Implementation DQN von [CleanRL](https://docs.cleanrl.dev/). Kann für die Mini-Challenge verwendet und erweitert werden.

### ppo_clean_rl.py

Beispiel-Implementation PPO von [CleanRL](https://docs.cleanrl.dev/). Kann für die Mini-Challenge verwendet und erweitert werden.


### utils/env.py

Beinhaltet das offizielle Environment (`make_env`) und die Evaluation (`evaluate`).

### utils/utils.py

Beinhaltet nützliche Funktionen und Klassen.

## Fehlerbehebung

### PyTorch CUDA-Probleme

Falls Probleme mit der CUDA-Unterstützung auftreten, überprüfen Sie:

1. Ob eine kompatible NVIDIA-Grafikkarte installiert ist
2. Ob die neuesten NVIDIA-Treiber installiert sind (PyTorch wird mit CUDA 12.8 installiert und benötigt Treiberversion 570 oder neuer)

Mit dem folgenden Befehl können Sie testen, ob PyTorch CUDA erkennt:

```
uv run python -c "import torch; print(torch.cuda.is_available())"
```

Auf macOS läuft PyTorch auf der CPU.

### Atari ROMs

Die Atari ROMs sind im Paket `ale-py` enthalten, ein separater Download oder eine Lizenzbestätigung ist nicht nötig.
