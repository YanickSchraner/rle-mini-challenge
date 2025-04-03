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

### run.sh

Script um `run.py` auf dem SLURM cluster auszuführen.

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
