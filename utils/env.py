"""Official Space Invaders environment and evaluation protocol of the mini-challenge.

Every submission is scored with `evaluate` below. Do not change this file: build your own
preprocessing (frame skipping, resizing, frame stacking, reward clipping, ...) as wrappers
on top of `make_env` and pass it to `evaluate` via `wrap_env`.
"""
import random
from typing import Callable, Optional

import ale_py
import gymnasium as gym
import numpy as np
import torch

gym.register_envs(ale_py)

ENV_ID = "ALE/SpaceInvaders-v5"

# Official evaluation protocol
EVAL_EPISODES = 100
EVAL_SEED = 1234


def make_env(render_mode: Optional[str] = None, video_folder: Optional[str] = None,
             episode_trigger: Optional[Callable[[int], bool]] = None) -> gym.Env:
    """Creates the official base environment.

    - Every step advances the emulator by exactly one frame (`frameskip=1`), so frame skipping
      is part of your agent's preprocessing (e.g. `AtariPreprocessing(frame_skip=4)`).
    - Sticky actions (`repeat_action_probability=0.25`) and a limit of 108'000 frames
      (30 minutes of play) per episode, as recommended by Machado et al. (2018).
    - `RecordEpisodeStatistics` sits directly on the base env, so `info["episode"]` always
      contains the unclipped score of a full game (all lives), regardless of your wrappers.
    """
    if video_folder is not None:
        render_mode = "rgb_array"
    env = gym.make(ENV_ID, frameskip=1, repeat_action_probability=0.25, render_mode=render_mode)
    if video_folder is not None:
        env = gym.wrappers.RecordVideo(env, video_folder, episode_trigger=episode_trigger, disable_logger=True)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env


def evaluate(policy: Callable[[np.ndarray], int], wrap_env: Callable[[gym.Env], gym.Env] = lambda env: env,
             num_episodes: int = EVAL_EPISODES, seed: int = EVAL_SEED, render_mode: Optional[str] = None,
             video_folder: Optional[str] = None) -> np.ndarray:
    """Runs the official evaluation and returns the score (unclipped game return) of every episode.

    Args:
        policy: maps a single (preprocessed) observation to an action.
        wrap_env: applies the same preprocessing wrappers your agent was trained with. Do not use
            wrappers that end an episode early (e.g. on life loss) - a game lasts until all lives are lost.
        num_episodes: number of episodes; grading uses `EVAL_EPISODES`.
        seed: seed of the evaluation episodes; grading uses `EVAL_SEED`.
    """
    # seed the env and all sources of randomness a policy might use, so evaluations are reproducible
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = wrap_env(make_env(render_mode, video_folder))
    env.action_space.seed(seed)
    scores = []
    obs, _ = env.reset(seed=seed)
    while len(scores) < num_episodes:
        obs, _, terminated, truncated, info = env.step(policy(obs))
        if terminated or truncated:
            if "episode" not in info:
                raise RuntimeError("A wrapper ended the episode before the game was over "
                                   "(e.g. on life loss). Remove it from `wrap_env` for evaluation.")
            scores.append(float(info["episode"]["r"]))
            print(f"eval_episode={len(scores) - 1}, score={scores[-1]:.0f}, frames={info['episode']['l']}")
            obs, _ = env.reset()
    env.close()
    scores = np.array(scores)
    print(f"Evaluation over {num_episodes} episodes: mean_score={scores.mean():.2f}, std={scores.std():.2f}, "
          f"min={scores.min():.0f}, max={scores.max():.0f}")
    return scores
