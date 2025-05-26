import datetime
import os
import time
from typing import Callable, Optional, Literal

import gymnasium as gym
import numpy as np
from absl import flags, app

import utils.env

# common flags
flags.DEFINE_string('logdir', './runs', 'Directory where all outputs are written to.')
flags.DEFINE_string('run_name', datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), 'Run name.')

# eval flags
flags.DEFINE_integer('eval_num_episodes', 30, 'Number of eval episodes.')
flags.DEFINE_enum('eval_render_mode', 'rgb_array', ['human', 'rgb_array', 'None'], 'Render env during eval.')
flags.DEFINE_integer('eval_seed', 1234, 'Eval seed.')

FLAGS = flags.FLAGS

def make_env_fn(seed: int, render_mode: Literal['human', 'rgb_array', None] = 'human', video_folder: Optional[str] = None) -> Callable[[], gym.Env]:
    """ returns a pickleable callable to create an env instance """
    def env_fn():
        env = utils.env.make_env(render_mode, video_folder)
        return env
    return env_fn


def evaluate():
    logdir = os.path.join(FLAGS.logdir, FLAGS.run_name)
    env_fn = make_env_fn(FLAGS.eval_seed, FLAGS.eval_render_mode, video_folder=os.path.join(logdir, 'videos', 'train'))
    env = env_fn()

    episode_rewards = []
    episode_scores = []
    episode_runtimes = []
    episode_accuracies = []
    episode_survival_times = []
    episode_time_to_first_death = []
    episode_exploration_coverage = []

    for episode_idx in range(FLAGS.eval_num_episodes):
        observation, info = env.reset(seed=42)
        terminated = False
        step = 0
        start_time = time.time()
        total_actions = 0
        successful_actions = 0  # Define what a "successful" action is for your env
        first_death_time = None
        unique_states = set()
        lives = info.get('lives', None)
        initial_lives = lives
        while not terminated:
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            step += 1
            total_actions += 1
            if reward > 0:
                successful_actions += 1
            unique_states.add(observation.tobytes())
            if lives is not None and info.get('lives', lives) < lives:
                if first_death_time is None:
                    first_death_time = time.time() - start_time
                lives = info.get('lives', lives)
            end_time = time.time()
            if terminated or truncated:
                print(f"Episode {episode_idx}: "
                      f"reward={info['episode']['r']}, "
                      f"steps={step}, "
                      f"frames={info['episode_frame_number']}")
                episode_rewards.append(info['episode']['r'])
                episode_scores.append(info['episode']['r'])
                episode_runtimes.append(end_time - start_time)
                episode_accuracies.append(successful_actions / total_actions if total_actions > 0 else 0)
                episode_survival_times.append(step)
                episode_time_to_first_death.append(
                    first_death_time if first_death_time is not None else end_time - start_time)
                episode_exploration_coverage.append(len(unique_states))
                observation, info = env.reset()
        print(f"Evaluation completed: "
              f"mean_episode_reward={np.mean(episode_rewards):.2f}, "
              f"std_episode_reward={np.std(episode_rewards):.2f}, "
              f"min_episode_reward={np.min(episode_rewards):.2f}, "
              f"max_episode_reward={np.max(episode_rewards):.2f}")
        print(f"Mean Score: {np.mean(episode_scores):.2f}")
        print(f"Mean Runtime (s): {np.mean(episode_runtimes):.2f}")
        print(f"Mean Accuracy: {np.mean(episode_accuracies):.2f}")
        print(f"Mean Survival Time (steps): {np.mean(episode_survival_times):.2f}")
        print(f"Mean Time to First Death (s): {np.mean(episode_time_to_first_death):.2f}")
        print(f"Mean Exploration Coverage (unique states): {np.mean(episode_exploration_coverage):.2f}")
        env.close()

def main(_):
    evaluate()

if __name__ == "__main__":
    app.run(main)