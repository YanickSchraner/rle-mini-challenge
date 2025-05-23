import datetime
import os
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


def eval():
    logdir = os.path.join(FLAGS.logdir, FLAGS.run_name)
    env_fn = make_env_fn(FLAGS.eval_seed, FLAGS.eval_render_mode, video_folder=os.path.join(logdir, 'videos', 'train'))
    env = env_fn()

    episode_rewards = []

    for episode_idx in range(FLAGS.eval_num_episodes):
        observation, info = env.reset(seed=42)
        terminated = False
        step = 0
        while not terminated:
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            step += 1

            if terminated or truncated:
                print(f"Episode {episode_idx}: "
                      f"reward={info['episode']['r']}, "
                      f"steps={step}, "
                      f"frames={info['episode_frame_number']}")
                episode_rewards.append(info['episode']['r'])
                observation, info = env.reset()
        print(f"Evaluation completed: "
              f"mean_episode_reward={np.mean(episode_rewards):.2f}, "
              f"std_episode_reward={np.std(episode_rewards):.2f}, "
              f"min_episode_reward={np.min(episode_rewards):.2f}, "
              f"max_episode_reward={np.max(episode_rewards):.2f}")
        env.close()

def main(_):
    eval()

if __name__ == "__main__":
    app.run(main)