import datetime
import os
from typing import Callable, Optional, Literal

import gymnasium as gym
from absl import flags, app

import utils.env

# common flags
flags.DEFINE_enum('mode', 'eval', ['train', 'eval'], 'Run mode.')
flags.DEFINE_string('logdir', './runs', 'Directory where all outputs are written to.')
flags.DEFINE_string('run_name', datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), 'Run name.')

# eval flags
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

    observation, info = env.reset(seed=42)
    for _ in range(1000):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()
    env.close()

def main(_):
    # if FLAGS.mode == 'train':
    #     train()
    # elif FLAGS.mode == 'eval':
    eval()


if __name__ == "__main__":
    app.run(main)