import datetime
import os
import random
import time
from collections import defaultdict
from typing import Callable, Optional

import gymnasium as gym
import numpy as np
from absl import flags, app
from tensorboardX import SummaryWriter

import utils.env

# common flags
flags.DEFINE_enum('mode', 'train', ['train', 'eval'], 'Run mode.')
flags.DEFINE_string('logdir', './runs', 'Directory where all outputs are written to.')
flags.DEFINE_string('run_name', datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), 'Run name.')
flags.DEFINE_bool('cuda', True, 'Whether to run the model on gpu (if available) or on cpu.')
flags.DEFINE_integer('seed', 42, 'Random seed.')

# train flags (all step counts are environment steps, summed over all parallel envs)
# TODO: add your train parameters here
flags.DEFINE_integer('num_envs', 2, 'Number of parallel envs.')
flags.DEFINE_integer('total_steps', 1_000_000, 'Total number of environment steps.')
flags.DEFINE_integer('checkpoint_freq', 100_000, 'Frequency at which checkpoints are stored.')
flags.DEFINE_integer('logging_freq', 10_000, 'Frequency at which logs are written.')

# eval flags
# TODO: add your eval parameters here
flags.DEFINE_string('eval_checkpoint', None, 'Path to the checkpoint to evaluate.')
flags.DEFINE_integer('eval_num_episodes', utils.env.EVAL_EPISODES, 'Number of eval episodes.')
flags.DEFINE_bool('eval_render', False, 'Render env during eval.')
flags.DEFINE_integer('eval_seed', utils.env.EVAL_SEED, 'Eval seed.')


FLAGS = flags.FLAGS


def preprocess(env: gym.Env) -> gym.Env:
    """ agent specific preprocessing, used for training and evaluation """

    # TODO: maybe add other gym.wrappers, e.g.
    #  env = gym.wrappers.AtariPreprocessing(env, frame_skip=4)
    #  env = gym.wrappers.FrameStackObservation(env, 4)

    return env


def make_env_fn(video_folder: Optional[str] = None) -> Callable[[], gym.Env]:
    """ returns a pickleable callable to create an env instance """
    def env_fn():
        return preprocess(utils.env.make_env(video_folder=video_folder))
    return env_fn


def train():
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    logdir = os.path.join(FLAGS.logdir, FLAGS.run_name)
    os.makedirs(logdir, exist_ok=False)

    FLAGS.append_flags_into_file(os.path.join(logdir, 'flags.txt'))

    writer = SummaryWriter(os.path.join(logdir, 'logs'))
    writer.add_text("config", FLAGS.flags_into_string())

    # initialize environments
    # SAME_STEP: when an episode ends, the env is reset within the same step and the
    # terminal observation is returned in infos["final_obs"]
    envs = gym.vector.AsyncVectorEnv([
        make_env_fn(video_folder=os.path.join(logdir, 'videos', 'train') if i == 0 else None)
        for i in range(FLAGS.num_envs)], autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)
    envs.action_space.seed(FLAGS.seed)

    # TODO: initialize agent, algorithm, etc...

    logs = defaultdict(list)
    last_log_step = 0
    last_log_time = time.time()

    def crossed(step: int, freq: int) -> int:
        """ number of multiples of freq in (step - num_envs, step] """
        return step // freq - (step - FLAGS.num_envs) // freq

    # every env gets its own seed, otherwise all envs would play the same game
    obs, _ = envs.reset(seed=[FLAGS.seed + i for i in range(FLAGS.num_envs)])

    for global_step in range(FLAGS.num_envs, FLAGS.total_steps + 1, FLAGS.num_envs):

        # TODO: select actions (one for each environment)
        actions = envs.action_space.sample()

        # execute actions in environment
        new_obs, rewards, terminations, truncations, infos = envs.step(actions)
        dones = terminations | truncations

        # vector envs reset automatically, so we have to manually get the terminal observations for these steps
        next_obs = new_obs.copy()
        if dones.any():
            final_info = infos["final_info"]
            for i in np.flatnonzero(dones):
                next_obs[i] = infos["final_obs"][i]
                if final_info.get("_episode", np.zeros(FLAGS.num_envs, bool))[i]:
                    logs["episode_reward"].append(final_info["episode"]["r"][i])
                    logs["episode_frames"].append(final_info["episode"]["l"][i])

        # TODO: update agent

        # set obs to new obs for next step
        obs = new_obs

        # store checkpoint
        if crossed(global_step, FLAGS.checkpoint_freq):

            # TODO: save agent checkpoint

            pass

        # logging
        if crossed(global_step, FLAGS.logging_freq):
            current_log_time = time.time()
            logs['steps_per_second'].append((global_step - last_log_step) / (current_log_time - last_log_time))
            for k, v in logs.items():
                writer.add_scalar(f'{k}/mean', np.mean(v), global_step)
                writer.add_scalar(f'{k}/std', np.std(v), global_step)
                writer.add_scalar(f'{k}/min', np.min(v), global_step)
                writer.add_scalar(f'{k}/max', np.max(v), global_step)
            print(" | ".join([f"step={global_step}"] + [f"{k}={np.mean(v):.2f}" for k, v in sorted(logs.items())]))
            logs = defaultdict(list)
            last_log_step = global_step
            last_log_time = current_log_time

    envs.close()
    writer.close()

    # TODO: save agent checkpoint


def eval():
    env = utils.env.make_env()
    action_space = env.action_space
    env.close()

    # TODO: initialize agent and load checkpoint from FLAGS.eval_checkpoint

    def policy(obs):

        # TODO: select action

        return action_space.sample()

    utils.env.evaluate(policy, preprocess, num_episodes=FLAGS.eval_num_episodes, seed=FLAGS.eval_seed,
                       render_mode='human' if FLAGS.eval_render else None)


def main(_):
    if FLAGS.mode == 'train':
        train()
    elif FLAGS.mode == 'eval':
        eval()


if __name__ == "__main__":
    app.run(main)
