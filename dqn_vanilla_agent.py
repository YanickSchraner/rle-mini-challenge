import datetime
import os
import time
import random
from typing import Callable, Optional, Literal

from absl import flags, app
from tensorboardX import SummaryWriter
import gymnasium as gym
import numpy as np
import torch
from torch import nn

import utils.env

# common flags
flags.DEFINE_enum('mode', 'train', ['train', 'eval'], 'Run mode.')
flags.DEFINE_string('logdir', './runs', 'Directory where all outputs are written to.')
flags.DEFINE_string('run_name', datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), 'Run name.')
flags.DEFINE_bool('mps', True, 'Whether to run the model on gpu or on cpu.')
flags.DEFINE_integer('seed', 42, 'Random seed.')

# train flags
flags.DEFINE_integer('num_envs', 16, 'Number of parallel env processes.')

# eval flags
flags.DEFINE_integer('eval_num_episodes', 30, 'Number of eval episodes.')
flags.DEFINE_bool('eval_render', False, 'Render env during eval.')
flags.DEFINE_integer('eval_seed', 1234, 'Eval seed.')

FLAGS = flags.FLAGS

def make_env_fn(seed: int, render_mode: Literal['human', 'rgb_array', None] = 'human', video_folder: Optional[str] = None) -> Callable[[], gym.Env]:
    """ returns a pickleable callable to create an env instance """
    def env_fn():
        env = utils.env.make_env(render_mode, video_folder)
        return env
    return env_fn

class DQN(nn.Module):
    def __init__(self, num_actions: int, in_channels: int = 1):
        super().__init__()
        self.qnet = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )

    def forward(self, obs):
        if len(obs.shape) == 3:
            obs = torch.unsqueeze(obs, dim=1)  # add channel dim
        obs = obs * (1. / 255.)
        return self.qnet(obs)

def train(device):
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)

    logdir = os.path.join(FLAGS.logdir, FLAGS.run_name)
    os.makedirs(logdir, exist_ok=False)

    FLAGS.append_flags_into_file(os.path.join(logdir, 'flags.txt'))

    writer = SummaryWriter(os.path.join(logdir, 'logs'))
    writer.add_text("config", FLAGS.flags_into_string())

    envs = gym.vector.SyncVectorEnv([
        make_env_fn(FLAGS.seed, 'rgb_array', video_folder=os.path.join(logdir, 'videos', 'train') if i == 0 else None)
        for i in range(FLAGS.num_envs)])

    env_name = envs.envs[0].spec.id

def evaluate(device):
    env_fn = make_env_fn(FLAGS.eval_seed, FLAGS.eval_render)
    env = env_fn()

    q_network = DQN(env.action_space.n).to(device)
    q_network.load_state_dict(torch.load(os.path.join(FLAGS.logdir, FLAGS.run_name, FLAGS.eval_checkpoint)))
    q_network.eval()

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
            q_values = q_network(torch.tensor([observation]).to(device))

            if random.random() < FLAGS.eval_epsilon:
                action = env.action_space.sample()
            else:
                action = int(torch.argmax(q_values, dim=1).cpu().numpy())

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
    device_name = "mps" if FLAGS.mps else "cpu"
    if device_name == "mps" and not torch.mps.is_available():
        raise RuntimeError("mps=true, but cuda is not available")
    device = torch.device(device_name)
    print(f"Using device: {device_name}")

    if FLAGS.mode == 'train':
        train(device)
    elif FLAGS.mode == 'eval':
        evaluate(device)


if __name__ == "__main__":
    app.run(main)