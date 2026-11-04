import datetime
import os
import random
import time
from collections import defaultdict
from typing import Callable, Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from absl import flags, app
from tensorboardX import SummaryWriter
from torch import nn, optim

import utils.env
from utils.utils import FrameStackReplayBuffer, LinearSchedule

# common flags
flags.DEFINE_enum('mode', 'train', ['train', 'eval'], 'Run mode.')
flags.DEFINE_string('logdir', './runs', 'Directory where all outputs are written to.')
flags.DEFINE_string('run_name', datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), 'Run name.')
flags.DEFINE_bool('cuda', True, 'Whether to run the model on gpu (if available) or on cpu.')
flags.DEFINE_integer('seed', 42, 'Random seed.')

# train flags (all step counts are environment steps, summed over all parallel envs)
flags.DEFINE_float('gamma', .99, 'Discount factor.')
flags.DEFINE_integer('batch_size', 32, 'Train batch size.')
flags.DEFINE_float('learning_rate', 2.5e-4, 'Learning rate.')
flags.DEFINE_float('max_grad_norm', 10, 'Maximum gradient norm. Gradients with larger norms will be clipped.')
flags.DEFINE_integer('num_envs', 16, 'Number of parallel envs.')
flags.DEFINE_integer('total_steps', 10_000_000, 'Total number of environment steps.')
flags.DEFINE_integer('warmup_steps', 80_000, 'Number of warmup steps to fill the replay buffer.')
flags.DEFINE_integer('buffer_size', 100_000, 'Replay buffer size (needs ~1.4 GB of RAM for 100k transitions).')
flags.DEFINE_float('exploration_epsilon_initial', 1.0, 'Initial exploration rate.')
flags.DEFINE_float('exploration_epsilon_final', 0.1, 'Final exploration rate.')
flags.DEFINE_float('exploration_fraction', 0.1, 'Fraction of total_steps it takes to decay initial to final epsilon.')
flags.DEFINE_integer('train_freq', 4, 'One gradient step every train_freq environment steps.')
flags.DEFINE_integer('checkpoint_freq', 1_000_000, 'Frequency at which checkpoints are stored.')
flags.DEFINE_integer('logging_freq', 10_000, 'Frequency at which logs are written.')
flags.DEFINE_integer('target_network_update_freq', 1000, 'Frequency at which the target network is updated.')

# eval flags
flags.DEFINE_string('eval_checkpoint', None, 'Path to the checkpoint to evaluate, e.g. runs/<run_name>/checkpoint-last.pt')
flags.DEFINE_integer('eval_num_episodes', utils.env.EVAL_EPISODES, 'Number of eval episodes.')
flags.DEFINE_float('eval_epsilon', 0.05, 'Epsilon-greedy during eval.')
flags.DEFINE_bool('eval_render', False, 'Render env during eval.')
flags.DEFINE_integer('eval_seed', utils.env.EVAL_SEED, 'Eval seed.')

FLAGS = flags.FLAGS


def preprocess(env: gym.Env) -> gym.Env:
    """ agent specific preprocessing: frame skipping with max-pooling, grayscale 84x84 frames, stack of 4 frames """
    env = gym.wrappers.AtariPreprocessing(env, noop_max=30, frame_skip=4, screen_size=84, grayscale_obs=True)
    env = gym.wrappers.FrameStackObservation(env, 4)
    return env


def make_env_fn(video_folder: Optional[str] = None) -> Callable[[], gym.Env]:
    """ returns a pickleable callable to create an env instance """
    def env_fn():
        return preprocess(utils.env.make_env(video_folder=video_folder))
    return env_fn


class DQN(nn.Module):
    def __init__(self, num_actions: int, in_channels: int = 4):
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

    # SAME_STEP: when an episode ends, the env is reset within the same step and the
    # terminal observation is returned in infos["final_obs"]
    envs = gym.vector.SyncVectorEnv([
        make_env_fn(video_folder=os.path.join(logdir, 'videos', 'train') if i == 0 else None)
        for i in range(FLAGS.num_envs)], autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)
    envs.action_space.seed(FLAGS.seed)

    exploration_epsilon_schedule = LinearSchedule(
        initial_value=FLAGS.exploration_epsilon_initial,
        final_value=FLAGS.exploration_epsilon_final,
        schedule_steps=int(FLAGS.exploration_fraction * FLAGS.total_steps)
    )

    q_network = DQN(envs.single_action_space.n).to(device)

    target_network = DQN(envs.single_action_space.n).to(device)
    target_network.load_state_dict(q_network.state_dict())

    optimizer = optim.Adam(q_network.parameters(), lr=FLAGS.learning_rate)

    # stores single frames instead of full frame stacks for obs and next_obs (4x less memory)
    stack_size, *frame_shape = envs.single_observation_space.shape
    replay_buffer = FrameStackReplayBuffer(size=FLAGS.buffer_size, num_envs=FLAGS.num_envs,
                                           stack_size=stack_size, frame_shape=frame_shape)

    logs = defaultdict(list)
    last_log_step = 0
    last_log_time = time.time()

    def crossed(step: int, freq: int) -> int:
        """ number of multiples of freq in (step - num_envs, step] """
        return step // freq - (step - FLAGS.num_envs) // freq

    # every env gets its own seed, otherwise all envs would play the same game
    obs, _ = envs.reset(seed=[FLAGS.seed + i for i in range(FLAGS.num_envs)])

    for global_step in range(FLAGS.num_envs, FLAGS.total_steps + 1, FLAGS.num_envs):
        # epsilon greedy action selection (independently for each env)
        epsilon = exploration_epsilon_schedule.value(global_step)
        with torch.no_grad():
            q_values = q_network(torch.tensor(obs, device=device))
        actions = torch.argmax(q_values, dim=1).cpu().numpy()
        explore = np.random.random(FLAGS.num_envs) < epsilon
        actions[explore] = np.random.randint(envs.single_action_space.n, size=explore.sum())
        logs["epsilon"].append(epsilon)

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

        # save data to replay buffer. Only terminations mask the next q values, truncated
        # episodes (time limit reached) would have continued.
        replay_buffer.put(obs, actions, rewards, next_obs, dones=terminations, episode_ends=dones)

        # set obs to new obs for next step
        obs = new_obs

        # optimize model (after initial warmup phase to fill the replay buffer)
        if global_step > FLAGS.warmup_steps:
            for _ in range(crossed(global_step, FLAGS.train_freq)):
                # sample a batch from the replay buffer
                batch = {
                    k: torch.tensor(v, device=device)
                    for k, v in replay_buffer.sample(FLAGS.batch_size).items()
                }

                # compute td target from the best q values of the next states (estimated by the target network);
                # mask q values where the episode has ended at the current step
                with torch.no_grad():
                    next_q_value, _ = target_network(batch['next_obs']).max(dim=1)
                    td_target = batch['rewards'] + FLAGS.gamma * next_q_value * (1 - batch['dones'])

                # compute estimated q values of actions taken in current step
                selected_q_values = q_network(batch['obs']).gather(1, batch['actions'].unsqueeze(1)).squeeze(1)

                # compute loss (huber loss)
                loss = F.smooth_l1_loss(selected_q_values, td_target)

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(q_network.parameters(), FLAGS.max_grad_norm)
                optimizer.step()

                logs['loss'].append(loss.item())
                logs['q_values'].append(selected_q_values.mean().item())
                logs['grad_norm'].append(grad_norm.item())

            # update the target network
            if crossed(global_step, FLAGS.target_network_update_freq):
                target_network.load_state_dict(q_network.state_dict())

        # store checkpoint
        if crossed(global_step, FLAGS.checkpoint_freq):
            torch.save(q_network.state_dict(), os.path.join(logdir, f'checkpoint-{global_step}.pt'))

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
    torch.save(q_network.state_dict(), os.path.join(logdir, 'checkpoint-last.pt'))


def eval(device):
    if FLAGS.eval_checkpoint is None:
        raise app.UsageError("--eval_checkpoint is required in eval mode")

    num_actions = utils.env.make_env().action_space.n
    q_network = DQN(num_actions).to(device)
    q_network.load_state_dict(torch.load(FLAGS.eval_checkpoint, map_location=device))
    q_network.eval()

    def policy(obs):
        if random.random() < FLAGS.eval_epsilon:
            return random.randrange(num_actions)
        with torch.no_grad():
            q_values = q_network(torch.tensor(obs, device=device).unsqueeze(0))
        return int(torch.argmax(q_values, dim=1).item())

    utils.env.evaluate(policy, preprocess, num_episodes=FLAGS.eval_num_episodes, seed=FLAGS.eval_seed,
                       render_mode='human' if FLAGS.eval_render else None)


def main(_):
    device_name = "cuda" if FLAGS.cuda and torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    print(f"Using device: {device_name}")

    if FLAGS.mode == 'train':
        train(device)
    elif FLAGS.mode == 'eval':
        eval(device)


if __name__ == "__main__":
    app.run(main)
