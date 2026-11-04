from typing import Tuple, Dict

import numpy as np


class LinearSchedule(object):
    def __init__(self, initial_value: float, final_value: float, schedule_steps: int) -> None:
        super().__init__()
        self._initial_value = initial_value
        self._final_value = final_value
        self._schedule_steps = schedule_steps

    def value(self, t: int):
        interpolation = min(t / self._schedule_steps, 1.0)
        return self._initial_value + interpolation * (self._final_value - self._initial_value)


class RingBuffer(object):
    def __init__(self, size, specs: Dict[str, Tuple[Tuple, np.dtype]]):
        self.size = size
        self.specs = specs
        self.buffers = {k: np.empty((size,) + tuple(shape), dtype) for k, (shape, dtype) in specs.items()}
        self.next_idx = 0
        self.num_in_buffer = 0

    def __len__(self):
        return self.num_in_buffer

    def put(self, samples: Dict[str, np.ndarray]) -> None:
        num_samples = next(iter(samples.values())).shape[0]
        for key, buffer in self.buffers.items():
            features = samples[key]
            assert features.shape[0] == num_samples
            if self.next_idx+num_samples > self.size:
                num_until_end = self.size - self.next_idx
                buffer[self.next_idx:] = features[:num_until_end]
                buffer[:num_samples - num_until_end] = features[num_until_end:]
            else:
                buffer[self.next_idx:self.next_idx+num_samples] = features
        self.next_idx = (self.next_idx + num_samples) % self.size
        self.num_in_buffer = min(self.size, self.num_in_buffer + num_samples)

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        idx = np.random.randint(0, self.num_in_buffer, batch_size)
        return {
            key: buffer[idx]
            for key, buffer in self.buffers.items()
        }


class FrameStackReplayBuffer(object):
    """Memory efficient replay buffer for stacked frames (`gym.wrappers.FrameStackObservation`).

    Consecutive observations share all but one frame. Instead of storing full stacks for obs and next_obs,
    only the newest frame of each is stored and the stacks are rebuilt from the previous steps of the same
    env when sampling. Frames before the start of an episode are padded with its first frame, like
    `FrameStackObservation(padding_type="reset")` does.
    """

    def __init__(self, size: int, num_envs: int, stack_size: int, frame_shape: Tuple, frame_dtype: np.dtype = np.uint8):
        self.capacity = size // num_envs  # number of vector steps
        self.num_envs = num_envs
        self.stack_size = stack_size
        self.obs_frames = np.zeros((self.capacity, num_envs) + tuple(frame_shape), frame_dtype)
        self.next_frames = np.zeros((self.capacity, num_envs) + tuple(frame_shape), frame_dtype)
        self.episode_steps = np.zeros((self.capacity, num_envs), np.int64)
        self.actions = np.zeros((self.capacity, num_envs), np.int64)
        self.rewards = np.zeros((self.capacity, num_envs), np.float32)
        self.dones = np.zeros((self.capacity, num_envs), np.float32)
        self.next_idx = 0
        self.num_in_buffer = 0  # number of vector steps
        self._current_episode_steps = np.zeros(num_envs, np.int64)

    def __len__(self):
        return self.num_in_buffer * self.num_envs

    def put(self, obs: np.ndarray, actions: np.ndarray, rewards: np.ndarray, next_obs: np.ndarray,
            dones: np.ndarray, episode_ends: np.ndarray) -> None:
        """ stores one vector step. dones masks the next q values (terminations), episode_ends are
        terminations or truncations (after which the env is reset) """
        i = self.next_idx
        self.obs_frames[i] = obs[:, -1]
        self.next_frames[i] = next_obs[:, -1]
        self.episode_steps[i] = self._current_episode_steps
        self.actions[i] = actions
        self.rewards[i] = rewards
        self.dones[i] = dones
        self._current_episode_steps = np.where(episode_ends, 0, self._current_episode_steps + 1)
        self.next_idx = (self.next_idx + 1) % self.capacity
        self.num_in_buffer = min(self.capacity, self.num_in_buffer + 1)

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        # skip the oldest stack_size - 1 vector steps, the frames before them are already overwritten
        ages = np.random.randint(0, self.num_in_buffer - (self.stack_size - 1), batch_size)
        idx = (self.next_idx - 1 - ages) % self.capacity
        envs = np.random.randint(0, self.num_envs, batch_size)

        # number of steps to go back for every frame of the stack, oldest first
        steps_back = np.minimum(np.arange(self.stack_size - 1, -1, -1)[None, :],
                                self.episode_steps[idx, envs][:, None])
        obs = self.obs_frames[(idx[:, None] - steps_back) % self.capacity, envs[:, None]]
        next_obs = np.concatenate([obs[:, 1:], self.next_frames[idx, envs][:, None]], axis=1)
        return {
            'obs': obs,
            'next_obs': next_obs,
            'actions': self.actions[idx, envs],
            'rewards': self.rewards[idx, envs],
            'dones': self.dones[idx, envs],
        }