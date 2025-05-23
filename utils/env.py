from typing import Callable, Literal

import gymnasium as gym
import ale_py

gym.register_envs(ale_py)


def make_env(render_mode: Literal['human', 'rgb_array', None] = 'human', video_folder: str = None, video_episode_trigger: Callable[[int], bool] = None):
    env = gym.make("ALE/SpaceInvaders-v5", render_mode=render_mode)
    # env = gym.make("ALE/SpaceInvaders-v5",
    #                obs_type='grayscale',
    #                frameskip=3,
    #                repeat_action_probability=0.,
    #                render_mode='human' if render_human else None)
    #env = gym.wrappers.RecordEpisodeStatistics(env)
    if video_folder is not None:
        env = gym.wrappers.RecordVideo(
            env,
            episode_trigger=lambda num: num % 2 == 0,
            video_folder="saved-video-folder",
            name_prefix="video-",
        )
        #env = gym.wrappers.RecordVideo(env, video_folder, video_episode_trigger)
    return env
