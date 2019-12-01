import os, sys, glob
import tqdm
import numpy as np
import cv2
import ujson as json
import gym
import torch

import spacelib
import spacelib.dataset

class ExpertEpisode(spacelib.dataset.Episode):
    def __init__(self, env_name):
        action_space, observation_space = get_env_spaces(env_name)
        super().__init__(observation_space, action_space)

    def read(self, cache_path, minerl_recording_path):
        if os.path.isdir(cache_path):
            self.readCache(cache_path)
        else:
            self.readOriginal(minerl_recording_path)
            self.toDisk(cache_path)
        return self

    def readOriginal(self, minerl_recording_path):
        self.set_episode_data(load_recording(minerl_recording_path))
        return self

    def readCache(self, cache_path):
        self.fromDisk(cache_path)
        return self

class CachedExpertDataset(spacelib.dataset.RecurrentReplayBuffer):
    def __init__(self, env_name, cache_root, minerl_data_root=None, hidden_dim=None):
        self.env_name = env_name
        self.data_root = find_path_to_dataset(env_name, data_root=minerl_data_root)
        self.cache_root = cache_root
        action_space, observation_space = get_env_spaces(env_name)
        episode_directories = find_recordings(self.data_root)
        super().__init__(observation_space, action_space, max_num_episodes=len(episode_directories), data_root=self.data_root)

        for ep_dir in tqdm.tqdm(episode_directories, desc='Expert recording', total=len(episode_directories)):
            cache_path = os.path.join(self.cache_root, self.env_name, os.path.basename(ep_dir))
            self.add_episode(
                ExpertEpisode(self.env_name).read(cache_path, ep_dir)
            )

    def begin_episode(self, *args, **kwargs):
        raise Exception("Can't add fresh experiences to cached expert dataset.")


def get_env_spaces(env_name):
    return (gym.envs.registration.spec(env_name)._kwargs['action_space'],
            gym.envs.registration.spec(env_name)._kwargs['observation_space'])


def ensure_dir(path):
    if os.path.exists(path) and not os.path.isdir(path):
        raise ValueError(f"{path} already exists but isn't a directory!")
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
    return path

def find_path_to_dataset(env_name, data_root=None):
    data_root = '' if data_root is None else str(data_root)
    check_paths = [
        data_root,
        os.environ.get('MINERL_DATA_ROOT', ''),
        os.path.expanduser('~/minerl/dataset')
    ]
    for data_root in check_paths:
        if os.path.isdir(data_root):
            break
    else:
        raise ValueError(
            f"Couldn't find minerl data root. Tried "
            + ','.join([f"\n\t'{path}''" for path in check_paths]))

    dataset_path = os.path.join(
        os.path.abspath(os.path.expanduser(data_root)),
        env_name)
    if not os.path.isdir(dataset_path):
        raise ValueError(f"Couldn't find folder for {env_name} data in root {dataset_path}.")
    return dataset_path

def find_recordings(path):
    required_extensions = ['mp4','npz','json']
    return [r
        for r, d, flist in os.walk(path)
        if all([
        any([f.endswith(ext) for f in flist])
        for ext in required_extensions
    ])]

def fix_observation(observation_space, x):
    res = {k: v for k, v in x.items() if k not in ('damage', 'maxDamage', 'type', 'inventory')}
    if 'damage' in x:
        res['equipped_items'] = {
            'mainhand': {
                'damage': x['damage'],
                'maxDamage': x['maxDamage'],
                'type': x['type']
            }
        }
    if 'inventory' in x:
        res['inventory'] = dict(zip(
                observation_space.spaces['inventory'].spaces.keys(),
                np.atleast_2d(x['inventory']).T
        ))
    return res

def read_frames(path, length, shape=(64, 64, 3)):

    cap = cv2.VideoCapture(path)
    # frames = np.empty((2*len, *shape), dtype=np.uint8)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = np.empty((frame_count+200, *shape), dtype=np.uint8)
    ret, frame_num = True, 0
    while ret:
        ret, frame = cap.read()
        if ret:
            cv2.cvtColor(frame, code=cv2.COLOR_BGR2RGB, dst=frames[frame_num])
            frame_num += 1
    cap.release()
    return frames[frame_num-(length+1):frame_num]

def load_recording(path, ix=None):

    paths = {
            'video': os.path.join(path, 'recording.mp4'),
            'numpy': os.path.join(path, 'rendered.npz'),
    }


    env_name = os.path.basename(os.path.dirname(path))
    action_space = gym.envs.registration.spec(env_name)._kwargs['action_space']
    observation_space = gym.envs.registration.spec(env_name)._kwargs['observation_space']

    ### Load npz data
    state = np.load(paths['numpy'], allow_pickle=True)

    episode_length = len(state['reward'])

    if ix is None:
        ix = slice(episode_length)

    reward_vec = state['reward'][ix]
    done_vec = 0*reward_vec
    done_vec[-1] = 1

    act_dict = {k.split('action_')[1]: v[ix] for k, v in state.items() if k.startswith('action_')}
    obs_dict = {k.split('observation_')[1]: v[ix] for k, v in state.items() if k.startswith('observation_')}

    ### Load video frames
    frames = []
    cap = cv2.VideoCapture(paths['video'])
    ret, frame_num = True, 0
    while ret:
        ret, frame = cap.read()
        if ret:
            frame_num += 1
            cv2.cvtColor(frame, code=cv2.COLOR_BGR2RGB, dst=frame)
            frames.append(np.asarray(np.clip(frame, 0, 255), dtype=np.uint8))
    cap.release()
    frames = np.asarray(frames)[-(episode_length):]

    obs_dict['pov'] = frames[ix]

    return (
        fix_observation(observation_space, obs_dict),
        act_dict,
        reward_vec,
        done_vec
    )