import os, sys
import datetime, time
from contextlib import contextmanager
from collections import namedtuple
import numpy as np

import pdb

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tensorboardX import SummaryWriter

import gym
import roboschool


from mcbravais.utils import ensure_dir, time_string
from spacelib.dataset import RecurrentReplayBuffer
from spacelib.flatter import Flatter

from mcbravais.rawr_nets import (
    RecurrentAWRAgent
)
import time
from pprint import pprint 

import pickle
from mcbravais.logging import TensorboardLogger


env_name = 'RoboschoolHalfCheetah-v1'

_env = gym.make(env_name)
observation_space = _env.observation_space
action_space = _env.action_space

SCRIPT_NAME = os.path.basename(__file__)
SCRIPT_START_TIME_STR = time_string()
tensorboard_log_dir = os.path.join(
    os.path.expanduser('~'),
    'minerl_logs_tensorboard',
    SCRIPT_NAME.replace('.','_'),
    env_name,
    SCRIPT_START_TIME_STR)
model_checkpoint_dir = os.path.join(
    os.path.expanduser('~'),
    'minerl_model_checkpoints',
    SCRIPT_NAME.replace('.','_'),
    env_name,
    SCRIPT_START_TIME_STR)

def get_model_checkpoint_dir(tag=None, with_time = False):
    args = [model_checkpoint_dir]
    if tag is not None:
        args += [tag]
    if with_time is not None:
        args += [time_string()]
    return os.path.join(*args)

writer = TensorboardLogger(tensorboard_log_dir)
print(f"Tensorboard logging in {tensorboard_log_dir}")





params = {
    'gamma': 0.95,

    'updates_per_episode': 20,

    'batch_size': 64,
    'minibatch_size': 64,
    'replay_sequence_length': 20,
    'sequence_warmup': 17,

    'pre_training_episodes': 1000,
    'random_action_episodes': 1000,
    'render_after_episode': 2500,

    'replay_buffer_n_episodes': 1000,
    'max_episodes': int(1e7),
    'max_episode_length': 10000,
    'max_episode_length_no_reward': 10000, # helpful for early exploration in sparse reward envs

    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}


train_status = {
    'episode_no': 0.0,
    'best_reward': -np.inf,
    'total_n_episodes': 0,
    'total_n_updates': 0,
    'total_n_steps': 0,
}

dino_bob = RecurrentAWRAgent(observation_space, action_space, params={
    'latent_size': 256,
    'encoder_trunk': [],
    'lstm_hidden_size': 512,
    'policy_trunk_hidden': [512, 512, 512],
    'value_function_hidden': [512, 512, 512],
    'beta': 1.0,
    'learning_rate': 1.e-3,
    'value_function_scaling': None
})

replay_buffer = RecurrentReplayBuffer(
    observation_space,
    action_space,
    hidden_dim=None,
    max_num_episodes=params['replay_buffer_n_episodes'],
    data_root='/fast/tmp')

device = torch.device(params['device'])

with dino_bob.torch_device(device):

    env = gym.make(env_name)
    obs = env.reset()

    for episode_no in range(params['max_episodes']):
        train_status['episode_no'] = episode_no
        
        if train_status['episode_no'] > params['render_after_episode']:
            env.render()

        episode_info = {
            'total_reward': 0,
            'length': 0
        }

        env.reset()
        replay_buffer.begin_episode()
        stop_episode = False
        while not stop_episode:

            if train_status['episode_no'] < params['random_action_episodes']:
                act = env.action_space.sample()
            else:
                act = dino_bob.get_action(obs)


            next_obs, rew, done, _ = env.step(act)
            replay_buffer.append((obs, act, rew, done))
            obs = next_obs

            stop_episode = (
                done or
                (episode_info['length'] > params['max_episode_length']) or
                ( (episode_info['total_reward'] == 0) and (episode_info['length']  > params['max_episode_length_no_reward']) )
            )

            episode_info['total_reward'] += rew
            episode_info['length'] += 1
            train_status['total_n_steps'] += 1

        replay_buffer.end_episode()
        train_status['best_reward'] = max(train_status['best_reward'], episode_info['total_reward'])
        train_status['total_n_episodes'] += 1

        print(episode_info)
        if train_status['total_n_episodes']%100 == 0:
            pprint(train_status)

        if writer:
            writer.add_scalar('episode/reward', episode_info['total_reward'], period=1)
            writer.add_scalar('episode/best_reward', train_status['best_reward'], period=1)

        if train_status['episode_no'] > params['pre_training_episodes']:
            for update_no in range(params['updates_per_episode']):

                samples = replay_buffer.iter_sample(
                    length=params['replay_sequence_length'],
                    batch_size=params['batch_size'],
                    minibatch_size=params['minibatch_size'],
                    gamma=params['gamma'],
                    hidden=True,
                    device=device)
                
                dino_bob.iter_update_combined(samples, warmup_k=params['sequence_warmup'], writer=writer)
                # dino_bob.update_value(sample_batch, writer=writer)
                # dino_bob.update_actor(sample_batch, writer=writer)

                train_status['total_n_updates'] += 1
