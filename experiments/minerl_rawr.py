import os, sys, glob, datetime
import torch
import numpy as np
import gym
import minerl

from mcbravais.minerl_data import CachedExpertDataset
from spacelib.dataset import RecurrentReplayBuffer
from spacelib.flatter import Flatter
from mcbravais.utils import export_video, time_string
from mcbravais.logging import TensorboardLogger
from mcbravais.rawr_nets import RecurrentAWRAgent

import pdb

SCRIPT_NAME = os.path.basename(__file__).replace('.','_')
START_TIME_STR = time_string()

env_name = 'MineRLObtainDiamond-v0'

observation_space = gym.envs.registration.spec(env_name)._kwargs['observation_space']
action_space = gym.envs.registration.spec(env_name)._kwargs['action_space']

writer = TensorboardLogger(os.path.expanduser(
    f'~/minerl_logs_tensorboard/{SCRIPT_NAME}/{env_name}/{START_TIME_STR}')
)

get_checkpoint_dir = lambda tag: os.path.expanduser(
    f'~/minerl_model_checkpoints/{SCRIPT_NAME}/{env_name}/{START_TIME_STR}/{tag}/{time_string()}'
)

params = {
    'gamma': 0.95,

    'updates_per_episode': 20,

    'early_batches': 10000,

    'batch_size': 8,
    'minibatch_size': 8,
    'replay_sequence_length': 70,
    'sequence_warmup': 20, # number of steps to ignore for sampled lstm sequences

    'pre_training_episodes': 0,
    'random_action_episodes': 1,
    'render_after_episode': 2500,

    'replay_buffer_n_episodes': 1000,
    'max_episodes': int(1e3),
    'max_episode_length': 8000,
    'max_episode_length_no_reward': 5000, # helpful for early exploration in sparse reward envs

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
    'beta': 5.0,
    'learning_rate': 3.e-4
})
dino_bob.save_models(get_checkpoint_dir('before_training'))

replay_buffer = RecurrentReplayBuffer(
    observation_space,
    action_space,
    hidden_dim=None,
    max_num_episodes=params['replay_buffer_n_episodes'],
    data_root='/fast/tmp')

expert_buffer = CachedExpertDataset(env_name, cache_root='/fast/cache_test')

device = torch.device(params['device'])



with dino_bob.torch_device(device):


    for batch_no in range(params['early_batches']):
        samples = expert_buffer.iter_sample(
            length=params['replay_sequence_length'],
            batch_size=params['batch_size'],
            minibatch_size=params['minibatch_size'],
            gamma=params['gamma'],
            hidden=True,
            device=device)
        
        dino_bob.iter_update_combined(samples, warmup_k=params['sequence_warmup'], writer=writer)

        if batch_no%500==0:
            dino_bob.save_models(get_checkpoint_dir(f'early_update_{batch_no}'))

    dino_bob.save_models(get_checkpoint_dir('after_early_batches'))

    env = gym.make(env_name)

    for episode_no in range(params['max_episodes']):
        train_status['episode_no'] = episode_no
        
        if train_status['episode_no'] > params['render_after_episode']:
            env.render()

        episode_info = {
            'total_reward': 0,
            'length': 0
        }

        obs = env.reset()
        replay_buffer.begin_episode()
        stop_episode = False
        while not stop_episode:
            # print('.', end='')

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

        if train_status['episode_no'] >= params['pre_training_episodes']:
            for update_no in range(params['updates_per_episode']):

                samples = expert_buffer.iter_sample(
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
