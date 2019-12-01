import torch
import numpy as np
import gym
import minerl

import pdb

from mcbravais.minerl_data import (
    CachedExpertDataset
)

from mcbravais.utils import export_video

env_name = 'MineRLObtainDiamond-v0'

ds = CachedExpertDataset(
    env_name=env_name,
    cache_root='/fast/cache_test')

for k, e in enumerate(ds.episodes):
    print(f"Episode {k} length is {len(e)} steps.")

export_video(ds.sample_sequence(1000).obs[-1], '/fast/test.webm')