from pathlib import Path

import numpy as np
import torch

from src.util import set_seed

np.set_printoptions(precision=3, linewidth=120)

from src.defaults import ROOT_DIR
from src.checkpoint import CheckpointableData, Checkpointer
from src.torch_util import device
from src.shared import get_env
from src.smbpo import SMBPO

ROOT_DIR = Path(ROOT_DIR)
SAVE_PERIOD = 5

set_seed(1)
torch.set_num_threads(1)
env_name = "humanoid"   # TODO

env_factory = lambda: get_env(env_name)
data = CheckpointableData()
alg = SMBPO(SMBPO.Config(), env_factory, data, islog=False)
alg.to(device)
checkpointer = Checkpointer(alg, "", 'ckpt_{}.pt')  # TODO
data_checkpointer = Checkpointer(data, "", 'data.pt')  # TODO

# Check if existing run
if data_checkpointer.try_load():
    print('Data load succeeded')
    if checkpointer.try_load(650):  # TODO
        print('Solver load succeeded')
        env = get_env(env_name)
        state = env.reset()
        i = 0
        gamma_return = 0
        for _ in range(1000):
            env.render()
            policy = alg.actor
            action = policy.act1(state, eval=True)
            state, reward, done, info = env.step(action)
            i = i + 1
            print(info)
            gamma_return += 0.99 * reward
            if done:
                break
        env.close()
        print("i :", i)
        print("return :", gamma_return)
    else:
        print('Solver load failed')
else:
    print('Data load failed')
