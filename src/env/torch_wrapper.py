from gym import Wrapper
import numpy as np
from ..torch_util import torchify, numpyify

ACTION_SIGMA = 0.1  # TODO


class TorchWrapper(Wrapper):
    def reset(self):
        return torchify(self.env.reset())

    def step(self, action):
        num_action = numpyify(action)
        z = np.random.normal(size=num_action.shape)
        sto_action = num_action + ACTION_SIGMA * z
        clip_sto_action = np.clip(sto_action, -1.0, 1.0)    # TODO
        observation, reward, done, info = self.env.step(clip_sto_action)
        return torchify(observation), float(reward), done, info
