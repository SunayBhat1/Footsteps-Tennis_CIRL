import gym
from gym import spaces, logger
from gym.utils import seeding
import pandas as pd
import numpy as np
import random

Init_State = (0,50,50)


class FootstepsEnv(gym.Env):
    # A Footsteps/Tennis Startegy Game environment for OpenAI gym
    # metadata = {'render.modes': ['human']}

    def __init__(self):
        self.state = Init_State
        print('Environment Initialized!: (', str(self.state[0]), ',', str(self.state[1]), ',', str(self.state[2]), ')')
        
    def _take_action(self, action):
        print('Action Taken')
    def step(self, action):
        print('Environment initialized')
    def reset(self):
        print('Environment reset')

    def render(self, mode='human', close=False):
        print('Rendering')
