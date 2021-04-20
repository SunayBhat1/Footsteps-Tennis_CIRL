import gym
from gym import spaces
import numpy as np
import random

Init_State = (0,50,50)
Opponent_Mode = 0


class PaperTennisEnv(gym.Env):
    # A PaperTennis Startegy Game environment for OpenAI gym

    """
    Description:
        This is the classic two-player 'paper' strategy game Tennis in which players move simultaneously
        to use their available points. Each round, the higher point value moves the ball away to the opponents
        side. A win occurs when a player wins three consecutive rounds. 

    Source:
        Devloped based on the game rules from wikipedia: https://en.wikipedia.org/wiki/Tennis_(paper_game)

    Observation:
        Type: Box(3)
        Num     Observation               Min                     Max
        0       Game Score                -3                      3
        1       Player 1 Score (P1)        0                      50
        2       Player 2 Score (P2)        0                      50

    Actions:
        Type: Discrete()
        Num   Action
        0     Push cart to the left
        1     Push cart to the right

    Reward:
        Reward is 1 for every step taken, including the termination step
    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]
    Episode Termination:
        Pole Angle is more than 12 degrees.
        Cart Position is more than 2.4 (center of the cart reaches the edge of
        the display).
        Episode length is greater than 200.
        Solved Requirements:
        Considered solved when the average return is greater than or equal to
        195.0 over 100 consecutive trials.
    """

    # metadata = {'render.modes': ['human']}

    def __init__(self):
        self.action_space = spaces.Discrete(Init_State[1])
        self.observation_space = spaces.Tuple((
            spaces.Discrete(7),
            spaces.Discrete(Init_State[1]+1),
            spaces.Discrete(Init_State[2]+1)))

        print('Environment Initialized!')
        
    def step(self, action):

        GameScore = self.state[0]
        P1 = self.state[1]
        P2 = self.state[2]

        p2_action = np.round(random.uniform(0,P2))

        P1 = P1 - action
        P2 = P2 - p2_action
        
        if (action > p2_action):
            if (GameScore <= 0):
                GameScore = GameScore - 1;
            else:
                GameScore = -1;
        elif (action < p2_action):
            if (GameScore >= 0):
                GameScore = GameScore + 1;
            else:
                GameScore = 1;
        else:
            GameScore = GameScore
            
        reward = 0;
        done = False;
        self.state = (GameScore,P1,P2)

        print('P1: ',str(action), ' P2: ', str(p2_action))

        return self.state, reward, done, {}

    def reset(self):
        self.state = Init_State
        print('Environment Reset!: (', str(self.state[0]), ',', str(self.state[1]), ',', str(self.state[2]), ')')

    def render(self, mode='human', close=False):
        print('Rendering')
