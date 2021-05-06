import gym
import numpy as np

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
        Num   Action
        0     Push cart to the left

    Reward:
    Starting State:
    Episode Termination:

    """

    # metadata = {'render.modes': ['human']}

    def __init__(self,Init_State = (0,50,50)):
        self.INIT_state = (0,50,50)
        self.state = (0,50,50)
        self.action_space = np.arange(50)+1

        # print('Environment Initialized!')

    def select_action(self,opponent):
        G = -self.state[0]
        S_self = self.state[2]
        S_opponent = self.state[1]

        play = 0;
        iter_play = 1;
        # Repeat until play is within acceptable bounds 
        while not (play > 0 and play <= S_self):
            
            if iter_play == 10:
                play = S_self
                break
            
            if S_self == 0:
                play = 0
                break

            # Mean player
            if opponent == 1:
                mean_play = S_opponent/2;
                if (G == 0):
                    play = round(np.random.normal(10, 1))
                else:
                    play = round(np.random.normal(mean_play,1))
                
            # Long Player
            if opponent == 2:
                if (G == 0):
                    play = round(np.random.gamma(2,4))
                elif (G == 1):
                    play = round(np.random.gamma(2,4))
                elif (G == 2):
                    play = S_opponent + 1 - round(np.random.gamma(1,2))
                elif (G == -1):
                    mean_play = S_opponent/2;
                    play = round(np.random.normal(mean_play,1))
                elif (G == -2):
                    play = S_opponent + 1 - round(np.random.gamma(1,2))
                
            # Short Player
            if opponent == 3:
                if (G == 0):
                    play = 20 - round(np.random.gamma(2,4))
                elif (G == 1):
                    mean_play = S_opponent/2;
                    play = round(np.random.normal(mean_play,1))
                elif (G == 2):
                    play = S_opponent + 1 - round(np.random.gamma(1,2))
                elif (G == -1):
                    play = round(S_self/2) - round(np.random.gamma(2,4))
                elif (G == -2):
                    play = S_self
                
            # Uniform Random Player
            if opponent == 4:
                play = round(np.random.rand()*S_self)

            iter_play += 1

        return play

        
    def step(self, action,opponent=4):

        GameScore = self.state[0]
        P1 = self.state[1]
        P2 = self.state[2]

        p2_action = self.select_action(opponent)

        P1 = P1 - action
        P2 = P2 - p2_action
        
        if (action > p2_action):
            if (GameScore <= 0):
                GameScore = GameScore - 1
            else:
                GameScore = -1
        elif (action < p2_action):
            if (GameScore >= 0):
                GameScore = GameScore + 1
            else:
                GameScore = 1
        else:
            GameScore = GameScore
            
        
        # Stopping condition (angle of pole 3 is in 60:300, ie. over 60 degrees from upright)
        done = bool(
            GameScore == 3
            or GameScore == -3
            or P1+P2 == 0
        )

        self.state = (GameScore,P1,P2)

        if not done:
            reward = 0
        else:
            if GameScore < 0:
                reward = 1
            else:
                reward = -1


        return self.state, reward, done

    def reset(self):
        self.state = self.INIT_state
        # print('Environment Reset!')

    def render(self, mode='human', close=False):
        print('Rendering')
