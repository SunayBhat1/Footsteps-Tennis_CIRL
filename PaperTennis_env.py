import gym
from gym import spaces, logger
import numpy as np
import matplotlib.pyplot as plt

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
        0-50  Possible actions are determined by available score. Invalid action will be returned, and a non-zero action is also
              required unless the player has a score of 0.

    Reward:
    Starting State:
    Episode Termination:

    """

    # metadata = {'render.modes': ['human']}

    def __init__(self,Init_State = (0,50,50)):

        super(PaperTennisEnv, self).__init__()

        self.INIT_STATE = (0,50,50)
        self.state = (0,50,50)
        self.gameHistory = np.array([])
        self.action_space = np.arange(50)+1
        self.ep_history = [self.state]
        self.rounds = 0

        self.action_space = spaces.Discrete(51)

        self.observation_space = spaces.Tuple((
        spaces.Discrete(7),
        spaces.Discrete(51),
        spaces.Discrete(51)))

    def get_opponent_action(self,opponent):
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
                    play = int(S_self)
                
            # Uniform Random Player
            if opponent == 4:
                play = round(np.random.rand()*S_self)

            iter_play += 1

        return int(play)

        
    def step(self, action,opponent=4):

        GameScore = self.state[0]
        P1 = self.state[1]
        P2 = self.state[2]

        # Get opponent action
        p2_action = self.get_opponent_action(opponent)

        P1 = P1 - action
        P2 = P2 - p2_action
        
        # Determine next game score
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
            
        
        # Stopping condition G={3,-3} or no points left
        done = bool(
            GameScore == 3
            or GameScore == -3
            or P1+P2 == 0
        )

        # Reward
        if not done:
            reward = 0
        else:
            if GameScore < 0:
                reward = 1
            else:
                reward = -1

        # Update state variables
        self.state = (GameScore,P1,P2)
        self.ep_history.append(self.state)
        self.rounds += 1


        return self.state, reward, done, {'history':self.ep_history}

    def reset(self):
        plt.close()
        self.state = self.INIT_STATE
        self.ep_history = [self.state]
        self.rounds = 0
        # print('Environment Reset!')

    def render(self, render_time = 0.0001):
        plt.close()
        plt.style.use('dark_background')
        img = plt.imread("SupportDocs/court.jpg")
        fig, (ax1,ax2) = plt.subplots(1,2,figsize=(13, 6),gridspec_kw={'width_ratios': [4, 1]})
        ax1.imshow(img,extent=[-3, 3, 0, 7])
        ax1.invert_yaxis()
        ax1.invert_xaxis()
        ax1.set_aspect(0.5)
        ax1.get_yaxis().set_visible(False)
        ax1.plot(list(list(zip(*self.ep_history))[0]),np.linspace(1,6,self.rounds+1),  linewidth=3, marker='o',markersize=12,color='yellow')
        ax2.bar(['Self Score', 'Opponent Score'],[self.ep_history[-1][1],self.ep_history[-1][2]])
        ax2.set_ylim(0, 50)
        ax1.set_title('Game Progress',fontweight='bold',fontsize = 20)
        ax2.set_title('Scores',fontweight='bold',fontsize = 20)
        plt.pause(render_time)
        