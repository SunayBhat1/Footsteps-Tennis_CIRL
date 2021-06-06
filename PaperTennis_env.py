import gym
from gym import spaces
from gym.envs.classic_control import rendering
import numpy as np
from PaperTennisOpponent import PT_Fixed_Oppenent, PT_File_Oppenent
import matplotlib.pyplot as plt


class PaperTennisEnv(gym.Env):
    """
    OpenAI Gym Info
    ----------
    Description (Short): OpenAI gym environment for strategy game Paper Tennis (aka 'Footsteps')

    Input:
        gamespace (Type: 2-Tuple of Ints, Default: (7,50)) - This defines the width in two dimensions of the game. The first 
            input is the 'size of the court'. This includes both players' active space, the net (gamescore =0), and the court 'ends' 
            or winning spots. The default is 7, which corresponds to 5 active spaces, or winning gamescores of 3,-3.

        opponent_strategy (Type: Int (1-4), Default: 4) - Selects the fixed or generated strategy to play against. See 'PaperTennisOpponent.py'
            for full description of fixed strategies. 
            {1: 'Mean', 2: 'Long', 3: 'Short', 4: 'Naive'}

        opponent_source (Type: String, Default: 'Fixed') - The default string selects from the fixed programmed opponents in the
            'PaperTennisOpponent.py' file. Any other source requires this variable to be the file path to the strategy to load. Note
            that unless this creates a class, and one must modify the 'Opponents" file to use any other source method besides a full
            table strategy lookup. 
        
        Default game:   |-3(P2 Wins) --- -2 --- -1 --- 0(Start) --- 1 --- 2 --- 3(P1 Wins)|

    Description (Long):
        This is an OpenAI gym environment of the classic two-player 'paper' strategy game Tennis, also called Footsteps, 
        in which players move simultaneously by using their available points (starting points specified by user, default = 50).
        Each round, the higher point value moves the ball away from the winner to the opponent's side. 
        A win occurs when a player wins a user-defined amount of consecutive rounds (default = 3), and thus places the ball 
        in the opponent's 'end court'. If both players run out of points, a win is awarded to the player who won the last round.
        Computer opponent strategies specified in separate 'PaperTennisOpponent.py' file and class. 

    Source:
        Environment based on the game rules from Wikipedia: https://en.wikipedia.org/wiki/Tennis_(paper_game)

    Observation:
        Type: Discrete(3)
        Num     Observation           Min(Default)       Max(Default)
        0       Game Score (G)            -3                  3
        1       Player 1 Score (S1)        0                  50
        2       Player 2 Score (S2)        0                  50

    Action:
        Type: Discrete(state[1]) Variable/State Dependent
        Num         Action
        0-state[1]  Points played by Agent/Player 1

        Note: Possible actions are determined by available score. Invalid action will be assert an error, 
              and a non-zero action is also required unless the player has a score of 0.

    Reward:
        Reward is 1 for winning the game, -1 for losing, 0 otherwise

    Starting State:
        (0,gamespace[1],gamespace[1]) Default: (0,50,50)

    Episode Termination:
        Gamescore (state[0]) is -3 or 3
        Player 1 and 2 Score is 0 

    Attributes
    ----------
    state: 3-tuple
        Tuple of current game state
    history: numpy array of tuples
        Array of tuples containing a full game/episode history

    Other Files
    ----------
    PaperTennisOpponent.py - Contains classes for fixed opponennt strategies or strategies generated using Genetic Programming
                             Only used for single agent environment. 
    """

    def __init__(self,gamespace = (7,50),opponent_strategy=4,opponent_source = 'Fixed'):

        super(PaperTennisEnv, self).__init__()

        self.INIT_STATE = (0,gamespace[1],gamespace[1])
        self.state = self.INIT_STATE
        self.opponent_strategy = opponent_strategy
        self.opponent_source = opponent_source
        self.history = [self.state]
        self.viewer = None

        if opponent_source == 'Fixed': self.Opponent = PT_Fixed_Oppenent()
        else: self.Opponent = PT_File_Oppenent(opponent_source)

        self.action_space = spaces.Discrete(self.state[1]+1)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(7),
            spaces.Discrete(self.state[1]+1),
            spaces.Discrete(self.state[2]+1)))

    def step(self, action):
        # Check for correct action
        self.action_space = spaces.Discrete(self.state[1]+1)
        error_msg = '{} ({!s}) Invalid action, must be integer between 1 and {}'.format(action, type(action),self.state[1])
        assert (self.action_space.contains(action) and not (action == 0 and self.state[1] != 0)), error_msg

        # Get opponent action
        action2 =  self.Opponent.get_action(self.state, self.opponent_strategy)

        # Update player scores (S1,S2) and gamescore (G)
        (G,S1,S2) = self.state
        S1 = S1 - action
        S2 = S2 - action2
        
        if (action > action2):
            if (G <= 0): G = G - 1
            else: G = -1
        elif (action < action2):
            if (G >= 0): G = G + 1
            else: G = 1
        else: G = G
            
        # Stopping condition G={3,-3} or no points left
        done = bool(
            G == 3
            or G == -3
            or S1+S2 == 0
        )

        # Reward
        if not done: reward = 0
        else:
            if G < 0: reward = 1
            else: reward = -1

        # Update state variables
        self.state = (G,S1,S2)
        self.history.append(self.state)

        return self.state, reward, done, {'history':self.history}

    def reset(self,opponent_strategy=4):

        self.state = self.INIT_STATE
        self.opponent_strategy = opponent_strategy
        self.history = [self.state]
        self.action_space = spaces.Discrete(self.state[1]+1)
        return self.state

    def renderPNG(self):
        plt.style.use('dark_background')
        img = plt.imread("SupportDocs/court.jpg")
        fig, (ax1,ax2) = plt.subplots(1,2,figsize=(13, 6),gridspec_kw={'width_ratios': [4, 1]})
        ax1.imshow(img,extent=[-3, 3, 0, 7])
        ax1.invert_yaxis()
        ax1.invert_xaxis()
        ax1.set_aspect(0.5)
        ax1.get_yaxis().set_visible(False)
        ax1.plot(list(list(zip(*self.history))[0]),np.linspace(1,6,np.shape(self.history)[0]),  linewidth=3, marker='o',markersize=12,color='yellow')
        ax2.bar(['Self Score', 'Opponent Score'],[self.history[-1][1],self.history[-1][2]])
        ax2.set_ylim(0, self.INIT_STATE[1])
        ax1.set_title('Game Progress',fontweight='bold',fontsize = 20)
        ax2.set_title('Scores',fontweight='bold',fontsize = 20)
        plt.style.use('default')
        fig.savefig('SupportDocs/RenderImage.png')

    def render(self,  mode='human'):
        screen_width = 975
        screen_height = 450

        if self.viewer is None: self.viewer = rendering.Viewer(screen_width, screen_height)

        self.renderPNG()
        self.img = rendering.Image('SupportDocs/RenderImage.png', 1,1)  
        self.imgtrans = rendering.Transform()
        self.imgtrans.scale = (screen_width,screen_height)
        self.img.add_attr(self.imgtrans) 
        self.viewer.add_onetime(self.img)
        self.imgtrans.set_translation(screen_width/2, screen_height/2)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

# 2-agent version of environment
class PaperTennis2AgentEnv(gym.Env):
    """
    OpenAI Gym Info
    ----------
    Description (Short): OpenAI gym environment for "2-agent" strategy game Paper Tennis (aka 'Footsteps'). Can be used to test
                         strategies against each other. 

    Input:
        gamespace (Type: 2-Tuple of Ints, Default: (7,50)) - This defines the width in two dimensions of the game. The first 
            input is the 'size of the court'. This includes both players' active space, the net (gamescore =0), and the court 'ends' 
            or winning spots. The default is 7, which corresponds to 5 active spaces, or winning gamescores of 3,-3.
        
        Default game:   |-3(P2 Wins) --- -2 --- -1 --- 0(Start) --- 1 --- 2 --- 3(P1 Wins)|

    Description (Long):
        This is an OpenAI gym environment of the classic two-player 'paper' strategy game Tennis, also called Footsteps, 
        in which players move simultaneously by using their available points (starting points specified by user, default = 50).
        Each round, the higher point value moves the ball away from the winner to the opponent's side. 
        A win occurs when a player wins a user-defined amount of consecutive rounds (default = 3), and thus places the ball 
        in the opponent's 'end court'. If both players run out of points, a win is awarded to the player who won the last round.

    Source:
        Environment based on the game rules from Wikipedia: https://en.wikipedia.org/wiki/Tennis_(paper_game)

    Observation:
        Type: Discrete(3)
        Num     Observation           Min(Default)       Max(Default)
        0       Game Score (G)            -3                  3
        1       Player 1 Score (S1)        0                  50
        2       Player 2 Score (S2)        0                  50

    Action:
        Type: Discrete(playerscore) Variable/State Dependent
        Num         Action
        0-state[1]  Points played by Agent/Player 1
        0-state[2]  Points played by Agent/Player 2

        Note: Possible actions are determined by available score. Invalid action will be assert an error, 
              and a non-zero action is also required unless the player has a score of 0.

    Reward:
        Reward is 1 for winning the game, -1 for losing, 0 otherwise
        Note!!: Reward must be reversed and fed to second agent

    Starting State:
        (0,gamespace[1],gamespace[1]) Default: (0,50,50)

    Episode Termination:
        Gamescore (state[0]) is -3 or 3
        Player 1 and 2 Score is 0 

    Attributes
    ----------
    state: 3-tuple
        Tuple of current game state
    history: numpy array of tuples
        Array of tuples containing a full game/episode history
    """

    def __init__(self,gamespace = (7,50)):

        super(PaperTennis2AgentEnv, self).__init__()

        self.INIT_STATE = (0,gamespace[1],gamespace[1])
        self.state = self.INIT_STATE
        self.Opponent = PT_Oppenent()
        self.history = [self.state]
        self.viewer = None

        self.action_space1 = spaces.Discrete(self.state[1]+1)
        self.action_space2 = spaces.Discrete(self.state[2]+1)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(7),
            spaces.Discrete(self.state[1]+1),
            spaces.Discrete(self.state[2]+1)))

    def step(self, action1,action2):
        # Check for correct action
        self.action_space1 = spaces.Discrete(self.state[1]+1)
        error_msg = '{} ({!s}) Invalid action agent 1, must be integer between 1 and {}'.format(action1, type(action1),self.state[1])
        assert (self.action_space1.contains(action1) and not (action1 == 0 and self.state[1] != 0)), error_msg
        self.action_space2 = spaces.Discrete(self.state[2]+1)
        error_msg = '{} ({!s}) Invalid action agent 2, must be integer between 1 and {}'.format(action2, type(action2),self.state[2])
        assert (self.action_space2.contains(action2) and not (action2 == 0 and self.state[1] != 0)), error_msg

        # Update player scores (S1,S2) and gamescore (G)
        (G,S1,S2) = self.state
        S1 = S1 - action1
        S2 = S2 - action2
        
        if (action1 > action2):
            if (G <= 0): G = G - 1
            else: G = -1
        elif (action1 < action2):
            if (G >= 0): G = G + 1
            else: G = 1
        else: G = G
            
        # Stopping condition G={3,-3} or no points left
        done = bool(
            G == 3
            or G == -3
            or S1+S2 == 0
        )

        # Reward
        if not done: reward = 0
        else:
            if G < 0: reward = 1
            else: reward = -1

        # Update state variables
        self.state = (G,S1,S2)
        self.history.append(self.state)

        return self.state, reward, done, {'history':self.history}

    def reset(self):

        self.state = self.INIT_STATE
        self.history = [self.state]
        self.action_space1 = spaces.Discrete(self.state[1]+1)
        self.action_space2 = spaces.Discrete(self.state[2]+1)
        return self.state

    def renderPNG(self):
        plt.style.use('dark_background')
        img = plt.imread("SupportDocs/court.jpg")
        fig, (ax1,ax2) = plt.subplots(1,2,figsize=(13, 6),gridspec_kw={'width_ratios': [4, 1]})
        ax1.imshow(img,extent=[-3, 3, 0, 7])
        ax1.invert_yaxis()
        ax1.invert_xaxis()
        ax1.set_aspect(0.5)
        ax1.get_yaxis().set_visible(False)
        ax1.plot(list(list(zip(*self.history))[0]),np.linspace(1,6,np.shape(self.history)[0]),  linewidth=3, marker='o',markersize=12,color='yellow')
        ax2.bar(['Self Score', 'Opponent Score'],[self.history[-1][1],self.history[-1][2]])
        ax2.set_ylim(0, self.INIT_STATE[1])
        ax1.set_title('Game Progress',fontweight='bold',fontsize = 20)
        ax2.set_title('Scores',fontweight='bold',fontsize = 20)
        plt.style.use('default')
        fig.savefig('SupportDocs/RenderImage.png')

    def render(self,  mode='human'):
        screen_width = 975
        screen_height = 450

        if self.viewer is None: self.viewer = rendering.Viewer(screen_width, screen_height)

        self.renderPNG()
        self.img = rendering.Image('SupportDocs/RenderImage.png', 1,1)  
        self.imgtrans = rendering.Transform()
        self.imgtrans.scale = (screen_width,screen_height)
        self.img.add_attr(self.imgtrans) 
        self.viewer.add_onetime(self.img)
        self.imgtrans.set_translation(screen_width/2, screen_height/2)
        

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
