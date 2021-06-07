import numpy as np
import pickle

class PT_Fixed_Oppenent():

    def __init__(self):

        super(PT_Fixed_Oppenent, self).__init__()


    def get_action(self,state,opponent):
        G = -state[0]
        S1 = state[2] # reversed for player 2
        S2 = state[1]

        play = 0;
        iter_play = 1;

        # Repeat until play is within acceptable bounds 
        while not (play > 0 and play <= S1):
            
            if iter_play == 10:
                play = S1
                break
            
            if S1 == 0:
                play = 0
                break

            mean_play = S2/2

            # Mean (Middle-Ground) player
            if opponent == 1:
                if (G == 0): play = round(np.random.normal(10, 1))
                else: play = round(np.random.normal(mean_play,1))
                
            # Long (Cautious) Player
            if opponent == 2:
                if (G == 0): play = round(np.random.gamma(2,4))
                elif (G == 1): play = round(np.random.gamma(2,4))
                elif (G == 2): play = S2 + 1 - round(np.random.gamma(1,2))
                elif (G == -1): play = round(np.random.normal(mean_play,1))
                elif (G == -2): play = S2 + 1 - round(np.random.gamma(1,2))
                
            # Short (Aggressive) Player
            if opponent == 3:
                if (G == 0): play = 20 - round(np.random.gamma(2,4))
                elif (G == 1): play = round(np.random.normal(mean_play,1))
                elif (G == 2): play = S2 + 1 - round(np.random.gamma(1,2))
                elif (G == -1): play = round(S1/2) - round(np.random.gamma(2,4))
                elif (G == -2): play = int(S1)
                
            # Uniform Random Player
            if opponent == 4: play = round(np.random.rand()*S1)

            iter_play += 1

        return int(play)


class PT_File_Oppenent():

    def __init__(self,filepath):

        super(PT_File_Oppenent, self).__init__()
        self.StrategyMatrix = pickle.load(open(filepath, 'rb' ))

    def get_action(self,state,opponent):
        return int(self.StrategyMatrix[opponent,state[0],state[2],state[1]])
