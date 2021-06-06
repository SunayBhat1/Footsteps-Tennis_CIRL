from PaperTennis_env import PaperTennisEnv
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
from os import path
from tqdm import tqdm

class MonteCarlo_QTable():

    def __init__(self,args,gamespace= (7,50),opponent_source = 'Fixed'):
        super(MonteCarlo_QTable, self).__init__()
        self.method = "AC"
        self.n = args['NUM_EPISODES']
        self.gamma = args['GAMMA']
        self.alpha = args['ALPHA']
        self.opp_strategy = args['OPP_STRAT']
        self.opp_freq = args['OPP_FREQ']
        self.gamespace = gamespace
        self.opponent_source = opponent_source

        self.space_action_size = [gamespace[0]-2,gamespace[1]+1,gamespace[1]+1,gamespace[1]]  
        self.Q_val = np.zeros(self.space_action_size)
        self.train_history = []
        self.directory = 'MonteCarlo_QT'
        
    def saveModel(self):
        torch.save(self.actor.state_dict(), dirname + 'actor.pt')
        torch.save(self.critic.state_dict(), dirname + 'critic.pt')
        torch.save(self.rewards_history, dirname + 'reward_history.pkl')
        torch.save(self.actor.state_dict(), dirname + 'Archive/actor_' + time.strftime("%Y%m%d-%H%M%S") + '.pt')
        torch.save(self.critic.state_dict(), dirname + 'Archive/critic_' + time.strftime("%Y%m%d-%H%M%S") + '.pt')
        torch.save(self.rewards_history, dirname + 'Archive/reward_history_' + time.strftime("%Y%m%d-%H%M%S") + '.pkl')


    def get_action(state,Q_val):
        if state[1] == 0:
            return 0

        p_epsilon = np.random.uniform(0,1)
        if p_epsilon < E_GREEDY:
            return np.argmax(np.random.uniform(0,1,(1,state[1]))) + 1

        q_s = np.zeros(state[1])

        for i in range(1,state[1]):
            q_s[i] = Q_val[state[0],state[1],state[2],i]

        return np.argmax(q_s) + 1

    def train(self,):

        env = PaperTennisEnv(gamespace = self.gamespace,
                             opponent_strategy=self.opponent_strategy,
                             opponent_source = self.opponent_source)

        wins = np.zeros(self.n)

        Returns = np.zeros(self.space_action_size,dtype=object)
        Counters = np.zeros(self.space_action_size,dtype=object)

        for episode in tqdm(range(self.n)):

            env.reset(OPP_Strat[episode])
            
            done = False

            # Generate an episode
            state_list = list()
            action_list = list()
            reward_list = list()
            
            # Monte Carlo Control (Section 5 S.B., psuedocode)
            state = env.state
            G = 0
            # Generate episode of data
            while not done:
                
                state_list.append(state)
                
                action = get_action(state,Q_val)
                
                state, reward, done, _ = env.step(action,OPP_Strat[episode])
                
                action_list.append(action)
                reward_list.append(reward)

            if reward == 1: wins[episode] = 1
                    
            # Update tables for MC estimation
            for t in range(0,len(state_list)):
                G = GAMMA * G + reward_list[t]
                Returns[state_list[t][0],state_list[t][1],state_list[t][2],action_list[t]-1] += G
                Counters[state_list[t][0],state_list[t][1],state_list[t][2],action_list[t]-1] += 1
                Q_val[state_list[t][0],state_list[t][1],state_list[t][2],action_list[t]-1] = Returns[state_list[t][0],state_list[t][1],state_list[t][2],action_list[t]-1]/Counters[state_list[t][0],state_list[t][1],state_list[t][2],action_list[t]-1]

            if episode >= 100: trend_wins[episode-100] = np.sum(wins[episode-100:episode])
        
        # Update total episode count
        train_episodes += episode +1 


