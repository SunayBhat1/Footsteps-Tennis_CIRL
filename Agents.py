from PaperTennis_env import PaperTennisEnv
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import time
import pickle
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

# SARSA Linear Spline Function Approx
class SARSA_LSFA():
    '''
    Description: Paper Tennis Reinforcement Learning agent utilizing a linear spline function approximation 
    and the SARSA method as specified in Sutton and Barto, Section 10.1.

    Source: Sutton, R. S., Barto, A. G. (2018 ). Reinforcement Learning: An Introduction. The MIT Press

    Attributes
    ----------

    Methods
    ----------
    '''

    def __init__(self,args,gamespace= (7,50),opponent_source = 'Fixed'):
        super(SARSA_LSFA, self).__init__()
        self.n = args['NUM_EPISODES']
        self.gamma = args['GAMMA']
        self.alpha = args['ALPHA']
        self.epsilon = args['EPSILON']
        self.opp_strategy = args['OPP_STRAT']
        self.opp_freq = args['OPP_FREQ']
        self.label = args['add_label']
        self.mean_window = args['AVG_WINDOW']
        self.n_test = args['TEST_EPISODES']
        self.check_stable = args['CHECK_STABLE']
        self.gamespace = gamespace
        self.opponent_source = opponent_source
        self.train_history = np.array([])
        self.train_iterations = 0
        self.FixedStrategies = {
            1: 'Mean',
            2: 'Long',
            3: 'Short',
            4: 'Naive',
            5: 'Dynamic'
        }
        if (self.opponent_source == 'Fixed'): self.strat_label = self.FixedStrategies[self.opp_strategy]
        else: self.strat_label = str(self.opp_strategy)

        # Generate opponent strategy vector
        if (self.opp_strategy) == 5:
            self.opp_strat_vector = np.random.randint(1,5,int(self.n/self.opp_freq))
            self.opp_strat_vector = np.repeat(self.opp_strat_vector, self.opp_freq)
        else: 
            self.opp_strat_vector = np.repeat(self.opp_strategy,self.n)

        # SARSA LSFA Specific
        self.w = np.random.rand(5,28)
        self.directory = 'Data/SARSA_LSFA/'

        # Save Args in text file
        with open('Data/SARSA_LSFA/RunInfo/' + self.strat_label + self.label + '_Info.txt','w') as data: data.write(str(args))

    def save_model(self):
        pickle.dump([self.train_history,self.w], open(self.directory + self.strat_label + self.label + '_Model.p', 'wb' ))

    def reset_model(self): 
        self.w = np.random.rand(5,28)
        self.train_history = np.array([])

    def load_model(self):
        [self.train_history,self.w] = pickle.load(open(self.directory + self.strat_label + self.label + '_Model.p', 'rb' ))

    def get_features(self,state,action):
        feat_vec = np.zeros(28)

        # Splines : Discretized spaces
        state_splines = [list(range(0,6)),list(range(6,18)),list(range(18,34)),list(range(34,51))]
        diff_splines = [list(range(-48,-25)),list(range(-25,-16)),list(range(-16,-11)),list(range(-11,-7)),
                        list(range(-7,-3)),list(range(-3,0)),list(range(0,1)),list(range(1,4)),
                        list(range(4,8)),list(range(8,12)), list(range(12,17)),list(range(17,26)),list(range(26,49))]
        action_splines = [list(range(1,2)),list(range(2,5)),list(range(5,10)),list(range(10,17)),list(range(17,26)),list(range(26,51))]

        # Activate state indiicator
        for i,spline in enumerate(state_splines):
            if state[1] in spline: feat_vec[i] = 1
            if state[2] in spline: feat_vec[i+4] = 1

        # Activate diff indicator
        for i,spline in enumerate(diff_splines):
            if (state[1]-state[1]) in spline: feat_vec[i+8] = 1
        
        # Activate action indicator
        for i,spline in enumerate(action_splines): 
            if action in spline: feat_vec[i+22] = 1
            
        return feat_vec

    def Q_value(self,state,action,weights):
        feat_vec = self.get_features(state,action)
        return (weights @ feat_vec).item()

    def get_action(self,state,weights):

        if state[1] == 0: return 0
    
        sample_epsilon = np.random.uniform(0,1)
        if (sample_epsilon < self.epsilon): return np.random.randint(1,state[1]+1)

        q_s = np.zeros(state[1])

        for i in range(0,state[1]): q_s[i] = self.Q_value(state,i,weights)

        return np.argmax(q_s) + 1

    def plot(self,type,save_data,iters=1):
        if type == 'train':
            win100 = np.flip(np.convolve(np.flip(self.train_history), np.ones(100), mode='valid'))
            win_avg = np.flip(np.convolve(np.flip(win100), np.ones(self.mean_window)/self.mean_window, mode='valid'))

            fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(15,5.5), facecolor='w', edgecolor='k')
            fig.suptitle('SARSA LSFA {} Agent ({} Iterations) -  Last {:,} Avg : {:.2f}%'.format(self.strat_label,iters,self.mean_window,win_avg[-1]),fontweight='bold',fontsize = 15)

            ax1.plot(range(99,self.n),win100,c='b')
            ax1.set_title('Running 100 Game Win%',fontweight='bold',fontsize = 14)
            ax1.set_xlabel('Episode',fontweight='bold',fontsize = 12)
            ax1.set_ylabel('Last 100 Win %',fontweight='bold',fontsize = 12)
            ax1.grid(True, color='w', linestyle='-', linewidth=1)

            ax2.plot(range(99+self.mean_window-1,self.n),win_avg,c='g')
            ax2.set_title('Running {:,} Avg Win%'.format(self.mean_window), fontweight='bold',fontsize = 14)
            ax2.set_xlabel('Episode', fontweight='bold',fontsize = 12)
            ax2.set_ylabel('Mean Win %', fontweight='bold',fontsize = 12)
            ax2.grid(True, color='w', linestyle='-', linewidth=1)

            if (self.opp_strategy == 5 and self.opp_freq > 1):
                ax3 = ax1.twinx()
                ax3.plot(range(100,self.n),self.opp_strat_vector[100:] , 'r-',alpha=0.3)
                ax3.set_ylabel('Opponent Strategy', color='r')

            if save_data: fig.savefig(self.directory + 'TrainResults/' + self.strat_label + self.label + '.png')
            else: plt.show()
            plt.close()

        if type == 'test':
            if (self.opponent_source == 'Fixed'): xlabels = ['Mean','Long','Short','Naive','Dynamic']
            else: xlabels = ['1','2','3','4','Dynamic']

            fig2,ax4 = plt.subplots(1,1,figsize=(10,6))
            ax4.bar(xlabels,self.test_win_percent)
            ax4.set_title('SARSA LSFA {} Agent Win %\n Number Test Episodes: {}'.format(self.strat_label,self.n_test),fontweight='bold',fontsize = 15)
            ax4.set_xlabel('Opponent Strategies',fontweight='bold',fontsize = 12)
            ax4.set_ylabel('Win %',fontweight='bold',fontsize = 12)
            for i,y in enumerate(self.test_win_percent):
                    ax4.text(i, y, y, ha = 'center',fontweight='bold',fontsize = 12)

            if save_data: fig2.savefig(self.directory + 'TestResults/' + self.strat_label + self.label + '.png')
            else: plt.show()
            plt.close()

    def train(self,save_data=True):
    
        train_iters = 0
        start_time = time.time()
        while (self.check_stable == True or train_iters < 2):
            print('Training Iteration {}:'.format(train_iters+1))
            
            env = PaperTennisEnv(gamespace = self.gamespace,
                                opponent_strategy = self.opp_strategy,
                                opponent_source = self.opponent_source)

            wins = np.zeros(self.n)

            #Loop through episodes
            for episode in tqdm(range(self.n),ncols=100):

                state = env.reset(self.opp_strat_vector[episode])
                action = self.get_action(state,self.w[state[0],:])
                
                done = False

                # Episode
                while not done:
                    # Step Environment
                    state_prime, reward, done, _ = env.step(action)

                    # Get Action e-greedy 
                    action_prime = self.get_action(state_prime,self.w[state_prime[0],:])

                    # SARSA Update
                    td_update = (reward + self.gamma * self.Q_value(state_prime,action_prime,self.w[state_prime[0],:])
                        -self.Q_value(state,action,self.w[state[0],:])) * self.get_features(state,action)

                    self.w[state[0],:] = self.w[state[0],:] + self.alpha * td_update

                    # Update state/action
                    state  = state_prime
                    action = action_prime

                wins[episode] = int(reward > 0)

            train_iters += 1
            if self.check_stable == True:
                win100 = np.flip(np.convolve(np.flip(wins), np.ones(100), mode='valid'))
                win_avg = np.flip(np.convolve(np.flip(win100), np.ones(self.mean_window)/self.mean_window, mode='valid'))
                if win_avg[-1] > 75: self.check_stable = False
                else:
                    print('Unstable run ({:.2f}%) ...'.format(win_avg[-1]))
                    self.reset_model()

        # Update total episode count
        self.train_history = np.concatenate((self.train_history,wins))

        if save_data: self.save_model()

        self.plot('train',save_data,iters = train_iters)

        # Print run details
        run_time = time.time() - start_time
        print('Iterations: {}'.format(train_iters))
        if run_time < 60: print('--- {:.2f} seconds ---'.format(run_time))
        else: print('--- {:.2f} minutes ---'.format(run_time/60))
        with open('Data/SARSA_LSFA/RunInfo/' + self.strat_label + self.label + '_Info.txt','a') as data: 
            data.write('\n\nRun Time: {:.2f} seconds\n\nTrain Iteration: {}'.format(run_time,train_iters))
        
        return self.train_history

    def evaluate(self,save_data=True):
        print('Evaluating:')
        test_wins = np.zeros((5,self.n_test))

        env = PaperTennisEnv(gamespace = self.gamespace,
                             opponent_strategy = self.opp_strategy,
                             opponent_source = self.opponent_source)

        # Run Through Strategies
        for opp_strategy in tqdm(range(1,6),ncols=100):

            if opp_strategy < 5: OPP_Strat = np.repeat(opp_strategy, self.n_test)
            else: OPP_Strat = np.random.randint(1,5,self.n_test)

            for episode in range(self.n_test):
                state = env.reset(OPP_Strat[episode])
                done = False
                while not done:
                    action = self.get_action(state,self.w[state[0],:])
                    state, reward, done, _ = env.step(action)

                test_wins[opp_strategy-1,episode] = int(reward > 0)

        self.test_win_percent = np.around(np.count_nonzero(test_wins,axis=1)/self.n_test * 100,2)
        self.plot('test',save_data)

        return self.test_win_percent


# SARSA Q-Value Tabular Agent
class SARSA_QTable():
    '''
    Description: Paper Tennis Reinforcement Learning agent utilizing the tabular Q-value SARSA
    method as specified in Sutton and Barto, Section 6.5.

    Source: Sutton, R. S., Barto, A. G. (2018 ). Reinforcement Learning: An Introduction. The MIT Press

    Attributes
    ----------

    Methods
    ----------
    '''

    def __init__(self,args,gamespace= (7,50),opponent_source = 'Fixed'):
        super(SARSA_QTable, self).__init__()
        self.n = args['NUM_EPISODES']
        self.gamma = args['GAMMA']
        self.alpha = args['ALPHA']
        self.epsilon = args['EPSILON']
        self.opp_strategy = args['OPP_STRAT']
        self.opp_freq = args['OPP_FREQ']
        self.label = args['add_label']
        self.mean_window = args['AVG_WINDOW']
        self.n_test = args['TEST_EPISODES']
        self.gamespace = gamespace
        self.opponent_source = opponent_source
        self.train_history = np.array([])
        self.FixedStrategies = {
            1: 'Mean',
            2: 'Long',
            3: 'Short',
            4: 'Naive',
            5: 'Dynamic'
        }
        if (self.opponent_source == 'Fixed'): self.strat_label = self.FixedStrategies[self.opp_strategy]
        else: self.strat_label = str(self.opp_strategy)

        # Generate opponent strategy vector
        if (self.opp_strategy) == 5:
            self.opp_strat_vector = np.random.randint(1,5,int(self.n/self.opp_freq))
            self.opp_strat_vector = np.repeat(self.opp_strat_vector, self.opp_freq)
        else: 
            self.opp_strat_vector = np.repeat(self.opp_strategy,self.n)

        # SARSA QT Specific
        self.space_action_size = [gamespace[0]-2,gamespace[1]+1,gamespace[1]+1,gamespace[1]]  
        self.Q_val = np.zeros(self.space_action_size)
        self.directory = 'Data/SARSA_QT/'

        # Save Args in text file
        with open('Data/SARSA_QT/RunInfo/' + self.strat_label + self.label + '_Info.txt','w') as data: data.write(str(args))

    def save_model(self):
        pickle.dump([self.train_history,self.Q_val], open(self.directory + self.strat_label + self.label + '_Model.p', 'wb' ))

    def load_model(self):
        [self.train_history,self.Q_val] = pickle.load(open(self.directory + self.strat_label + self.label + '_Model.p', 'rb' ))

    def get_action(self,state):
        if (state[1] == 0): return 0

        sample_epsilon = np.random.uniform(0,1)
        if (sample_epsilon < self.epsilon): return np.random.randint(1,state[1]+1)

        return np.argmax(self.Q_val[state[0],state[1],state[2],0:state[1]]) + 1

    def plot(self,type,save_data):
        if type == 'train':
            win100 = np.flip(np.convolve(np.flip(self.train_history), np.ones(100), mode='valid'))
            win_avg = np.flip(np.convolve(np.flip(win100), np.ones(self.mean_window)/self.mean_window, mode='valid'))

            fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(15,5.5), facecolor='w', edgecolor='k')
            fig.suptitle('SARSA Q-Table {} Agent -  Last {:,} Avg : {:.2f}%'.format(self.strat_label,self.mean_window,win_avg[-1]),fontweight='bold',fontsize = 16)

            ax1.plot(range(99,self.n),win100,c='b')
            ax1.set_title('Running 100 Game Win%',fontweight='bold',fontsize = 14)
            ax1.set_xlabel('Episode',fontweight='bold',fontsize = 12)
            ax1.set_ylabel('Last 100 Win %',fontweight='bold',fontsize = 12)
            ax1.grid(True, color='w', linestyle='-', linewidth=1)

            ax2.plot(range(99+self.mean_window-1,self.n),win_avg,c='g')
            ax2.set_title('Running {:,} Avg Win%'.format(self.mean_window), fontweight='bold',fontsize = 14)
            ax2.set_xlabel('Episode', fontweight='bold',fontsize = 12)
            ax2.set_ylabel('Mean Win %', fontweight='bold',fontsize = 12)
            ax2.grid(True, color='w', linestyle='-', linewidth=1)

            if (self.opp_strategy == 5 and self.opp_freq > 1):
                ax3 = ax1.twinx()
                ax3.plot(range(100,self.n),self.opp_strat_vector[100:] , 'r-',alpha=0.3)
                ax3.set_ylabel('Opponent Strategy', color='r')

            if save_data: fig.savefig(self.directory + 'TrainResults/' + self.strat_label + self.label + '.png')
            else: plt.show()
            plt.close()

        if type == 'test':
            if (self.opponent_source == 'Fixed'): xlabels = ['Mean','Long','Short','Naive','Dynamic']
            else: xlabels = ['1','2','3','4','Dynamic']

            fig2,ax4 = plt.subplots(1,1,figsize=(10,6))
            ax4.bar(xlabels,self.test_win_percent)
            ax4.set_title('SARSA Q-Table {} Agent Win %\n Number Test Episodes: {}'.format(self.strat_label,self.n_test),fontweight='bold',fontsize = 15)
            ax4.set_xlabel('Opponent Strategies',fontweight='bold',fontsize = 12)
            ax4.set_ylabel('Win %',fontweight='bold',fontsize = 12)
            for i,y in enumerate(self.test_win_percent):
                    ax4.text(i, y, y, ha = 'center',fontweight='bold',fontsize = 12)

            if save_data: fig2.savefig(self.directory + 'TestResults/' + self.strat_label + self.label + '.png')
            else: plt.show()
            plt.close()

    def train(self,save_data=True):
        print('Training:')
        start_time = time.time()
        env = PaperTennisEnv(gamespace = self.gamespace,
                             opponent_strategy = self.opp_strategy,
                             opponent_source = self.opponent_source)

        wins = np.zeros(self.n)

        #Lopp through episodes
        for episode in tqdm(range(self.n),ncols=100):

            state = env.reset(self.opp_strat_vector[episode])
            action = self.get_action(state)
            
            done = False

            # Episode
            while not done:
                # Step Environment
                state_prime, reward, done, _ = env.step(action)

                # Get Action e-greedy 
                action_prime = self.get_action(state_prime)

                # SARSA tabular update
                td_update = (reward + self.gamma * self.Q_val[state_prime[0],state_prime[1],state_prime[2],action_prime-1]
                    -self.Q_val[state[0],state[1],state[2],action-1])

                self.Q_val[state[0],state[1],state[2],action-1] = self.Q_val[state[0],state[1],state[2],action-1] + self.alpha * td_update
                
                # Update state/action
                state  = state_prime
                action = action_prime

            wins[episode] = int(reward > 0)
        
        # Update total episode count
        self.train_history = np.concatenate((self.train_history,wins))

        if save_data: self.save_model()

        self.plot('train',save_data)

        # Print run details
        run_time = time.time() - start_time
        if run_time < 60: print('--- {:.2f} seconds ---'.format(run_time))
        else: print('--- {:.2f} minutes ---'.format(run_time/60))
        with open('Data/SARSA_QT/RunInfo/' + self.strat_label + self.label + '_Info.txt','a') as data: data.write('\n\nRun Time: {:.2f} seconds'.format(run_time))
        
        return self.train_history

    def evaluate(self,save_data=True):
        print('Evaluating:')
        test_wins = np.zeros((5,self.n_test))

        env = PaperTennisEnv(gamespace = self.gamespace,
                             opponent_strategy = self.opp_strategy,
                             opponent_source = self.opponent_source)

        # Run Through Strategies
        for opp_strategy in tqdm(range(1,6),ncols=100):

            if opp_strategy < 5: OPP_Strat = np.repeat(opp_strategy, self.n_test)
            else: OPP_Strat = np.random.randint(1,5,self.n_test)

            for episode in range(self.n_test):
                state = env.reset(OPP_Strat[episode])
                done = False
                while not done:
                    action = self.get_action(state)
                    state, reward, done, _ = env.step(action)

                test_wins[opp_strategy-1,episode] = int(reward > 0)

        self.test_win_percent = np.around(np.count_nonzero(test_wins,axis=1)/self.n_test * 100,2)
        self.plot('test',save_data)

        return self.test_win_percent


# Monte Carlo Agent
class MonteCarlo_QTable():
    '''
    Description: Paper Tennis Reinforcement Learning agent utilizing the tabular Q-value Monte Carlo
    method as specified in Sutton and Barto, Section 5.

    Source: Sutton, R. S., Barto, A. G. (2018 ). Reinforcement Learning: An Introduction. The MIT Press

    Attributes
    ----------

    Methods
    ----------
    '''

    def __init__(self,args,gamespace= (7,50),opponent_source = 'Fixed'):
        super(MonteCarlo_QTable, self).__init__()
        self.n = args['NUM_EPISODES']
        self.gamma = args['GAMMA']
        self.alpha = args['ALPHA']
        self.epsilon = args['EPSILON']
        self.opp_strategy = args['OPP_STRAT']
        self.opp_freq = args['OPP_FREQ']
        self.label = args['add_label']
        self.mean_window = args['AVG_WINDOW']
        self.n_test = args['TEST_EPISODES']
        self.gamespace = gamespace
        self.opponent_source = opponent_source
        self.train_history = np.array([])
        self.FixedStrategies = {
            1: 'Mean',
            2: 'Long',
            3: 'Short',
            4: 'Naive',
            5: 'Dynamic'
        }
        if (self.opponent_source == 'Fixed'): self.strat_label = self.FixedStrategies[self.opp_strategy]
        else: self.strat_label = str(self.opp_strategy)

        # Generate opponent strategy vector
        if (self.opp_strategy) == 5:
            self.opp_strat_vector = np.random.randint(1,5,int(self.n/self.opp_freq))
            self.opp_strat_vector = np.repeat(self.opp_strat_vector, self.opp_freq)
        else: 
            self.opp_strat_vector = np.repeat(self.opp_strategy,self.n)

        # Monte Carlo Specific
        self.space_action_size = [gamespace[0]-2,gamespace[1]+1,gamespace[1]+1,gamespace[1]]  
        self.Q_val = np.zeros(self.space_action_size)
        self.directory = 'Data/MonteCarlo_QT/'

        # Save Args in text file
        with open('Data/MonteCarlo_QT/RunInfo/' + self.strat_label + self.label + '_Info.txt','w') as data: data.write(str(args))

    def save_model(self):
        pickle.dump([self.train_history,self.Q_val], open(self.directory + self.strat_label + self.label + '_Model.p', 'wb' ))

    def load_model(self):
        [self.train_history,self.Q_val] = pickle.load(open(self.directory + self.strat_label + self.label + '_Model.p', 'rb' ))

    def get_action(self,state):
        if (state[1] == 0): return 0

        sample_epsilon = np.random.uniform(0,1)
        if (sample_epsilon < self.epsilon): return np.random.randint(1,state[1]+1)

        return np.argmax(self.Q_val[state[0],state[1],state[2],0:state[1]]) + 1

    def plot(self,type,save_data):
        if type == 'train':
            win100 = np.flip(np.convolve(np.flip(self.train_history), np.ones(100), mode='valid'))
            win_avg = np.flip(np.convolve(np.flip(win100), np.ones(self.mean_window)/self.mean_window, mode='valid'))

            fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(15,5.5), facecolor='w', edgecolor='k')
            fig.suptitle('Monte Carlo Q-Table {} Agent -  Last {:,} Avg : {:.2f}%'.format(self.strat_label,self.mean_window,win_avg[-1]),fontweight='bold',fontsize = 16)

            ax1.plot(range(99,self.n),win100,c='b')
            ax1.set_title('Running 100 Game Win%',fontweight='bold',fontsize = 14)
            ax1.set_xlabel('Episode',fontweight='bold',fontsize = 12)
            ax1.set_ylabel('Last 100 Win %',fontweight='bold',fontsize = 12)
            ax1.grid(True, color='w', linestyle='-', linewidth=1)

            ax2.plot(range(99+self.mean_window-1,self.n),win_avg,c='g')
            ax2.set_title('Running {:,} Avg Win%'.format(self.mean_window), fontweight='bold',fontsize = 14)
            ax2.set_xlabel('Episode', fontweight='bold',fontsize = 12)
            ax2.set_ylabel('Mean Win %', fontweight='bold',fontsize = 12)
            ax2.grid(True, color='w', linestyle='-', linewidth=1)

            if (self.opp_strategy == 5 and self.opp_freq > 1):
                ax3 = ax1.twinx()
                ax3.plot(range(100,self.n),self.opp_strat_vector[100:] , 'r-',alpha=0.3)
                ax3.set_ylabel('Opponent Strategy', color='r')

            if save_data: fig.savefig(self.directory + 'TrainResults/' + self.strat_label + self.label + '.png')
            else: plt.show()
            plt.close()

        if type == 'test':
            if (self.opponent_source == 'Fixed'): xlabels = ['Mean','Long','Short','Naive','Dynamic']
            else: xlabels = ['1','2','3','4','Dynamic']

            fig2,ax4 = plt.subplots(1,1,figsize=(10,6))
            ax4.bar(xlabels,self.test_win_percent)
            ax4.set_title('Monte Carlo Q-Table {} Agent Win %\n Number Test Episodes: {}'.format(self.strat_label,self.n_test),fontweight='bold',fontsize = 15)
            ax4.set_xlabel('Opponent Strategies',fontweight='bold',fontsize = 12)
            ax4.set_ylabel('Win %',fontweight='bold',fontsize = 12)
            for i,y in enumerate(self.test_win_percent):
                    ax4.text(i, y, y, ha = 'center',fontweight='bold',fontsize = 12)

            if save_data: fig2.savefig(self.directory + 'TestResults/' + self.strat_label + self.label + '.png')
            else: plt.show()
            plt.close()

    def train(self,save_data=True):
        print('Training:')
        start_time = time.time()
        env = PaperTennisEnv(gamespace = self.gamespace,
                             opponent_strategy = self.opp_strategy,
                             opponent_source = self.opponent_source)

        wins = np.zeros(self.n)

        Returns = np.zeros(self.space_action_size,dtype=object)
        Counters = np.zeros(self.space_action_size,dtype=object)

        for episode in tqdm(range(self.n),ncols=100):

            state = env.reset(self.opp_strat_vector[episode])
            
            done = False

            # Generate an episode
            state_list = list()
            action_list = list()
            reward_list = list()

            # Generate episode of data
            while not done:
                
                state_list.append(state)
                
                action = self.get_action(state)
                
                state, reward, done, _ = env.step(action)
                
                action_list.append(action)
                reward_list.append(reward)

            wins[episode] = int(reward > 0)

            G = 0
            # Update tables for MC estimation (Every visit)
            for t in range(0,len(state_list)):
                G = self.gamma * G + reward_list[t]
                Returns[state_list[t][0],state_list[t][1],state_list[t][2],action_list[t]-1] += G
                Counters[state_list[t][0],state_list[t][1],state_list[t][2],action_list[t]-1] += 1
                self.Q_val[state_list[t][0],state_list[t][1],state_list[t][2],action_list[t]-1] = \
                    Returns[state_list[t][0],state_list[t][1],state_list[t][2],action_list[t]-1]/ \
                    Counters[state_list[t][0],state_list[t][1],state_list[t][2],action_list[t]-1]
        
        # Update total episode count
        self.train_history = np.concatenate((self.train_history,wins))

        if save_data: self.save_model()

        self.plot('train',save_data)

        # Print run details
        run_time = time.time() - start_time
        if run_time < 60: print('--- {:.2f} seconds ---'.format(run_time))
        else: print('--- {:.2f} minutes ---'.format(run_time/60))
        with open('Data/MonteCarlo_QT/RunInfo/' + self.strat_label + self.label + '_Info.txt','a') as data: data.write('\n\nRun Time: {:.2f} seconds'.format(run_time))
        
        return self.train_history

    def evaluate(self,save_data=True):
        print('Evaluating:')
        test_wins = np.zeros((5,self.n_test))

        env = PaperTennisEnv(gamespace = self.gamespace,
                             opponent_strategy = self.opp_strategy,
                             opponent_source = self.opponent_source)

        # Run Through Strategies
        for opp_strategy in tqdm(range(1,6),ncols=100):

            if opp_strategy < 5: OPP_Strat = np.repeat(opp_strategy, self.n_test)
            else: OPP_Strat = np.random.randint(1,5,self.n_test)

            for episode in range(self.n_test):
                state = env.reset(OPP_Strat[episode])
                done = False
                while not done:
                    action = self.get_action(state)
                    state, reward, done, _ = env.step(action)

                test_wins[opp_strategy-1,episode] = int(reward > 0)

        self.test_win_percent = np.around(np.count_nonzero(test_wins,axis=1)/self.n_test * 100,2)
        self.plot('test',save_data)

        return self.test_win_percent


# Define Actor and Critic DNNs
class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, self.action_size-1)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        output = self.linear3(output)
        distribution = Categorical(F.softmax(output, dim=-1))
        return distribution

class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, 1)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        value = self.linear3(output)
        return value

# Policy Gradient Actor-Critic DQN Approx
class PGAC_DNN():

    '''
    Description: Paper Tennis Reinforcement Learning agent utilizing an Actor-Critic Policy Gradient
    method with deep Q network function approximation as specified in Sutton and Barto, Section 13.6.

    Source: Sutton, R. S., Barto, A. G. (2018 ). Reinforcement Learning: An Introduction. The MIT Press

    Attributes
    ----------

    Methods
    ----------
    '''

    def __init__(self,args,gamespace= (7,50),opponent_source = 'Fixed'):
        super(PGAC_DNN, self).__init__()
        self.n = args['NUM_EPISODES']
        self.gamma = args['GAMMA']
        self.alpha = args['ALPHA']
        self.epsilon = args['EPSILON']
        self.opp_strategy = args['OPP_STRAT']
        self.opp_freq = args['OPP_FREQ']
        self.label = args['add_label']
        self.mean_window = args['AVG_WINDOW']
        self.n_test = args['TEST_EPISODES']
        self.check_stable = args['CHECK_STABLE']
        self.gamespace = gamespace
        self.opponent_source = opponent_source
        self.train_history = np.array([])
        self.FixedStrategies = {
            1: 'Mean',
            2: 'Long',
            3: 'Short',
            4: 'Naive',
            5: 'Dynamic'
        }
        if (self.opponent_source == 'Fixed'): self.strat_label = self.FixedStrategies[self.opp_strategy]
        else: self.strat_label = str(self.opp_strategy)

        # Generate opponent strategy vector
        if (self.opp_strategy) == 5:
            self.opp_strat_vector = np.random.randint(1,5,int(self.n/self.opp_freq))
            self.opp_strat_vector = np.repeat(self.opp_strat_vector, self.opp_freq)
        else: 
            self.opp_strat_vector = np.repeat(self.opp_strategy,self.n)

        # PGAC Specific 
        env = PaperTennisEnv(gamespace = self.gamespace)
        state_size = len(env.observation_space.spaces)
        action_size = env.action_space.n
        self.actor = Actor(state_size, action_size)
        self.critic = Critic(state_size, action_size)
        self.directory = 'Data/PGAC_DNN/'

        # Save Args in text file
        with open('Data/PGAC_DNN/RunInfo/' + self.strat_label + self.label + '_Info.txt','w') as data: data.write(str(args))

    def save_model(self):
        torch.save(self.actor.state_dict(),self.directory + self.strat_label + self.label + '_Actor.pt')
        torch.save(self.critic.state_dict(),self.directory + self.strat_label + self.label + '_Critic.pt')
        pickle.dump(self.train_history, open(self.directory + self.strat_label + self.label + '_TrainHistory.p', 'wb' ))

    def load_model(self):
        actor_model = torch.load(self.directory + self.strat_label + self.label + '_Actor.pt')
        critic_model = torch.load(self.directory + self.strat_label + self.label + '_Critic.pt')
        self.actor.load_state_dict(actor_model)
        self.critic.load_state_dict(critic_model)
        self.train_history = pickle.load(open(self.directory + self.strat_label + self.label + '_TrainHistory.p', 'rb' ))

    def reset_model(self):
        env = PaperTennisEnv(gamespace = self.gamespace)
        state_size = len(env.observation_space.spaces)
        action_size = env.action_space.n
        self.actor = Actor(state_size, action_size)
        self.critic = Critic(state_size, action_size)

    def get_action(self,state):
        if (state[1] == 0): return 0
        s = torch.FloatTensor(state)
        dist = self.actor(s)
        return  int(dist.sample().cpu().numpy()*state[1]/51)+1

    def plot(self,type,save_data,iters=1):
        if type == 'train':
            win100 = np.flip(np.convolve(np.flip(self.train_history), np.ones(100), mode='valid'))
            win_avg = np.flip(np.convolve(np.flip(win100), np.ones(self.mean_window)/self.mean_window, mode='valid'))

            fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(15,5.5), facecolor='w', edgecolor='k')
            fig.suptitle('PGAC_DNN {} Agent ({} Iterations) -  Last {:,} Avg : {:.2f}%'.format(self.strat_label,iters,self.mean_window,win_avg[-1]),fontweight='bold',fontsize = 15)

            ax1.plot(range(99,self.n),win100,c='b')
            ax1.set_title('Running 100 Game Win%',fontweight='bold',fontsize = 14)
            ax1.set_xlabel('Episode',fontweight='bold',fontsize = 12)
            ax1.set_ylabel('Last 100 Win %',fontweight='bold',fontsize = 12)
            ax1.grid(True, color='w', linestyle='-', linewidth=1)

            ax2.plot(range(99+self.mean_window-1,self.n),win_avg,c='g')
            ax2.set_title('Running {:,} Avg Win%'.format(self.mean_window), fontweight='bold',fontsize = 14)
            ax2.set_xlabel('Episode', fontweight='bold',fontsize = 12)
            ax2.set_ylabel('Mean Win %', fontweight='bold',fontsize = 12)
            ax2.grid(True, color='w', linestyle='-', linewidth=1)

            if (self.opp_strategy == 5 and self.opp_freq > 1):
                ax3 = ax1.twinx()
                ax3.plot(range(100,self.n),self.opp_strat_vector[100:] , 'r-',alpha=0.3)
                ax3.set_ylabel('Opponent Strategy', color='r')

            if save_data: fig.savefig(self.directory + 'TrainResults/' + self.strat_label + self.label + '.png')
            else: plt.show()
            plt.close()

        if type == 'test':
            if (self.opponent_source == 'Fixed'): xlabels = ['Mean','Long','Short','Naive','Dynamic']
            else: xlabels = ['1','2','3','4','Dynamic']

            fig2,ax4 = plt.subplots(1,1,figsize=(10,6))
            ax4.bar(xlabels,self.test_win_percent)
            ax4.set_title('PGAC_DNN {} Agent Win %\n Number Test Episodes: {}'.format(self.strat_label,self.n_test),fontweight='bold',fontsize = 15)
            ax4.set_xlabel('Opponent Strategies',fontweight='bold',fontsize = 12)
            ax4.set_ylabel('Win %',fontweight='bold',fontsize = 12)
            for i,y in enumerate(self.test_win_percent):
                    ax4.text(i, y, y, ha = 'center',fontweight='bold',fontsize = 12)

            if save_data: fig2.savefig(self.directory + 'TestResults/' + self.strat_label + self.label + '.png')
            else: plt.show()
            plt.close()

    def compute_returns(self,next_value, rewards, masks, gamma):
        R = next_value
        returns = []
        for step in reversed(range(len(rewards))):
            R = rewards[step] + gamma * R * masks[step]
            returns.insert(0, R)
        return returns

    def train(self,save_data=True):
        train_iters = 0
        start_time = time.time()

        while (self.check_stable == True or train_iters < 2):

            optimizerA = optim.Adam(self.actor.parameters(),lr=self.alpha)
            optimizerC = optim.Adam(self.critic.parameters(),lr=self.alpha)

            print('Training Iteration {}:'.format(train_iters+1))
            
            env = PaperTennisEnv(gamespace = self.gamespace,
                                opponent_strategy = self.opp_strategy,
                                opponent_source = self.opponent_source)

            wins = np.zeros(self.n)

            # Loop through episodes
            for episode in tqdm(range(self.n),ncols=100):

                state = env.reset(self.opp_strat_vector[episode])
                
                done = False

                log_probs = []
                values = []
                rewards = []
                masks = []
                entropy = 0

                # Episode
                while not done:
                    state = torch.FloatTensor(state)
                    dist, value = self.actor(state), self.critic(state)

                    action = dist.sample()
                    if env.state[1] > 0: true_action = int(action.cpu().numpy()*env.state[1]/51)+1
                    else: true_action = 0
                    next_state, reward, done, _ = env.step(true_action)

                    log_prob = dist.log_prob(action).unsqueeze(0)
                    entropy += dist.entropy().mean()

                    log_probs.append(log_prob)
                    values.append(value)
                    rewards.append(torch.tensor([reward], dtype=torch.float))
                    masks.append(torch.tensor([1-done], dtype=torch.float))

                    state = next_state

                next_state = torch.FloatTensor(next_state)
                next_value = self.critic(next_state)
                returns = self.compute_returns(next_value, rewards, masks,self.gamma)

                log_probs = torch.cat(log_probs)
                returns = torch.cat(returns).detach()
                values = torch.cat(values)

                advantage = returns - values

                actor_loss = -(log_probs * advantage.detach()).mean()
                critic_loss = advantage.pow(2).mean()

                optimizerA.zero_grad()
                optimizerC.zero_grad()
                actor_loss.backward()
                critic_loss.backward()
                optimizerA.step()
                optimizerC.step()

                wins[episode] = int(reward > 0)

            train_iters += 1
            if self.check_stable == True:
                win100 = np.flip(np.convolve(np.flip(wins), np.ones(100), mode='valid'))
                win_avg = np.flip(np.convolve(np.flip(win100), np.ones(self.mean_window)/self.mean_window, mode='valid'))
                if win_avg[-1] > 75: self.check_stable = False
                else:
                    print('Unstable run ({:.2f}%) ...'.format(win_avg[-1]))
                    self.reset_model()

        # Update total episode count
        self.train_history = np.concatenate((self.train_history,wins))

        if save_data: self.save_model()

        self.plot('train',save_data,iters = train_iters)

        # Print run details
        run_time = time.time() - start_time
        print('Iterations: {}'.format(train_iters))
        if run_time < 60: print('--- {:.2f} seconds ---'.format(run_time))
        else: print('--- {:.2f} minutes ---'.format(run_time/60))
        with open('Data/PGAC_DNN/RunInfo/' + self.strat_label + self.label + '_Info.txt','a') as data: 
            data.write('\n\nRun Time: {:.2f} seconds\n\nTrain Iteration: {}'.format(run_time,train_iters))
        
        return self.train_history

    def evaluate(self,save_data=True):
        print('Evaluating:')
        test_wins = np.zeros((5,self.n_test))

        env = PaperTennisEnv(gamespace = self.gamespace,
                             opponent_strategy = self.opp_strategy,
                             opponent_source = self.opponent_source)

        # Run Through Strategies
        for opp_strategy in tqdm(range(1,6),ncols=100):

            if opp_strategy < 5: OPP_Strat = np.repeat(opp_strategy, self.n_test)
            else: OPP_Strat = np.random.randint(1,5,self.n_test)

            for episode in range(self.n_test):
                state = env.reset(OPP_Strat[episode])
                done = False
                while not done:
                    action = self.get_action(state)
                    state, reward, done, _ = env.step(action)

                test_wins[opp_strategy-1,episode] = int(reward > 0)

        self.test_win_percent = np.around(np.count_nonzero(test_wins,axis=1)/self.n_test * 100,2)
        self.plot('test',save_data)

        return self.test_win_percent


