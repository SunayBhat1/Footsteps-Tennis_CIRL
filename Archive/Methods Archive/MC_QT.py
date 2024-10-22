# Monte Carlo methods

from PaperTennis_env import PaperTennisEnv
import gym, os
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
import os.path
from os import path
from tqdm import tqdm

#Hyperparms
STRATEGY = 'Naive'
add_info = ''
GAMMA = 0.9
ALPHA = 0.1
E_GREEDY = 0.1
NUM_EPISODES = 1000000
OPP_FREQ = 1
mean_window = 10000

# Get Strategy index
Strategies = {
    "Mean": 1,
    "Long": 2,
    "Short": 3,
    "Naive": 4,
    "Random": 5
}

# Generate opponent strategy vector
if Strategies[STRATEGY] == 5:
    OPP_Strat = np.around(np.random.uniform(1,4,int(NUM_EPISODES/OPP_FREQ)))
    OPP_Strat = np.repeat(OPP_Strat, OPP_FREQ)
else: 
    OPP_Strat = np.repeat(Strategies[STRATEGY], NUM_EPISODES)

# Get action e-greedy
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

## Train
start_time = time.time()

# Init Q table and train history
Q_val = np.zeros([5,51,51,50])
train_episodes = 0

# init env
env = PaperTennisEnv()

# Plotting Stuff
trend_wins = np.zeros(NUM_EPISODES-100)
wins = np.zeros(NUM_EPISODES)

Returns = np.zeros([5,51,51,50],dtype=object)
Counters = np.zeros([5,51,51,50],dtype=object)

for episode in tqdm(range(NUM_EPISODES)):

    env.reset()
    
    done = False

    # Generate an episode
    state_list = list()
    action_list = list()
    reward_list = list()
    total_reward = 0
    error_episode = 0
    
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
        
# Save Q-val Table and episode count
pickle.dump([train_episodes,Q_val], open('TrainedModels/' + STRATEGY + '_MC_QT' + add_info + '.p', "wb" ))

# Print run details
run_time = time.time() - start_time
if run_time > 60: print("--- {:.2f} seconds ---".format(run_time))
else: print("--- {:.2f} minutes ---".format(run_time/60))
print("Final mean win percent = {:.2f}%".format(np.mean(trend_wins[-10000:])))
print("Total Episodes {} ".format(train_episodes))


## Plotting
add_info = ''

fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(14,6), dpi= 150, facecolor='w', edgecolor='k')
fig.suptitle('MC Q-Table {} Agent\n Last 10k: {:.2f}% Wins'.format(STRATEGY,(np.mean(trend_wins[-10000:]))),fontweight='bold',fontsize = 16)

ax1.plot(range(100,NUM_EPISODES),trend_wins)
ax1.set_title('Running 100 Win%',fontweight='bold',fontsize = 15)
ax1.set_xlabel('Episode',fontweight='bold',fontsize = 12)
ax1.set_ylabel('Win %',fontweight='bold',fontsize = 12)
ax1.grid()

ax2.plot(range(mean_window-1,len(trend_wins)), np.convolve(trend_wins, np.ones(mean_window)/mean_window, mode='valid'),c='g')
ax2.set_title('Moving 10000 Avg Win%', fontweight='bold',fontsize = 12)
ax2.set_xlabel('Episode', fontweight='bold',fontsize = 12)
ax2.set_ylabel('Mean Win %', fontweight='bold',fontsize = 12)
ax2.grid()

if Strategies[STRATEGY] == 5:
    ax3 = ax1.twinx()
    ax3.plot(range(100,NUM_EPISODES),OPP_Strat[100:] , 'r-',alpha=0.3)
    ax3.set_ylabel('Strategy', color='r')

fig.savefig('TrainResults/' + STRATEGY + '_MC_QT_Train' + add_info + '.png')
plt.show()