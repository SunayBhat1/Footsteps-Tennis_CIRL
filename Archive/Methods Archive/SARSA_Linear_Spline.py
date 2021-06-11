from PaperTennis_env import PaperTennisEnv
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
import os.path
from os import path
from tqdm import tqdm

#Hyperparms
#Hyperparms
STRATEGY = 'Random'
add_info = '_f1000'
GAMMA = 1
ALPHA = 0.01
E_GREEDY = 0.05
NUM_EPISODES = 10000
OPP_FREQ = 1000
mean_window = 100


# Get Strategy index
Strategies = {
    'Mean': 1,
    'Long': 2,
    'Short': 3,
    'Naive': 4,
    'Random': 5
}

# Generate opponent strategy vector
if Strategies[STRATEGY] == 5:
    OPP_Strat = np.around(np.random.uniform(1,4,int(NUM_EPISODES/OPP_FREQ)))
    OPP_Strat = np.repeat(OPP_Strat, OPP_FREQ)
else: 
    OPP_Strat = np.repeat(Strategies[STRATEGY], NUM_EPISODES)



# Function approx to compute
def get_features(state,action):
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
        if (state[1]-state[2]) in spline: feat_vec[i+8] = 1
    
    # Activate action indicator
    for i,spline in enumerate(action_splines): 
        if action in spline: feat_vec[i+22] = 1
        
    return feat_vec

def Q_value(state,action,linear_weights):

    feat_vec = get_features(state,action)
    return (linear_weights @ feat_vec).item()

# Get action e-greedy
def get_action(linear_weights, state):
    if state[1] == 0: return 0
    
    p_epsilon = np.random.uniform(0,1)
    if p_epsilon < E_GREEDY: return np.argmax(np.random.uniform(0,1,state[1])) + 1

    q_s = np.zeros(state[1])

    for i in range(0,state[1]): q_s[i] = Q_value(state,i,linear_weights)

    return np.argmax(q_s) + 1


start_time = time.time()

# Game Score Splined weights
w = np.random.rand(5,28)
train_episodes = 0

# Init Env
env = PaperTennisEnv()

# Plotting Stuff
trend_wins = np.zeros(NUM_EPISODES-100)
trend_TD = np.zeros(NUM_EPISODES-100)

wins = np.zeros(NUM_EPISODES)
td_error = np.zeros(NUM_EPISODES)

for episode in  tqdm(range(NUM_EPISODES),ncols=100):

    env.reset()
    state = env.state
    done = False

    action = get_action(w[state[0],:], state)

    # Generate an episode
    error_episode = 0
    while not done:

        # Step Environment
        state_prime, reward, done,_ = env.step(action,OPP_Strat[episode])

        # Get Action e-greedy
        action_prime = get_action(w[state_prime[0],:], state_prime)

        if action_prime>state_prime[1]: print('Error! State: {}, Action {}'.format(state_prime,action_prime))
        
        # Linear SARSA update (Section 10.1, psuedocode) 
        td_update = (reward + GAMMA * Q_value(state_prime,action_prime,w[state_prime[0],:])-
                     Q_value(state,action,w[state[0],:])) * get_features(state,action)      
        w[state[0],:] = w[state[0],:] + ALPHA*td_update

        error_episode += np.sum(td_update)

        state  = state_prime
        action = action_prime

    if reward == 1: wins[episode] = 1        

    # Track progress
    td_error[episode] = error_episode
    if episode >= 100: trend_wins[episode-100] = np.sum(wins[episode-100:episode])    

# Update total episode count
train_episodes += episode +1

# Save Q-val Table and episode count
pickle.dump([train_episodes,w], open('TrainedModels/' + STRATEGY + '_SARSA_LS' + add_info + '.p', 'wb' ))

## Plotting

fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(14,6), dpi= 150, facecolor='w', edgecolor='k')
fig.suptitle('SARSA Linear Spline {} Agent\n Last 10k: {:.2f}% Wins'.format(STRATEGY,(np.mean(trend_wins[-100:]))),fontweight='bold',fontsize = 16)

ax1.plot(range(100,NUM_EPISODES),trend_wins)
ax1.set_title('Running 100 Win%',fontweight='bold',fontsize = 15)
ax1.set_xlabel('Episode',fontweight='bold',fontsize = 12)
ax1.set_ylabel('Win %',fontweight='bold',fontsize = 12)
ax1.grid()

ax2.plot(range(mean_window-1,len(trend_wins)), np.convolve(trend_wins, np.ones(mean_window)/mean_window, mode='valid'),c='g')
ax2.set_title('Moving 100 Avg Win%', fontweight='bold',fontsize = 12)
ax2.set_xlabel('Episode', fontweight='bold',fontsize = 12)
ax2.set_ylabel('Mean Win %', fontweight='bold',fontsize = 12)
ax2.grid()

if Strategies[STRATEGY] == 5:
    ax3 = ax1.twinx()
    ax3.plot(range(100,NUM_EPISODES),OPP_Strat[100:] , 'r-',alpha=0.3)
    ax3.set_ylabel('Strategy', color='r')

fig2, ax4 = plt.subplots(figsize=(8,4.5), dpi= 120, facecolor='w', edgecolor='k')
ax4.plot(range(mean_window-1,len(td_error)), np.convolve(td_error, np.ones(mean_window)/mean_window, mode='valid'),c='r')
ax4.set_title('TD Error (Moving 100 Avg Window)', fontweight='bold',fontsize = 15)
ax4.set_xlabel('Episode', fontweight='bold',fontsize = 12)
ax4.set_ylabel('TD Error', fontweight='bold',fontsize = 12)
ax4.grid()

fig.savefig('TrainResults/' + STRATEGY + '_SARSA_LS_Train' + add_info + '.png')
fig2.savefig('TrainResults/' + STRATEGY + '_SARSA_LS_TD' + add_info + '.png')
plt.show()