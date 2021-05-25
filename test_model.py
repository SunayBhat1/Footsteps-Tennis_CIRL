# test render
from PaperTennis_env import PaperTennisEnv
import matplotlib.pyplot as plt
import numpy as np
import pickle

#HParms
NUM_EPISODES = 1000
RENDER = False
NUM_STRATS = 4
STRAT = 'Short'
METHOD = 'SARSA_QT'
Strategy_Shift = 1

# Setup opponent strategy
Startegies = {
    "Mean": 1,
    "Long": 2,
    "Short": 3,
    "Rand": 4,
    "Combo": 5
}

# Get Opponent Startegy Vector
if Startegies[STRAT] < 5:
    OPP_Strat = np.repeat(Startegies[STRAT], NUM_EPISODES)
else:
    OPP_Strat = np.around(np.random.uniform(0,3,int(NUM_EPISODES/Strategy_Shift))+1)
    OPP_Strat = np.repeat(OPP_Strat, Strategy_Shift)


# Load a Strategy
train_episodes, Q_val = pickle.load(open('TrainedModels/' + STRAT + '_' + METHOD + '.p', "rb" ))

# Get action
def get_action(state,Q_val):
    if state[1] == 0:
        return 0

    q_s = np.zeros(state[1])

    for i in range(1,state[1]):
        q_s[i] = Q_val[state[0],state[1],state[2],i]

    return np.argmax(q_s) + 1


# Create Environment and wins vector
env = PaperTennisEnv()
wins = np.zeros((NUM_STRATS,NUM_EPISODES))

# Setup Plot

for opp_strategy in range(1,NUM_STRATS+1):

    for episode in range(NUM_EPISODES):

        env.reset()
        state = env.state
        done = False
        while not done:
            if RENDER: env.render()
            action = get_action(state,Q_val)

            state, reward, done,_ = env.step(action,opp_strategy)

        if (reward == 1):
            wins[opp_strategy-1,episode] = 1

env.reset()

x = ['Mean','Long','Short','Rand']
win_percent = np.around(np.count_nonzero(wins,axis=1)/np.shape(wins)[1] * 100,2)

plt.style.use('bmh')
plt.figure(1,figsize=(10,6), dpi=100,facecolor='w', edgecolor='k')
plt.bar(x,win_percent)
plt.title('{} Trained Agent Win Percent\n Test Episodes: {}'.format(STRAT,NUM_EPISODES),fontweight='bold',fontsize = 15)
plt.xlabel('Opponent Strategies',fontweight='bold',fontsize = 12)
plt.ylabel('Win %',fontweight='bold',fontsize = 12)
for i,y in enumerate(win_percent):
        plt.text(i, y, y, ha = 'center',fontweight='bold',fontsize = 12)

plt.savefig('TestResults/' + STRAT + '_Agent_' + METHOD + '.png')
plt.show()