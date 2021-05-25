# test render
from PaperTennis_env import PaperTennisEnv
from AC_NN import Actor
import torch
import matplotlib.pyplot as plt
import numpy as np

#HParms 
NUM_EPISODES = 1000
RENDER = False
STRAT = 'Random'
METHOD = 'Actor_NN'
add_info = ''
Strategy_Shift = 1

# Load a Strategy
# train_episodes, Q_val = pickle.load(open('TrainedModels/' + STRAT + '_' + METHOD + add_info + '.p', "rb" ))
# print('Episodes Trained {}'.format(train_episodes))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = PaperTennisEnv()
state_size = len(env.observation_space.spaces)
action_size = env.action_space.n
model = torch.load('TrainedModels/' + STRAT + '_' +  METHOD + '.pkl')
actor = Actor(state_size, action_size).to(device)
actor.load_state_dict(model)

# Get action
def get_action_QT(state,Q_val):
    if state[1] == 0:
        return 0

    q_s = np.zeros(state[1])

    for i in range(1,state[1]):
        q_s[i] = Q_val[state[0],state[1],state[2],i]

    return np.argmax(q_s) + 1

def get_action_AC(state,actor):
    s = torch.FloatTensor(state).to(device)
    dist = actor(s)
    return dist.sample()

# Create Environment and wins vector 
wins = np.zeros((5,NUM_EPISODES))

# Setup Plot

for opp_strategy in range(1,6):

    # Get Opponent Startegy Vector
    if opp_strategy < 5:
        OPP_Strat = np.repeat(opp_strategy, NUM_EPISODES)
    else:
        OPP_Strat = np.around(np.random.uniform(1,4,NUM_EPISODES))

    for episode in range(NUM_EPISODES):

        env.reset()
        state = env.state
        done = False
        while not done:
            if RENDER: env.render()
            # action = get_action_QT(state,Q_val)
            action = get_action_AC(state,actor)

            state, reward, done,_ = env.step(action,OPP_Strat[episode])

        if (reward == 1): wins[opp_strategy-1,episode] = 1

env.reset()

x = ['Mean','Long','Short','Naive','Random']
win_percent = np.around(np.count_nonzero(wins,axis=1)/np.shape(wins)[1] * 100,2)

plt.style.use('bmh')
plt.figure(1,figsize=(10,6), dpi=100,facecolor='w', edgecolor='k')
plt.bar(x,win_percent)
plt.title('{} Trained Agent Win Percent {}\n Test Episodes: {}'.format(STRAT,add_info,NUM_EPISODES),fontweight='bold',fontsize = 15)
plt.xlabel('Opponent Strategies',fontweight='bold',fontsize = 12)
plt.ylabel('Win %',fontweight='bold',fontsize = 12)
for i,y in enumerate(win_percent):
        plt.text(i, y, y, ha = 'center',fontweight='bold',fontsize = 12)

plt.savefig('TestResults/' + STRAT + '_Agent_' + METHOD + add_info + '.png')
plt.show()