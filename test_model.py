# test render
from PaperTennis_env import PaperTennisEnv
from AC_NN import Actor
import torch
import matplotlib.pyplot as plt
import numpy as np
import pickle
from tqdm import tqdm

#HParms 
NUM_EPISODES = 500
RENDER = False
STRAT = 'Random'
METHOD = 'SARSA_LS'
add_info = ''
Strategy_Shift = 1

# Load a Strategy
# train_episodes, Q_val = pickle.load(open('TrainedModels/' + STRAT + '_' + METHOD + add_info + '.pt', "rb" ))
# print('Episodes Trained {}'.format(train_episodes))

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# env = PaperTennisEnv()
# state_size = len(env.observation_space.spaces)
# action_size = env.action_space.n
# model = torch.load('TrainedModels/' + STRAT + '_' +  METHOD + '.pt')
# actor = Actor(state_size, action_size).to(device)
# actor.load_state_dict(model)

train_episodes, w = pickle.load(open('TrainedModels/' + STRAT + '_' + METHOD  + '.p', "rb" ))


# Get action
def get_action_QT(state,Q_val):
    if state[1] == 0:
        return 0

    q_s = np.zeros(state[1])

    for i in range(1,state[1]):
        q_s[i] = Q_val[state[0],state[1],state[2],i]

    return np.argmax(q_s) + 1

def get_action_AC(state,actor):
    if state[1] == 0: return 0
    s = torch.FloatTensor(state)
    dist = actor(s)
    return  int(dist.sample().cpu().numpy()*state[1]/51)+1

# Function approx to compute
def get_features(state,action):
    feat_vec = np.zeros(28)

    # Splines : Discretized spaces
    state_splines = [list(range(0,6)),list(range(6,18)),list(range(18,34)),list(range(34,56))]
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

def Q_value(state,action,linear_weights):

    feat_vec = get_features(state,action)
    return (linear_weights @ feat_vec).item()

# Get action greedy
def get_action_LS(linear_weights, state):
    if state[1] == 0: return 0

    q_s = np.zeros(state[1])

    for i in range(0,state[1]): q_s[i] = Q_value(state,i,linear_weights)

    return np.argmax(q_s) + 1


# Create Environment and wins vector 
wins = np.zeros((5,NUM_EPISODES))

# Init Env
env = PaperTennisEnv()

# Run Through Strategies
for opp_strategy in tqdm(range(1,6),ncols=100):

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
            # action = get_action_AC(state,actor)
            action = get_action_LS(w[state[0],:], state)
            state, reward, done, _ = env.step(action,int(OPP_Strat[episode]))

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

# plt.savefig('TestResults/' + STRAT + '_Agent_' + METHOD + add_info + '.png')
plt.show()