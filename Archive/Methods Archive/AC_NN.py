from PaperTennis_env import PaperTennisEnv
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


env = PaperTennisEnv()

state_size = len(env.observation_space.spaces)
action_size = env.action_space.n
learnrate = 0.0001

# Parms
STRATEGY = 'Random'
add_info = '_f1000'
NUM_EPISODES = 10000
mean_window = 10
OPP_FREQ = 1000

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

# Plot stuff
trend_wins = np.zeros(NUM_EPISODES-100)
trend_wins10 = np.zeros(NUM_EPISODES-10)
wins = np.zeros(NUM_EPISODES)

# Define Actor and Critic NNs
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

# Define Functions for use
def compute_returns(next_value, rewards, masks, gamma=1):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns

if __name__ == '__main__':

    # Class Instances and optimizers
    actor = Actor(state_size, action_size).to(device)
    critic = Critic(state_size, action_size).to(device)
    optimizerA = optim.Adam(actor.parameters(),lr=learnrate)
    optimizerC = optim.Adam(critic.parameters(),lr=learnrate)

    ### Train
    for episode in tqdm(range(NUM_EPISODES),ncols=100):
        
        env.reset()
        state = env.state
        done = False

        log_probs = []
        values = []
        rewards = []
        masks = []
        entropy = 0

        while not done:
            state = torch.FloatTensor(state).to(device)
            dist, value = actor(state), critic(state)

            action = dist.sample()
            if env.state[1] > 0: true_action = int(action.cpu().numpy()*env.state[1]/51)+1
            else: true_action = 0
            next_state, reward, done, info = env.step(true_action,OPP_Strat[episode])

            log_prob = dist.log_prob(action).unsqueeze(0)
            entropy += dist.entropy().mean()

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.tensor([reward], dtype=torch.float, device=device))
            masks.append(torch.tensor([1-done], dtype=torch.float, device=device))

            state = next_state

            if done:
                if reward == 1: wins[episode] = 1
                if episode >= 100: trend_wins[episode-100] = np.sum(wins[episode-100:episode])
                if episode >= 10: trend_wins10[episode-10] = np.sum(wins[episode-10:episode])
                break


        next_state = torch.FloatTensor(next_state).to(device)
        next_value = critic(next_state)
        returns = compute_returns(next_value, rewards, masks)

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



    torch.save(actor.state_dict(), 'TrainedModels/' + STRATEGY + '_Actor_NN.pt')

    ## Plotting

    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(14,6), dpi= 150, facecolor='w', edgecolor='k')
    fig.suptitle('Actor-Critic Neural Network {} Agent\n Last 10k: {:.2f}% Wins'.format(STRATEGY,(np.mean(trend_wins[-100:]))),fontweight='bold',fontsize = 16)

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

    fig.savefig('TrainResults/' + STRATEGY + '_AC_NN_Train' + add_info + '.png')
    plt.show()






