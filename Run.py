# import gym
from PaperTennis_env import PaperTennisEnv
import numpy as np
# import math
import matplotlib.pyplot as plt
import time
import pickle

start_time = time.time()

"""
Resources:
https://www.andrew.cmu.edu/course/10-403/slides/S19_lecture8_FApredictioncontrol.pdf
https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
https://github.com/ankitdhall/CartPole/blob/master/Qlearning-linear.py
"""

#Hyperparms
OPPONENT=2
NUM_EPISODES = 100000
GAMMA = 0.9
ALPHA = 0.1
E_GREEDY = 0.1

# Get action e-greedy
def get_action(state,Q_val):
    if state[1] == 0:
        return 0

    p_epsilon = np.random.uniform(0,1)
    if p_epsilon < E_GREEDY:
        return np.argmax(np.random.uniform(0,1,(1,state[1]))) + 1

    q_s = np.zeros(state[1])

    for i in range(0,state[1]):
        q_s[i] = Q_val[state[0],state[1],state[2],i]

    return np.argmax(q_s) + 1

# init env
env = PaperTennisEnv()

# Init value method
Q_val = np.zeros([5,51,51,50])

# Plotting Stuff
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
trend_wins = np.zeros(NUM_EPISODES-100)
trend_TD = np.zeros(NUM_EPISODES-100)
cum_reward = np.zeros(NUM_EPISODES+1)

stepiter = 0

wins = np.zeros(NUM_EPISODES)
td_error = np.zeros(NUM_EPISODES)

for episode in range(NUM_EPISODES):

    env.reset()
    state = env.state
    done = False

    action = get_action(state,Q_val)

    # Generate an episode
    error_episode = 0
    while not done:

        state_prime, reward, done = env.step(action,OPPONENT)

        action_prime = get_action(state_prime,Q_val)

        # SARSA tabular update (Section 6.5 S.B., psuedocode)
        TD_error = (reward + GAMMA * Q_val[state_prime[0],state_prime[1],state_prime[2],action_prime-1]
            -Q_val[state[0],state[1],state[2],action-1])

        Q_val[state[0],state[1],state[2],action-1] = Q_val[state[0],state[1],state[2],action-1] + ALPHA*TD_error

        error_episode += abs(TD_error)

        state  = state_prime
        action = action_prime

        stepiter += 1

        if (done and reward == 1):
            wins[episode] = 1
            

    td_error[episode] = error_episode
    if episode > 100:
        trend_wins[episode-100] = np.sum(wins[episode-100:episode])
        trend_TD[episode-100] = np.sum(td_error[episode-100:episode])
    
    if episode%1000 == 0: print("Episode %d completed with reward %d" % (episode, reward))
    cum_reward[episode+1] = cum_reward[episode] + reward

print("--- %s seconds ---" % (time.time() - start_time))
print("Final mean win percent = %s " % (np.mean(trend_wins[-10000:])))

ax1.scatter(range(100,NUM_EPISODES),trend_wins,s=3,marker='x')
ax2.scatter(range(100,NUM_EPISODES),trend_TD,s=3,marker='x',c='g')
ax3.scatter(range(0,NUM_EPISODES+1),cum_reward,s=3,marker='x',c='r')
ax1.title.set_text("Last 100 Win%")
ax2.title.set_text("TD Error Convergence")
ax3.title.set_text("Cumulative Reward vs Episode ")
fig.suptitle('SARSA TD For Q-Value Control\nLong player')
ax1.set_xscale('log')
ax2.set_xscale('log')
ax3.set_xscale('log')
ax3.set_yscale('log')
ax1.grid()
ax2.grid()
ax3.grid()
# ax4.grid()
plt.show()