# Genetic Programming

from PaperTennis_env import PaperTennisEnv
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm


def run_game(strat1,strat2):
    env = PaperTennisEnv(two_player=True)
    state = env.reset()
    done = False

    while not done:
        state, reward, done,_ = env.step(int(strat1[state[0],state[1],state[2]]),int(strat2[state[0],state[2],state[1]]))

    return reward,1-reward


def getFitness(Strategies):
    env = PaperTennisEnv(two_player=True)

    wins = np.zeros(100)
    for i in range(100):
        for j in range(i,100):
            if i==j: continue
            ri,rj= run_game(Strategies[i,:,:,:],Strategies[j,:,:,:])
            wins[i] += ri
            wins[j] += rj
    return wins
    
def next_gen(Strategies,fitness,num_best = 12):
    new_Strategies = np.zeros((100,5,51,51))
    fitness = fitness/np.sum(fitness)

    best = np.random.choice(np.array(range(100)),num_best, p=fitness)
    new_Strategies[0:len(best),:,:,:] = Strategies[best,:,:,:] 

    # "num_best choose 2" 50% combinations of strategies
    istrat = len(best)
    count = 0
    for i in range(len(best)):
        iS = best[i]
        for j in range(i,len(best)):
            jS = best[j]
            if i == j: continue
            cross_mask = np.random.random(size=Strategies.shape[1:4]) < 0.5
            new_Strategies[istrat,cross_mask] = Strategies[i,cross_mask]
            new_Strategies[istrat,np.invert(cross_mask)] = Strategies[j,np.invert(cross_mask)]

            istrat +=1

    # Remaining strategies are totally new
    for i in range(1,51): new_Strategies[istrat:100,:,i,:] = np.around(np.random.uniform(1,i,np.shape(new_Strategies[istrat:100,:,:,:])[0:3]))
    np.shape(Strategies[78:100,:,:,:])

    return new_Strategies

# Begin

# Setup random strategy space
Strategies = np.zeros((100,5,51,51))
Startegies90 = np.zeros((1,5,51,51))
Strategies[:,:,0,:] =  np.zeros((100,5,51))
for i in range(1,51): Strategies[:,:,i,:] = np.around(np.random.uniform(1,i,(100,5,51)))

# Iterate
for i in tqdm(range(10000),ncols=100):
    fitness = getFitness(Strategies)
    if np.shape(np.where( fitness > 90 ))[1] > 0:
        print('On gen {}, Indices greater than 90:  {}'.format(i,np.where(fitness > 90 )[0].tolist()))
        ind = np.argpartition(fitness, -1)[-1:]
        if np.shape(Startegies90)[0] == 1: Startegies90 = Strategies[ind,:,:,:]
        else: Startegies90 = np.concatenate((Startegies90,Strategies[ind,:,:,:]))


    Strategies = next_gen(Strategies,fitness) 

# Save best 4 at end
ind = np.argpartition(fitness, -4)[-4:]
pickle.dump(Strategies[ind,:,:,:], open('FixedStrategies/Strategy_1.p', 'wb' ))

pickle.dump(Startegies90, open('FixedStrategies/Strategy90_1.p', 'wb' ))


plt.plot(fitness)
plt.show()


