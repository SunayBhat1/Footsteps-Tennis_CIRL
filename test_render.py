# test render
from PaperTennis_env import PaperTennisEnv
import numpy as np
env = PaperTennisEnv()

for episode in range(10):

    env.reset()
    state = env.state
    done = False
    while not done:
        env.render()
        action = np.around(np.random.uniform(0,state[1]))

        state, reward, done,_ = env.step(action,2)

    


