from Agents import SARSA_LSFA, SARSA_QTable, MonteCarlo_QTable
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')
from urllib import request

notify = True

gamespace = (7,50)
args = {
        'OPP_STRAT' : 1,
        # Default 1
        'OPP_FREQ' : 1,
        # Default 1000000
        'NUM_EPISODES' : 1000000, # All
        # Default 1
        'GAMMA' : 1, # All
        # Default 0.1
        'ALPHA' : 0.1, # SARSA, MC
        # Default 0.1
        'EPSILON' : 0.1, # SARSA, MC
        # Default: 10000,100
        'AVG_WINDOW' : 100,
        # Default: 1000
        'TEST_EPISODES': 1000,
        # Default: True
        'CHECK_STABLE': True,
        # Default: ''
        'add_label' : '',
    }

# Agent = SARSA_LSFA(args)
# Agent.train()
# Agent.evaluate()


# Run All 
stringM = 'Monte Carlo Q-Table'
savepath = 'Data/MonteCarlo_QT/'
train_histories = np.zeros((5,args['NUM_EPISODES']))
win100 = np.zeros((5,args['NUM_EPISODES']-99))
win_avg = np.zeros((5,args['NUM_EPISODES']-98-args['AVG_WINDOW']))
test_win_percent = np.zeros(5)

fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(15,5.5), facecolor='w', edgecolor='k')
fig.suptitle(stringM + ' All Agents Training',fontweight='bold',fontsize = 16)

fig2,ax3 = plt.subplots(1,1,figsize=(10,6))
xlabels = ['Mean','Long','Short','Naive','Dynamic']
ax3.set_title(stringM + ' All Agent Win %\n Number Test Episodes: {}'.format(args['TEST_EPISODES']),fontweight='bold',fontsize = 15)

for i in range (0,5):
    args['OPP_STRAT'] = i+1
    Agent = MonteCarlo_QTable(args)
    train_histories[i,:] = Agent.train()
    test_win_percent = Agent.evaluate()

    win100[i,:] = np.flip(np.convolve(np.flip(train_histories[i,:]), np.ones(100), mode='valid'))
    win_avg[i,:] = np.flip(np.convolve(np.flip(win100[i,:]), np.ones(args['AVG_WINDOW'])/args['AVG_WINDOW'], mode='valid'))
    ax1.plot(range(99,args['NUM_EPISODES']),win100[i,:])
    ax2.plot(range(99+args['AVG_WINDOW']-1,args['NUM_EPISODES']),win_avg[i,:])
    ax3.bar(np.arange(5)+i*0.15-0.375,test_win_percent,width=0.15)
    for ibar,y in enumerate(test_win_percent):
        ax3.text(ibar+i*0.15-0.375, y, y, ha = 'center',fontweight='bold',fontsize = 6)

ax1.set_title('Running 100 Game Win%',fontweight='bold',fontsize = 14)
ax1.set_xlabel('Episode',fontweight='bold',fontsize = 12)
ax1.set_ylabel('Last 100 Win %',fontweight='bold',fontsize = 12)
ax1.grid(True, color='w', linestyle='-', linewidth=1)
ax1.legend(['Mean','Long','Short','Naive','Dynamic'])

ax2.set_title('Running {:,} Avg Win%'.format(args['AVG_WINDOW']), fontweight='bold',fontsize = 14)
ax2.set_xlabel('Episode', fontweight='bold',fontsize = 12)
ax2.set_ylabel('Mean Win %', fontweight='bold',fontsize = 12)
ax2.grid(True, color='w', linestyle='-', linewidth=1)
ax2.legend(['Mean','Long','Short','Naive','Dynamic'])

ax3.set_xlabel('Opponent Strategies',fontweight='bold',fontsize = 12)
ax3.set_ylabel('Win %',fontweight='bold',fontsize = 12)
ax3.set_ylim([0,100])
ax3.set_xticklabels(['','Mean','Long','Short','Naive','Dynamic'])
ax3.legend(['Mean','Long','Short','Naive','Dynamic'],title='Trained Agents',loc="upper center", ncol=5)

fig2.savefig(savepath + 'TestAll.png')
fig.savefig( savepath + 'TrainAll.png')
plt.show()




if notify:
    key = "Lmdwc3Ei4h0vfiIwLA4K0"
    message = 'Python_Script_Is_Done'
    request.urlopen("https://maker.ifttt.com/trigger/notify/with/key/%s?value1=%s" % (key,message))