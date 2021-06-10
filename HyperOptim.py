from Agents import SARSA_LSFA, PGAC_DNN, SARSA_QTable, MonteCarlo_QTable
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')
from urllib import request
import sys

notify = True

gamespace = (7,50)
args = { # 1-5
        'OPP_STRAT' : 1,
        # Default 1
        'OPP_FREQ' : 1,
        # Default 1000000, 10000  (QT, LSFA/DNN)
        'NUM_EPISODES' : 5000, # All        !!!! Change for LSFA
        # Default 0.9,1,1(QT, LSFA, DNN)          
        'GAMMA' : 1, # All
        # Default 0.2, 0.01, 0.0001   (QT,LSFA,DNN)
        'ALPHA' : 0.008, # SARSA, MC, LSFA    !!!! Change for LSFA
        # Default 0.05, 0.01
        'EPSILON' : 0.05, # SARSA, MC         !!!! Change for LSFA
        # Default: 10000,100
        'AVG_WINDOW' : 100,  #               !!!! Change for LSFA
        # Default: 1000
        'TEST_EPISODES': 1000,
        # Default: True
        'CHECK_STABLE': False,
        # Default: ''
        'add_label' : '',
    }

savepath = 'Data/SARSA_LSFA/Optim/'

# Hyper Optimization 

# Gamma
# outF = open(savepath + 'Gamma Values.txt', 'w')
# gamma_range = np.arange(0,1.1,0.1)
# gamma_op = np.zeros((len(gamma_range),args['NUM_EPISODES']))
# win100 = np.zeros((len(gamma_range),args['NUM_EPISODES']-99))
# fig1, (ax1, ax2)= plt.subplots(1,2,figsize=(13,6), facecolor='w', edgecolor='k')
# args['OPP_STRAT'] = 1
# for i,args['GAMMA'] in enumerate(gamma_range):
#     Agent = SARSA_LSFA(args)
#     gamma_op[i,:] = Agent.train()
#     win100[i,:] = np.flip(np.convolve(np.flip(gamma_op[i,:]), np.ones(100), mode='valid'))
#     ax1.plot(range(99,args['NUM_EPISODES']),win100[i,:],label=round(Agent.gamma,2))
# means = np.around(np.mean(win100[:,-1000:],1),2)
# stds = np.around(np.std(win100[:,-1000:],1),2)
# outF.write('Mean Agent Gammas: ' + str(gamma_range) + '\nAvgs: ' + str(means) +'\nStds: ' + str(stds))

# args['OPP_STRAT'] = 5
# for i,args['GAMMA'] in enumerate(gamma_range):
#     Agent = SARSA_LSFA(args)
#     gamma_op[i,:] = Agent.train()
#     win100[i,:] = np.flip(np.convolve(np.flip(gamma_op[i,:]), np.ones(100), mode='valid'))
#     ax2.plot(range(99,args['NUM_EPISODES']),win100[i,:],label=round(Agent.gamma,2))
# means = np.around(np.mean(win100[:,-1000:],1),2)
# stds = np.around(np.std(win100[:,-1000:],1),2)
# outF.write('\nDynamic Agent Gammas: ' + str(gamma_range) + '\nAvgs: ' + str(means) +'\nStds: ' + str(stds))

# ax1.set_title(r'$\gamma$ Optimization Mean Agent',fontweight='bold',fontsize = 14)
# ax1.set_xlabel('Episode',fontweight='bold',fontsize = 12)
# ax1.set_ylabel('Last 100 Win %',fontweight='bold',fontsize = 12)
# ax1.grid(True, color='w', linestyle='-', linewidth=1)
# ax1.legend()

# ax2.set_title(r'$\gamma$ Optimization Dynamic Agent',fontweight='bold',fontsize = 14)
# ax2.set_xlabel('Episode',fontweight='bold',fontsize = 12)
# ax2.set_ylabel('Last 100 Win %',fontweight='bold',fontsize = 12)
# ax2.grid(True, color='w', linestyle='-', linewidth=1)
# ax2.legend()
# outF.close()
# fig1.savefig( savepath + 'Gamma.png')




# Alpha
# outF = open(savepath + 'Alpha Values.txt', 'w')
# alpha_range = np.arange(0.004,0.015,0.002)
# alpha_op = np.zeros((len(alpha_range),args['NUM_EPISODES']))
# win100 = np.zeros((len(alpha_range),args['NUM_EPISODES']-99))
# fig2, (ax3, ax4)= plt.subplots(1,2,figsize=(13,6), facecolor='w', edgecolor='k')
# args['OPP_STRAT'] = 1
# for i,args['ALPHA'] in enumerate(alpha_range):
#     Agent = SARSA_LSFA(args)
#     alpha_op[i,:] = Agent.train()
#     win100[i,:] = np.flip(np.convolve(np.flip(alpha_op[i,:]), np.ones(100), mode='valid'))
#     ax3.plot(range(99,args['NUM_EPISODES']),win100[i,:],label=round(Agent.alpha,4))
# means = np.around(np.mean(win100[:,-1000:],1),2)
# stds = np.around(np.std(win100[:,-1000:],1),2)
# outF.write('Mean Agent Alphas: ' + str(alpha_range) + '\nAvgs: ' + str(means) +'\nStds: ' + str(stds))

# args['OPP_STRAT'] = 5
# for i,args['ALPHA'] in enumerate(alpha_range):
#     Agent = SARSA_LSFA(args)
#     alpha_op[i,:] = Agent.train()
#     win100[i,:] = np.flip(np.convolve(np.flip(alpha_op[i,:]), np.ones(100), mode='valid'))
#     ax4.plot(range(99,args['NUM_EPISODES']),win100[i,:],label=round(Agent.alpha,4))
# means = np.around(np.mean(win100[:,-1000:],1),2)
# stds = np.around(np.std(win100[:,-1000:],1),2)
# outF.write('\nDynamic Agent Alphas: ' + str(alpha_range) + '\nAvgs: ' + str(means) +'\nStds: ' + str(stds))

# ax3.set_title(r'$\alpha$ Optimization Mean Agent',fontweight='bold',fontsize = 14)
# ax3.set_xlabel('Episode',fontweight='bold',fontsize = 12)
# ax3.set_ylabel('Last 100 Win %',fontweight='bold',fontsize = 12)
# ax3.grid(True, color='w', linestyle='-', linewidth=1)
# ax3.legend()

# ax4.set_title(r'$\alpha$ Optimization Dynamic Agent',fontweight='bold',fontsize = 14)
# ax4.set_xlabel('Episode',fontweight='bold',fontsize = 12)
# ax4.set_ylabel('Last 100 Win %',fontweight='bold',fontsize = 12)
# ax4.grid(True, color='w', linestyle='-', linewidth=1)
# ax4.legend()
# outF.close()
# fig2.savefig( savepath + 'Alpha.png')

# Epsilon
outF = open(savepath + 'Epsilon Values.txt', 'w')
eps_range = np.arange(0.05,0.11,0.01)
eps_op = np.zeros((len(eps_range),args['NUM_EPISODES']))
win100 = np.zeros((len(eps_range),args['NUM_EPISODES']-99))
fig3, (ax5, ax6)= plt.subplots(1,2,figsize=(13,6), facecolor='w', edgecolor='k')
args['OPP_STRAT'] = 1
for i,args['EPSILON'] in enumerate(eps_range):
    Agent = SARSA_LSFA(args)
    eps_op[i,:] = Agent.train()
    win100[i,:] = np.flip(np.convolve(np.flip(eps_op[i,:]), np.ones(100), mode='valid'))
    ax5.plot(range(99,args['NUM_EPISODES']),win100[i,:],label=round(Agent.epsilon,2))
means = np.around(np.mean(win100[:,-1000:],1),2)
stds = np.around(np.std(win100[:,-1000:],1),2)
outF.write('Mean Agent Epsilons: ' + str(eps_range) + '\nAvgs: ' + str(means) +'\nStds: ' + str(stds))

args['OPP_STRAT'] = 5
for i,args['EPSILON'] in enumerate(eps_range):
    Agent = SARSA_LSFA(args)
    eps_op[i,:] = Agent.train()
    win100[i,:] = np.flip(np.convolve(np.flip(eps_op[i,:]), np.ones(100), mode='valid'))
    ax6.plot(range(99,args['NUM_EPISODES']),win100[i,:],label=round(Agent.epsilon,2))
means = np.around(np.mean(win100[:,-1000:],1),2)
stds = np.around(np.std(win100[:,-1000:],1),2)
outF.write('Dynamic Agent Epsilons: ' + str(eps_range) + '\nAvgs: ' + str(means) +'\nStds: ' + str(stds))

ax5.set_title(r'$\epsilon$ Optimization Mean Agent',fontweight='bold',fontsize = 14)
ax5.set_xlabel('Episode',fontweight='bold',fontsize = 12)
ax5.set_ylabel('Last 100 Win %',fontweight='bold',fontsize = 12)
ax5.grid(True, color='w', linestyle='-', linewidth=1)
ax5.legend()

ax6.set_title(r'$\epsilon$ Optimization Dynamic Agent',fontweight='bold',fontsize = 14)
ax6.set_xlabel('Episode',fontweight='bold',fontsize = 12)
ax6.set_ylabel('Last 100 Win %',fontweight='bold',fontsize = 12)
ax6.grid(True, color='w', linestyle='-', linewidth=1)
ax6.legend()
outF.close()
fig3.savefig( savepath + 'Epsilon.png')


if notify:
    key = "Lmdwc3Ei4h0vfiIwLA4K0"
    message = 'Python_Script_' + sys.argv[0] + '_Is_Done'
    request.urlopen("https://maker.ifttt.com/trigger/notify/with/key/%s?value1=%s" % (key,message))

plt.show()