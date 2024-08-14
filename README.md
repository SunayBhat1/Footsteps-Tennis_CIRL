Paper Tennis 
==============================
Description: An OpenAI Gym environemnt for the two player startegy game Paper Tennis
as well as staretgy testing and modeling. 

Getting Started
------------
The following python libraries are needed:  
-gym

# Resources:
Dice Probability Theory: https://mathworld.wolfram.com/Dice.html

Project Organization
------------

    ├── README.md             <- The top-level README 
    │
    ├── PaperTennis_env.py      <- OpenAI Gym environment for two-player Liar's Dice game

Game Rules
------------
Rules of Game:
-Each players starts with 5 dice


Environment notes:
-Currently the environment ends the episode at one round by default (although some code was written with the possibility for more rounds)
--The losing player would lose a die, dice are re-rolled, and the rounds continue until one player loses all their dice
-Currently the environment requires an input for each player, hence it's two-agent
--At least one, possibly more fixed strategies could be developed as bots that more complex agents can be trained against. 


--------
