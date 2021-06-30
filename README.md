Liars Dice Environment and Startegy Project
==============================
Description: An OpenAI Gym environemnt for the two player startegy game Liar's Dice
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
    ├── LiarsDice_env.py      <- OpenAI Gym environment for two-player Liar's Dice game

Game Rules
------------
Rules of Game:
-Each players starts with 5 dice
-Randomly chosen who starts the betting
-Random roll is performed of all 10 dice when environment is created
-Players bet on total number in play, based only on the 5 dice they have rolled and past bets
-Starting player bets quantity and face; Example: (2,3) is two threes 
-Each subsequent bet must raise either the quantity (3 of a kind or higher) or the value (2 of 4’s,5’s,6’s)
-The round ends once a player calls BS/liar on the previous bet
--If they are correct and the last bet was a lie, the player receives a reward of 1
--If they are incorrect and the last bet was true, the player receives 0 reward

Environment notes:
-Currently the environment ends the episode at one round by default (although some code was written with the possibility for more rounds)
--The losing player would lose a die, dice are re-rolled, and the rounds continue until one player loses all their dice
-Currently the environment requires an input for each player, hence it's two-agent
--At least one, possibly more fixed strategies could be developed as bots that more complex agents can be trained against. 


--------
