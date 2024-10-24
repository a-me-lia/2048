# 2048
2048 Game with Reinforcement Learning Bot
## Project Overview
This project implements the classic game 2048 using Pygame and incorporates a reinforcement learning (RL) bot that can be trained to play the game. The game allows both human and bot players to interact with the board, providing a unique opportunity to explore how an RL-based bot learns to play an intricate puzzle game.

The project features:

- A GUI using Pygame for playing 2048.
- An option to choose between human or bot players.
- The ability to train the bot using Q-learning, save the training results in a pickle file, and load saved Q-tables for future use.
- A Tkinter interface to select player type and other options.
- The ability to enable or disable animations during gameplay.

## Game Mechanics
The game follows the rules of 2048:

- A 4x4 grid where numbered tiles combine when pushed in any direction.
- The goal is to create a tile with the number 2048 by combining similar tiles.
- Every move spawns a new tile (2 or 4) in a random empty spot.
- The game ends when no more valid moves are possible.
- The reinforcement learning (RL) method used in this project is Q-learning, a form of model-free learning where the bot interacts with the environment (the game) and improves its - performance over time. The bot learns which actions lead to the most favorable outcomes based on rewards.

## Q-Learning Process:
- State Representation: The state is the current state of the board, represented by the positions of all tiles.
- Action Space: The possible actions include moving the tiles in four directions: up, down, left, and right.
- Reward: The reward system incentivizes achieving higher numbers. The bot receives a reward based on the sum of the numbers of combined tiles in a move.
- Q-Table: The Q-table stores the expected future rewards for each action in a particular state. Initially, the table is random, but it updates as the bot plays the game. The Q-values are updated using the formula:

```Q(s, a) = Q(s, a) + α * [r + γ * max(Q(s', a')) - Q(s, a)]```

Where:
- Q(s, a) is the current Q-value for state s and action a.
- α is the learning rate.
- r is the reward for the action.
- γ is the discount factor.
- max(Q(s', a')) is the maximum future reward for the next state s'.

Over time, the bot explores the game environment and exploits its learned knowledge to make better decisions.

## Challenges 
During the development of this RL-based bot, I faced a number of challenges that significantly impacted progress:

Incorrectly Programmed Game Mechanics: Early in development, there were critical issues with the implementation of the game's mechanics. These bugs caused glitches where the tiles did not merge correctly or move as intended, which severely impacted the bot's training. This led to scenarios where the bot learned incorrect strategies or failed to learn anything meaningful at all. Debugging these game mechanics required significant time and effort, as I had to carefully inspect every part of the game's logic.

Seemingly No Results from Training: Another major challenge was the fact that, after several rounds of training, the bot showed no apparent improvement. After extensive testing and investigation, I discovered that the problem lay in how the Q-table was being updated. There were issues with the reward system and the state-action transitions, causing the bot to essentially “forget” what it had learned from previous moves. By adjusting the reward values and improving the handling of state transitions, the training started showing better results.

Hyperparameter Tuning: Finding the right combination of learning rate, discount factor, and exploration-exploitation balance was also tricky. Too much exploration led the bot to ignore good strategies, while too much exploitation led it to repeat mediocre strategies without improving. Through experimentation, I managed to fine-tune these parameters to achieve a balance where the bot gradually improved its gameplay.

## Performance
Despite these challenges, the bot was able to show gradual improvements over time. The training process, however, still requires a large number of iterations to achieve satisfactory results, which reflects the complexity of the game. The ability to save and load the Q-table after training is essential to avoid retraining from scratch each time.

Running on an Intel I9-12900@5.35GHz (Single core), this game can run 10k iterations in a little less than a hour.

How to Run the Project
Prerequisites
Python 3.12 or higher
Pygame 2.5.1 or higher
Tkinter (for the player type selection GUI)
Matplotlib (for optional visualization of training results)
Setup Instructions
Clone this repository:

bash
Copy code
git clone https://github.com/yourusername/2048-rl-bot.git
cd 2048-rl-bot
Install the required Python packages:

bash
Copy code
pip install -r requirements.txt
Run the 2048 game:

bash
Copy code
python src/game_2048.py
Choose the player type (human or bot) from the Tkinter GUI that appears.

If choosing the bot player, the bot will begin making moves automatically. You can also train the bot by allowing it to play multiple games in succession and saving the resulting Q-table.