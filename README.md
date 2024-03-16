# Reinforcement learning with Proximal Policy Optimization (PPO) on Connect Four 
This project combines Kaggle's ConnectX gym environment with Stable Basslines PPO training to attempt to train a model on how to play connect four. The project is a continuation of Kaggel's reinforcement learning tutorial. 
## Policy Approximation Model
The policy that PPO is attempting to optimize is modeled by a convolutional neural network created with Pytorch. The network consists of 7 layers of 2d convolutions and between each layer is batch normalization and a rectified linear activation function (relu). After the last layer of convolution, the output is flattened and pasted through a final linear layer with another relu activation. 
## Reward Function
The agent receives a score of 10 for winning the game and -10 for losing. Any invalid move, trying to play on a full column, gives a score of -50. In addition to this, blocking a loss by putting a piece where the opponent would need to win gives a reward of 0.5.
## Method
The PPO agent plays against an opponent for 100,000 games, and then the updated agent has its mean reward evaluated by playing 150 games vs the opponent it was training against. In this project, two opponents were used. The first was an opponent playing random moves. The PPO agent played this opponent until it was able to achieve greater than 10.0 mean reward during evaluation or 10,000,000 games were played. The opponent was then changed to a min/max algorithm. This lets the opponent pick the best move for itself and assumes the PPO agent will also pick its best move. The PPO agent played this opponent until the mean reward minus the standard deviation of the reward was greater than 0, meaning the PPO agent was winning more than losing.    
## Results
The PPO agent was able to achieve a winning record vs the min/max opponent after 157 iterations (15,700,000 games), 100 versus a random opponent, and 57 vs the min/max opponent. This took approximately 52 hours on my computer. 
## Resources
Stable Baseline: https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html \
Kaggle ConnectX: https://github.com/Kaggle/kaggle-environments/blob/master/kaggle_environments/envs/connectx/connectx.ipynb
