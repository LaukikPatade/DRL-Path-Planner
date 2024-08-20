# Path Planning in a Custom Agricultural Environment using Q-Learning and Deep Q-Learning
## Overview
This project demonstrates the implementation of Q-Learning and Deep Q-Learning algorithms for path planning in a custom agricultural environment. The environment is modeled on the existing Frozen Lake environment provided by the Gymnasium API. The goal is to navigate through the environment from a start point to a goal, avoiding obstacles and maximizing rewards.

## Environment Description
The environment represents an agricultural field with the following features:
* Field Layout: The environment is structured as a grid (similar to the Frozen Lake environment) with different types of terrain, including plowed soil, waterlogged areas, and crop patches.
* State Space: Each state corresponds to a position on the grid.
* Action Space: The agent can move in four directions: up, down, left, and right.

## Rewards:
* Positive reward for reaching the goal (e.g., the endpoint of a crop row).
* Negative reward for stepping on waterlogged areas or leaving the field boundaries.
* Neutral or slightly positive rewards for moving on plowed soil.

## Algorithms
1. Q-Learning
A tabular Q-Learning approach is used to estimate the optimal policy for the agent.
The algorithm iteratively updates the Q-values based on the agent's experience, using the epsilon-greedy algorithm.
2. Deep Q-Learning (DQN)
A Deep Q-Network is employed to handle the larger state space and improve the learning process.
The network approximates the Q-values, enabling the agent to generalize its experience to unseen states.


