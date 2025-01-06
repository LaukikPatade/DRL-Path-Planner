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

![FrozenLake2024-08-2122-48-13-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/43224eeb-6332-42f5-b86b-8eded5b1aa73)

# Path Planning in a Custom Agricultural Environment using Actor-Critic Algorithms in ROS & Gazebo  

## Overview  
Building upon the earlier Q-Learning and Deep Q-Learning approaches, this project explores advanced actor-critic algorithms to handle continuous action spaces for path planning in a custom agricultural environment. The environment is simulated in a 3D setup using ROS and Gazebo, enabling a more realistic representation of the agricultural field, including terrain variability and real-world physics.  

## Environment Description  
The environment models an agricultural field with the following features:  
- **Field Layout**: A realistic 3D grid field containing plowed rows, waterlogged areas, crop patches, and uneven terrains.  
- **State Space**: Defined by the UGV's position, orientation, and sensor readings (e.g., LiDAR, IMU).  
- **Action Space**: Continuous actions, such as precise steering angles and throttle control, allow finer movements suited to real-world UGV dynamics.  
- **Sensors**: Simulated sensors provide data for obstacle detection and environmental feedback.  

## Rewards  
The reward structure is designed to encourage optimal path planning:  
- **Positive Rewards**: For reaching the goal efficiently and moving along plowed crop rows.  
- **Negative Rewards**: For collisions, veering off the field, or stepping into waterlogged areas.  
- **Continuous Feedback**: The reward adjusts based on the distance to the goal and smoothness of the path.  

## Algorithms  
### 1. Deep Deterministic Policy Gradient (DDPG)  
- DDPG is employed for its ability to handle continuous action spaces.  
- The actor network predicts the optimal action (steering and throttle values) for a given state.  
- The critic network evaluates the quality of the action using the Q-value.  
- Replay buffers and target networks are used to stabilize training.  

### 2. Twin Delayed Deep Deterministic Policy Gradient (TD3)  
- An enhancement of DDPG that addresses the overestimation bias in Q-value predictions.  
- Includes techniques like clipped double Q-learning and delayed policy updates for robust learning.  

## Implementation in ROS & Gazebo  
The environment is simulated in Gazebo with ROS for communication and control:  
- **Simulation Setup**: The agricultural field is constructed using Gazebo's terrain and object modeling features. Obstacles, crop rows, and varying terrain types are included.  
- **UGV Control**: A differential-drive robot is simulated with ROS controllers managing wheel velocities.  
- **Training Framework**: Custom nodes handle state updates, reward calculation, and action execution using actor-critic algorithms.  
- **Sensor Integration**: Simulated LiDAR and IMU data aid obstacle avoidance and precise navigation.  

![3D-FARM](https://github.com/user-attachments/assets/cc81e93d-d1f9-454b-939e-7d076d33fa02)
![goal_static](https://github.com/user-attachments/assets/49e3befd-f36d-4f3e-8563-2db03b9b38d2)
![goal_dynamic](https://github.com/user-attachments/assets/aebb39f8-b436-4ec9-80fb-a2b7ba107c94)

## Results and Observations  
1. **Episode Length**: Actor-critic methods reduced the average episode length compared to discrete action methods.  
2. **Smoothness of Path**: Continuous action control achieved smoother paths, essential for real-world UGV applications.  
3. **Noise Robustness**: TD3 demonstrated robustness in noisy environments, performing better than DDPG in scenarios with sensor inaccuracies.  

## Conclusion  
Actor-critic algorithms, particularly DDPG and TD3, significantly improve path planning performance in a realistic agricultural environment. By leveraging continuous action spaces and advanced reward structures, these methods pave the way for deploying autonomous UGVs in precision agriculture scenarios.  



