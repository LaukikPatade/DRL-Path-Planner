import os
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import torch
from torch import nn
import torch.nn.functional as F
from crop_env import create_environment
from torch.utils.tensorboard import SummaryWriter

# Define model
class DoubleDQN(nn.Module):
    def __init__(self, in_states, h1_nodes, out_actions):
        super().__init__()
        self.input_nodes = in_states

        # Define network layers
        self.fc1 = nn.Linear(in_states, h1_nodes)    # first fully connected layer
        self.fc2 = nn.Linear(h1_nodes, h1_nodes)     # second fully connected layer
        self.out = nn.Linear(h1_nodes, out_actions)  # ouptut layer

    def forward(self, x):
        x = F.relu(self.fc1(x)) # Apply rectified linear unit (ReLU) activation
        x = F.relu(self.fc2(x))
        x = self.out(x)         # Calculate output
        return x

# Define memory for Experience Replay
class ReplayMemory():
    def __init__(self, maxlen):
        self.memory = deque([], maxlen = maxlen)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)

# CropField Deep Q-Learning
class CropFieldDQL():
    # Hyperparameters
    learning_rate_a = 1e-3          # learning rate (alpha)
    discount_factor_g = 0.9         # discount rate (gamma)
    network_sync_rate = 10          # number of steps the agent takes before syncing the policy and target network
    replay_memory_size = 1000       # size of replay memory
    mini_batch_size = 32            # size of the training data set sampled from the replay memory

    # Neural Network
    loss_fn = nn.MSELoss()          # NN Loss function. MSE = Mean Squared Error.
    optimizer = None                # NN Optimizer. Initialize later.

    ACTIONS = ['L','D','R','U']     # for printing 0,1,2,3 => L(eft),D(own),R(ight),U(p)

    # Train the CropField environment
    def train(self, episodes, render=False, is_slippery=False, map="4x4"):
        writer = SummaryWriter(f'./runs_trailing/d2qn_{map}')

        # Create CropField instance
        env = create_environment(render_mode="human" if render else None, map_name=map, is_slippery=is_slippery)
        num_states = env.observation_space.n
        num_actions = env.action_space.n
        
        epsilon = 1 # 1 = 100% random actions
        memory = ReplayMemory(self.replay_memory_size)

        # Create policy and target network.
        policy_dqn = DoubleDQN(in_states=num_states, h1_nodes=num_states, out_actions=num_actions)
        target_dqn = DoubleDQN(in_states=num_states, h1_nodes=num_states, out_actions=num_actions)

        # Make the target and policy networks the same (copy weights/biases from one network to the other)
        target_dqn.load_state_dict(policy_dqn.state_dict())

        print('Policy (random, before training):')
        self.print_dqn(policy_dqn)

        # Policy network optimizer.
        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

        # List to keep track of rewards collected per episode. Initialize list to 0's.
        rewards_per_episode = np.zeros(episodes)

        # List to keep track of epsilon decay
        epsilon_history = []

        # Track number of steps taken. Used for syncing policy => target network.
        step_count = 0

        for i in range(episodes):
            print("Episode: ", i)
            state = env.reset()[0]  # Initialize to state 0
            terminated = False      # True when agent crashes or reached goal
            truncated = False       # True when agent takes more than 200 actions
            steps = 0

            # Agent navigates map until it falls into hole/reaches goal (terminated), or has taken 200 actions (truncated).
            while(not terminated and not truncated):

                # Select action based on epsilon-greedy
                if random.random() < epsilon:
                    # select random action
                    action = env.action_space.sample() # actions: 0=left,1=down,2=right,3=up
                else:
                    # select best action
                    with torch.no_grad():
                        action = policy_dqn(self.state_to_dqn_input(state, num_states)).argmax().item()

                # Execute action
                new_state, reward, terminated, truncated, _ = env.step(action)
                memory.append((state, action, new_state, reward, terminated)) # Save experience into memory

                # Move to the next state
                state = new_state

                # Increment step counter
                step_count+=1

                steps += 1
                if steps > 50000: terminated=True

            # Keep track of the rewards collected per episode.
            if reward == 1:
                rewards_per_episode[i] = 1

            # Check if enough experience has been collected and if at least 1 reward has been collected
            if len(memory)>self.mini_batch_size:
                mini_batch = memory.sample(self.mini_batch_size)
                self.optimize(mini_batch, policy_dqn, target_dqn)        

                # Decay epsilon
                epsilon = max(epsilon - 1/episodes, 0)
                epsilon_history.append(epsilon)

                # Copy policy network to target network after a certain number of steps
                if step_count > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step_count=0
            writer.add_scalar('Total Rewards', np.sum(rewards_per_episode[max(0, i-100):(i+1)]), i)
            # writer.add_scalar('Total Rewards', np.sum(rewards_per_episode), i)
            writer.add_scalar('Moves per Episode', steps, i)

        # Close Environment
        env.close()
        writer.close()

        # Create new graph
        plt.figure(1)

        # Plot average rewards (Y-axis) vs episodes (X-axis)
        sum_rewards = np.zeros(episodes)
        for x in range(episodes):
            sum_rewards[x] = np.sum(rewards_per_episode[max(0, x-100):(x+1)])
        plt.subplot(121) # plot on a 1 row x 2 col grid, at cell 1
        plt.plot(sum_rewards)

        # Plot epsilon decay (Y-axis) vs episodes (X-axis)
        plt.subplot(122) # plot on a 1 row x 2 col grid, at cell 2
        plt.plot(epsilon_history)

        # Save plots
        if is_slippery:
            plt.savefig(f'noisy_d2qn_{map}.png')
            torch.save(policy_dqn.state_dict(), f"noisy_d2qn_{map}.pt")

        else:
            plt.savefig(f'd2qn_{map}.png')
            torch.save(policy_dqn.state_dict(), f"d2qn_{map}.pt")


    # Optimize policy network
    def optimize(self, mini_batch, policy_dqn, target_dqn):

        # Get number of input nodes
        num_states = policy_dqn.input_nodes

        current_q_list = []
        target_q_list = []

        for state, action, new_state, reward, terminated in mini_batch:

            if terminated:
                # Agent either reached goal (reward=1) or fell into hole (reward=0)
                # When in a terminated state, target q value should be set to the reward.
                target = torch.FloatTensor([reward])
            else:
                # Calculate target q value
                next_action = policy_dqn(self.state_to_dqn_input(new_state, num_states)).argmax()

                # Evaluate the Q-value of the next action using the target network.
                with torch.no_grad():
                    target = torch.FloatTensor(
                        reward + self.discount_factor_g * target_dqn(self.state_to_dqn_input(new_state, num_states))[next_action]
                    )

            # Get the current set of Q values
            current_q = policy_dqn(self.state_to_dqn_input(state, num_states))
            current_q_list.append(current_q)

            # Get the target set of Q values
            target_q = target_dqn(self.state_to_dqn_input(state, num_states))
            target_q[action] = target      # Adjust the specific action to the target
            target_q_list.append(target_q)

        # Compute loss for the whole minibatch
        loss = self.loss_fn(torch.stack(current_q_list), torch.stack(target_q_list))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    '''
    Converts an state (int) to a tensor representation.
    For example, the FrozenLake 4x4 map has 4x4=16 states numbered from 0 to 15.

    Parameters: state=1, num_states=16
    Return: tensor([0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    '''

    def state_to_dqn_input(self, state:int, num_states:int)->torch.Tensor:
        input_tensor = torch.zeros(num_states)
        input_tensor[state] = 1
        return input_tensor

    # Run the CropField environment with the learned policy
    def test(self, episodes, is_slippery=False, map="4x4"):
        # Create CropField instance
        env = create_environment(render_mode="human", map_name=map, is_slippery=is_slippery)
        num_states = env.observation_space.n
        num_actions = env.action_space.n

        # Load learned policy
        policy_dqn = DoubleDQN(in_states=num_states, h1_nodes=num_states, out_actions=num_actions)

        if is_slippery:
            policy_dqn.load_state_dict(torch.load(f"./models/d2qn/noisy_d2qn_{map}.pt"))
        else:
            policy_dqn.load_state_dict(torch.load(f"./models/d2qn/d2qn_{map}.pt"))
        
        policy_dqn.eval()    # switch model to evaluation mode

        print('Planned Path:')
        self.print_dqn(policy_dqn)

        for i in range(episodes):
            path="start"
            counter = 0
            state = env.reset()[0]  # Initialize to state 0
            terminated = False      # True when agent falls in hole or reached goal
            truncated = False       # True when agent takes more than 200 actions

            # Agent navigates map until it falls into a hole (terminated), reaches goal (terminated), or has taken 200 actions (truncated).
            while(not terminated and not truncated):
                # Select best action
                counter += 1
                with torch.no_grad():
                    action = policy_dqn(self.state_to_dqn_input(state, num_states)).argmax().item()

                    if action==0:
                        path+=", left"
                    elif action==1:
                        path+=", down"
                    elif action==2:
                        path+=", right"
                    elif action==3:
                        path+=", up"

                # Execute action
                state, reward, terminated, truncated, _ = env.step(action)

                if (counter%4) == 0:
                    path+="\n"

            print(path+", goal")
            break

        env.close()

    # Print DQN: state, best action, q values
    def print_dqn(self, dqn):
        # Get number of input nodes
        num_states = dqn.input_nodes

        # Loop each state and print policy to console
        print("Planned Path:")
        for s in range(num_states):
            #  Format q values for printing
            q_values = ''
            for q in dqn(self.state_to_dqn_input(s, num_states)).tolist():
                q_values += "{:+.2f}".format(q)+' '  # Concatenate q values, format to 2 decimals
            q_values=q_values.rstrip()              # Remove space at the end

            # Map the best action to L D R U
            best_action = self.ACTIONS[dqn(self.state_to_dqn_input(s, num_states)).argmax()]

            # Print policy in the format of: state, action, q values
            # The printed layout matches the FrozenLake map.

            print(f'{s:02},{best_action}', end=' ')
            if (s+1)%4==0:
                print() # Print a newline every 4 states

if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK']='True'

    is_slippery = False
    map = "8x8"
    episodes = 10000

    crop_field = CropFieldDQL()
    crop_field.train(episodes, is_slippery=is_slippery, map=map)
    # crop_field.test(20, is_slippery=is_slippery, map=map)
