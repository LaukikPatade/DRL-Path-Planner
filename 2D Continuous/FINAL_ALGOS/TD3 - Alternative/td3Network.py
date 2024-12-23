
from replayBuffer import ReplayBuffer
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



#Twin Delayed Deep Deterministic Policy Gradient (TD3) implementation for continuous action space control.
#Paper: https://arxiv.org/pdf/1802.09477.pdf
#Author: https://github.com/sfujim/TD3


class Actor(nn.Module):
    def __init__(self, state_size, action_size, max_action, fc1=256, fc2=256):
        """
        Initializes actor object.
        @Param:
        1. state_size: env.observation_space.shape[0].
        2. action_size: env.action_space.shape[0].
        3. max_action: abs(env.action_space.low), sets boundary/clip for policy approximation.
        4. fc1: number of hidden units for the first fully connected layer, fc1. Default = 256.
        5. fc2: number of hidden units for the second fully connected layer, fc1. Default = 256.
        """
        super(Actor, self).__init__()

        #Layer 1
        self.fc1 = nn.Linear(state_size, fc1)
        #Layer 2
        self.fc2 = nn.Linear(fc1, fc2)
        #Layer 3
        self.mu = nn.Linear(fc2, action_size)

        #Define boundary for action space.
        self.max_action = max_action
    
    def forward(self, state):
        """Peforms forward pass to map state--> pi(s)"""
        #Layer 1
        x = self.fc1(state)
        x = F.relu(x)
        #Layer 2
        x = self.fc2(x)
        x = F.relu(x)
        #Output layer
        mu = torch.tanh(self.mu(x))#set action b/w -1 and +1
        return self.max_action * mu


class Critic(nn.Module):
    def __init__(self, state_size, action_size, fc1=256, fc2=256):
        """
        Initializes Critic object, Q1 and Q2.
        Architecture different from DDPG. See paper for full details.
        @Param:
        1. state_size: env.observation_space.shape[0].
        2. action_size: env.action_space.shape[0].
        3. fc1: number of hidden units for the first fully connected layer, fc1. Default = 256.
        4. fc2: number of hidden units for the second fully connected layer, fc1. Default = 256.
        """
        super(Critic, self).__init__()

        #---------Q1 architecture---------
        
        #Layer 1
        self.l1 = nn.Linear(state_size + action_size, fc1)
        #Layer 2
        self.l2 = nn.Linear(fc1, fc2)
        #Output layer
        self.l3 = nn.Linear(fc2, 1)#Q-value

        #---------Q2 architecture---------

        #Layer 1
        self.l4 = nn.Linear(state_size + action_size, fc1)
        #Layer 2
        self.l5 = nn.Linear(fc1, fc2)
        #Output layer
        self.l6 = nn.Linear(fc2, 1)#Q-value
    
    def forward(self, state, action):
        """Perform forward pass by mapping (state, action) --> Q-value"""
        x = torch.cat([state, action], dim=1) #concatenate state and action such that x.shape = state.shape + action.shape

        #---------Q1 critic forward pass---------
        #Layer 1
        q1 = F.relu(self.l1(x))
        #Layer 2
        q1 = F.relu(self.l2(q1))
        #value prediction for Q1
        q1 = self.l3(q1)

        #---------Q2 critic forward pass---------
        #Layer 1
        q2 = F.relu(self.l4(x))
        #Layer 2
        q2 = F.relu(self.l5(q2))
        #value prediction for Q2
        q2 = self.l6(q2)

        return q1, q2

#implementation from paper: https://arxiv.org/pdf/1802.09477.pdf
#source: https://github.com/sfujim/TD3/blob/master/TD3.py

#Set to cuda (gpu) instance if compute available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Agent():
    """Agent that plays and learn from experience. Hyper-paramters chosen from paper."""
    def __init__(
            self, 
            state_size, 
            action_size, 
            max_action,
            dir_name, 
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2
        ):
        """
        Initializes the Agent.
        @Param:
        1. state_size: env.observation_space.shape[0]
        2. action_size: env.action_size.shape[0]
        3. max_action: list of max values that the agent can take, i.e. abs(env.action_space.high)
        4. discount: return rate
        5. tau: soft target update
        6. policy_noise: noise reset level, DDPG uses Ornstein-Uhlenbeck process
        7. noise_clip: sets boundary for noise calculation to prevent from overestimation of Q-values
        8. policy_freq: number of timesteps to update the policy (actor) after
        """
        super(Agent, self).__init__()

        #Actor Network initialization
        self.actor = Actor(state_size, action_size, max_action).to(device)
        self.actor.apply(self.init_weights)
        self.actor_target = copy.deepcopy(self.actor) #loads main model into target model
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=0.0005)

        #Critic Network initialization
        self.critic = Critic(state_size, action_size).to(device)
        self.critic.apply(self.init_weights)
        self.critic_target = copy.deepcopy(self.critic) #loads main model into target model
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=0.0005)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.total_it = 0
        self.dir_name=dir_name

    def init_weights(self, layer):
        """Xaviar Initialization of weights"""
        if(type(layer) == nn.Linear):
          nn.init.xavier_normal_(layer.weight)
          layer.bias.data.fill_(0.01)

    def select_action(self, state):
        """Selects an automatic epsilon-greedy action based on the policy"""
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()
    
    def train(self, replay_buffer:ReplayBuffer):
        """Train the Agent"""

        self.total_it += 1

        # Sample replay buffer 
        state, action, reward, next_state, done = replay_buffer.sample()#sample 256 experiences

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)
            

            next_action = (
                self.actor_target(next_state) + noise #noise only set in training to prevent from overestimation
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action) #Q1, Q2
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1 - done) * self.discount * target_Q #TD-target

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action) #Q1, Q2

        # Compute critic loss using MSE
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates (DDPG baseline = 1)
        if(self.total_it % self.policy_freq == 0):

            # Compute actor loss
            actor_loss = -self.critic(state, self.actor(state))[0].mean()
            
            # Optimize the actor 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Soft update by updating the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


    def save(self, filename):
        """Saves the Actor Critic local and target models"""
        torch.save(self.critic.state_dict(), f"tmp/{self.dir_name}/"+ filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), f"tmp/{self.dir_name}/" + filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), f"tmp/{self.dir_name}/" + filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), f"tmp/{self.dir_name}/" + filename + "_actor_optimizer")


    def load(self, filename):
        """Loads the Actor Critic local and target models"""
        self.critic.load_state_dict(torch.load(f"tmp/{self.dir_name}/" + filename + "_critic", map_location='cpu',weights_only=True))
        self.critic_optimizer.load_state_dict(torch.load(f"tmp/{self.dir_name}/" + filename + "_critic_optimizer", map_location='cpu',weights_only=True))#optional
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(f"tmp/{self.dir_name}/" + filename + "_actor", map_location='cpu',weights_only=True))
        self.actor_optimizer.load_state_dict(torch.load(f"tmp/{self.dir_name}/" + filename + "_actor_optimizer", map_location='cpu',weights_only=True))#optional
        self.actor_target = copy.deepcopy(self.actor)