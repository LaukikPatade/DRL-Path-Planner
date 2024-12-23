import numpy as np
import torch
import gymnasium as gym
import random
import matplotlib.pyplot as plt
from EnvWithObstacles import Continuous2DEnvWithRectObstaclesBox
import utils
from td3Network import Agent
from replayBuffer import ReplayBuffer
import argparse
import os
from torch.utils.tensorboard import SummaryWriter

def create_new_directory(base_path="tmp/"):
    new_dir = input("Enter the name for the new directory: ")
    os.makedirs(os.path.join(base_path, new_dir), exist_ok=True)
    x_start = float(input("Enter the x-coordinate of the starting pos: "))
    y_start = float(input("Enter the y-coordinate of the starting pos: "))
    x_end = float(input("Enter the x-coordinate of the goal: "))
    y_end = float(input("Enter the y-coordinate of the goal: "))
    layout = input("Enter the layout type: ")
    return new_dir,layout, [x_start, y_start],[x_end, y_end]
#set seeds
# random_seed = 1
# env.seed(random_seed)
# np.random.seed(random_seed)
# torch.manual_seed(random_seed)
# random.seed(random_seed)


#Set exploration noise for calculating action based on some noise factor
def train(env,dir_name):

    writer = SummaryWriter(f'../runs/{dir_name}')
    exploration_noise = 0.1

    #Define observation and action space
    state_space = env.observation_space.shape[0]
    action_space = env.action_space.shape[0] 
    max_action = float(env.action_space.high[0])

    #Create Agent
    policy = Agent(state_space, action_space, max_action,dir_name)

    # try:
    #     policy.load("temp")
    # except:
    #     raise IOError("Couldn't load policy")

    #Create Replay Buffer
    replay_buffer = ReplayBuffer()

    stability=0
    threshold=20
    #Train the model
    max_episodes = 4_000
    max_timesteps = 2_000

    ep_reward = [] #get list of reward for range(max_episodes)
    episode=0
    while(True):
        episode+=1
        avg_reward = 0
        state = env.reset()
        steps=0
        for t in range(1, max_timesteps + 1):
            steps+=1
            # select action and add exploration noise:
            action = policy.select_action(state) + np.random.normal(0, max_action * exploration_noise, size=action_space)
            action = action.clip(env.action_space.low, env.action_space.high)
                
            # take action in env:
            next_state, reward, done, _ = env.step(action)
            replay_buffer.add(state, action, reward, next_state, done)
            state = next_state
                
            avg_reward += reward

            #Renders an episode
            # env.render()
            
            if(len(replay_buffer) > 100):#make sure sample is less than overall population
                policy.train(replay_buffer) #training mode

            # if episode is done then update policy:
            if(done or t >=max_timesteps):
                print(f"Episode {episode} reward: {avg_reward} | Rolling average: {np.mean(ep_reward)}")
                print(f"Current time step: {t}")
                
                ep_reward.append(avg_reward)
                break 
        writer.add_scalar('Reward collected', avg_reward, episode)
        writer.add_scalar('Steps taken', steps, episode)
        if(np.mean(ep_reward[-100:]) >= -5):
            policy.save("final")
            break

        if(episode % 100 == 0 and episode > 0):
            #Save policy and optimizer every 100 episodes
            policy.save("temp")

def test(env,dir_name):
    exploration_noise = 0.1
    #Define observation and action space
    state_space = env.observation_space.shape[0]
    action_space = env.action_space.shape[0] 
    max_action = float(env.action_space.high[0])

    #Create Agent
    policy = Agent(state_space, action_space, max_action,dir_name)
    try:
        policy.load("temp")
    except:
        raise IOError("Couldn't load policy")


    #Train the model
    max_episodes = 1
    max_timesteps = 2000

    ep_reward = [] #get list of reward for range(max_episodes)

    avg_reward = 0
    state = env.reset()
    trajectory=[]
    trajectory.append(state)   
    steps=0
    path_length=0
    while(True):
        steps+=1
        # select action and add exploration noise:
        action = policy.select_action(state) + np.random.normal(0, max_action * exploration_noise, size=action_space)
        action = action.clip(env.action_space.low, env.action_space.high)
            
        # take action in env:
        next_state, reward, done, _ = env.step(action)
        path_length+=np.linalg.norm(next_state - state)
        state = next_state
            
        avg_reward += reward

        #Renders an episode
        trajectory.append(state)
        # if episode is done then display stats:
        if(done or steps >=max_timesteps):
            print(f"reward: {avg_reward} | Rolling average: {np.mean(ep_reward)}")
            print(f"Current time step: {steps}")
            ep_reward.append(avg_reward)
            break 
    print("path length: ",path_length)
    env.render(trajectory)


def main():
    
    parser = argparse.ArgumentParser(description="Train or Test the model")
    parser.add_argument("mode", type=str, choices=["train", "test"], 
                        help="Mode to run the program: 'train' or 'test'")
    # parser.add_argument("dir", type=str)
    args = parser.parse_args()
    if args.mode==None:
        print("Enter a mode")
    else:
        map={
            "td3_v2":{
                "start":[1,8],
                "goal":[18,15],
                "layout":"simple"
            },
            "td3_v3":{
                "start":[1,8],
                "goal":[18,15],
                "layout":"simple"
            },
            "td3_v4":{
                "start":[1,8],
                "goal":[3,17.5],
                "layout":"simple"
            },
            "td3_simple":{
                "start":[1,1],
                "goal":[18,15],
                "layout":"simple"
        },
        "td3_complex":{
                "start":[2.5,17.5],
                "goal":[16,1],
                "layout":"complex"
        },
        "td3_moderate":{
                "start":[1,1],
                "goal":[14,17.5],
                "layout":"moderate"
        }
        }
        
        if args.mode == "train":
            dirs = [d for d in os.listdir("tmp/")]
            print(dirs)
            choice = input("Do you want to train an existing model or create a new one? (existing/new): ").lower()
            if choice == "existing":
                dir_name = input("Enter the directory name from the list: ")
                if dir_name in dirs:
                    env=Continuous2DEnvWithRectObstaclesBox(start=map[dir_name]["start"],goal=map[dir_name]["goal"],layout=map[dir_name]["layout"])
                    # agent=Agent(alpha=0.000025,beta=0.00025,input_dims=[2],tau=0.001,epsilon_start = 1.0,epsilon_min= 0.1,epsilon_decay= 1e-6,env=env,batch_size=64,layer1_size=400,layer2_size=200,n_actions=2,chkpt_dir="tmp/"+dir_name)
                else:
                    print("Directory not found.")
                    return
            elif choice == "new":
                dir_name, layout,start,goal = create_new_directory()
                env=Continuous2DEnvWithRectObstaclesBox(start,goal,layout)
            else:
                print("Invalid choice. Exiting.")
                return
            
            
            train(env,dir_name)

        elif args.mode == "test":
            dirs = [d for d in os.listdir("tmp/")]
            print(dirs)
            dir_name = input("Enter the directory name from the list: ")
            if dir_name in dirs:
                    env=Continuous2DEnvWithRectObstaclesBox(start=map[dir_name]["start"],goal=map[dir_name]["goal"],layout=map[dir_name]["layout"])
            else:
                print("Directory not found.")
                return
            test(env,dir_name)
        else:
            print("Invalid mode. Use 'train' or 'test'.")

if __name__ == "__main__":
    main()





