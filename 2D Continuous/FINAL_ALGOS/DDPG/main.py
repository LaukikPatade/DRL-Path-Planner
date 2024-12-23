# This is the main executable file
import gymnasium as gym
from ddpgNetwork import Agent
import numpy as np
from utils import plotLearning
from EnvWithObstacles import Continuous2DEnvWithRectObstaclesBox
import argparse
import os
from torch.utils.tensorboard import SummaryWriter


def train(env,dir_name):
    writer = SummaryWriter(f'../runs/{dir_name}')
    np.random.seed(0)
    # episodes=5000
    agent=Agent(alpha=0.0001,
                beta=0.001,
                input_dims=[2],
                tau=0.001,
                epsilon_start = 1.0,
                epsilon_min= 0.1,
                epsilon_decay= 1e-6,
                env=env,
                batch_size=64,
                layer1_size=400,
                layer2_size=200,
                n_actions=2,
                chkpt_dir="tmp/"+dir_name)
    # agent.load_models()
    avg_score=0
    episode=0
    score_history=[]
    steps_history=[]
    threshold=30
    stability=0
    threshold_ep=0
    stop_steps=30
    flag=False
    avg_steps=10000000000
    while(True):
        if(avg_steps<=30):
            stability+=1
            if not flag:
                threshold_ep=episode
                flag=True
        episode+=1
        observation=env.reset()
        done=False
        total_reward=0
        steps=0
        # print(observation)
        while not done:
            steps+=1
            action=agent.choose_action(observation)
            new_state, reward, done, info =env.step(action)
            agent.remember(observation,action,reward,new_state,done)
            agent.learn()
            total_reward+=reward
            observation=new_state
        writer.add_scalar('Reward collected', total_reward, episode)
        writer.add_scalar('Steps taken', steps, episode)
        score_history.append(total_reward)
        steps_history.append(steps)
        if episode % 25==0:
            agent.save_models()
        avg_score=np.mean(score_history[-100:])
        avg_steps=np.mean(steps_history[-100:])
        print('episode ', episode, '| score:%.2f' % total_reward,'| steps taken: ',steps,
              '| Avg score for trailing hundred games:%.3f' % avg_score ,'| Epsilon:%.4f'% agent.epsilon)
        if (avg_score>-5 and avg_steps<=30):
            stability_perc=stability/(episode-threshold_ep)
            print(f"Our model learnt in {episode} episodes and gave a stability percentage of {stability_perc}")
            break

    filename1 = 'E:/Research paper/FINAL_ALGOS/DDPG/'+agent.chkpt_dir+'/DDPG_REWARDS.png'
    plotLearning(score_history, filename1,"Episodes","Rewards Collected", window=100)
    filename2 = 'E:/Research paper/FINAL_ALGOS/DDPG/'+agent.chkpt_dir+'/DDPG_STEPS.png'
    plotLearning(steps_history, filename2,"Episodes","Steps Taken", window=100)

def test(env,dir_name):
    agent=Agent(alpha=0.0001,
                beta=0.001,
                input_dims=[2],
                tau=0.001,
                epsilon_start = 1.0,
                epsilon_min= 0.1,
                epsilon_decay= 1e-6,
                env=env,
                batch_size=64,
                layer1_size=400,
                layer2_size=200,
                n_actions=2,
                chkpt_dir="tmp/"+dir_name)
    agent.load_models()
    observation=env.reset()
    done=False
    total_reward=0
    trajectory=[]
    trajectory.append(observation)
    path_length=0
    while not done:
        
        action=agent.choose_action(observation)
        new_state, reward, done, info =env.step(action)
        path_length+=np.linalg.norm(new_state - observation)
        total_reward+=reward
        observation=new_state
        trajectory.append(observation)
    print(f"hello {total_reward} {path_length}")
    env.render(trajectory)










def create_new_directory(base_path="tmp/"):
    new_dir = input("Enter the name for the new directory: ")
    os.makedirs(os.path.join(base_path, new_dir), exist_ok=True)
    x_start = float(input("Enter the x-coordinate of the starting pos: "))
    y_start = float(input("Enter the y-coordinate of the starting pos: "))
    x_end = float(input("Enter the x-coordinate of the goal: "))
    y_end = float(input("Enter the y-coordinate of the goal: "))
    layout = input("Enter the layout type: ")
    return new_dir,layout, [x_start, y_start],[x_end, y_end]

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
            "ddpg_v2":{
                "goal":[18,15],
                "start":[1,8],
                "layout":"simple"
            },
            "ddpg_v3":{
                "start":[1,8],
                "goal":[18,15],
                "layout":"simple"
            },
            "ddpg_v4":{
                "start":[1,8],
                "goal":[16,1],
                "layout":"simple"
            },
            "ddpg_v5":{
                "start":[1,8],
                "goal":[3,17.5],
                "layout":"simple"
            }
            ,
            "ddpg_complex":{
                "start":[2.5,17.5],
                "goal":[16,1],
                "layout":"complex"
            },
            "ddpg_simple":{
                "start":[1,1],
                "goal":[18,15],
                "layout":"simple"
            },
            "ddpg_simple_v2":{
                "start":[1,1],
                "goal":[18,15],
                "layout":"simple"
            },
            "ddpg_simple_v3":{
                "start":[1,1],
                "goal":[18,15],
                "layout":"simple"
            },
            "ddpg_moderate":{
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


