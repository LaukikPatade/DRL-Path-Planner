# This is an env file
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt



class Continuous2DEnvWithRectObstaclesBox:
    def __init__(self,start=[1,8],goal=[16,1],layout="simple"):
        # Action space: move in x and y direction [-1, 1]
        low = np.array([0.0, 0.0], dtype=np.float32)
        high = np.array([20.0,20.0], dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        # Observation space: position in x and y
        self.visited_states=set()
        self.start = np.array(start, dtype=np.float32) 
        self.goal = np.array(goal, dtype=np.float32) 
        # Define rectangular obstacles (each obstacle is defined by its bottom-left corner and size)
        maps={
        "simple":[
            {'bottom_left': np.array([2.0, 10.0]), 'width': 3.0, 'height': 6.0},
            {'bottom_left': np.array([7.0, 15.0]), 'width': 5.0, 'height': 3.0},
            {'bottom_left': np.array([13.0, 12.0]), 'width': 4.0, 'height': 5.0},
            {'bottom_left': np.array([9.0, 5.0]), 'width': 5.0, 'height': 2.0},
            {'bottom_left': np.array([3.0, 3.0]), 'width': 3.0, 'height': 2.0},
            {'bottom_left': np.array([14.0, 3.0]), 'width': 3.0, 'height': 3.0}
        ],
        "moderate":[
            {'bottom_left': np.array([2.0, 14.0]), 'width': 5.0, 'height': 2.0},
            {'bottom_left': np.array([9.0, 3.0]), 'width': 3.0, 'height': 5.0},
            {'bottom_left': np.array([5.0, 6.0]), 'width': 2.0, 'height': 6.0},
            {'bottom_left': np.array([14.0, 6.0]), 'width': 4.0, 'height': 4.0},
            {'bottom_left': np.array([16.0, 14.0]), 'width': 3.0, 'height': 5.0},
            {'bottom_left': np.array([11.0, 16.0]), 'width': 2.0, 'height': 2.0},
            {'bottom_left': np.array([4.0, 1.0]), 'width': 2.0, 'height': 2.0}
            ],
        "complex":[
            {'bottom_left': np.array([1.0, 3.0]), 'width': 3.0, 'height': 7.0},
            {'bottom_left': np.array([6.0, 14.0]), 'width': 5.0, 'height': 2.0},
            {'bottom_left': np.array([8.0, 8.0]), 'width': 2.0, 'height': 6.0},
            {'bottom_left': np.array([11.0, 2.0]), 'width': 4.0, 'height': 3.0},
            {'bottom_left': np.array([13.0, 10.0]), 'width': 6.0, 'height': 2.0},
            {'bottom_left': np.array([18.0, 1.0]), 'width': 2.0, 'height': 5.0},
            {'bottom_left': np.array([10.0, 16.0]), 'width': 3.0, 'height': 4.0}
            ]
        }
        
        self.obstacles =maps[layout]
        self.reset()

    def reset(self):
        self.state = self.start  # Starting position
        # self.observation=self.state
        # self.goal = np.array([18.0, 15.0], dtype=np.float32)  # Goal position for ddpg_box v3
         # Goal position for ddpg_box v4
        self.max_steps = 100
        self.current_step = 0
        return self.state

    def step(self, action):
        self.current_step += 1
        self.state = self.state + action
        self.state = np.clip(self.state, 0, 20)
        reward=0

        distance_to_goal = np.linalg.norm(self.goal - self.state)
        max_distance=np.linalg.norm(self.goal - self.start)
        if distance_to_goal < 0.7:
           reward+=5
        else:
            if self._is_collision_with_obstacle() :
                reward-=2
            if self._is_close_to_the_boundary() :
                reward-=1
            if tuple(self.state) in self.visited_states:  # Penalty for repeated states
                reward -= 0.5
            reward-=0.01
            reward-=distance_to_goal/max_distance
            # reward+=1/distance_to_goal
            self.visited_states.add(tuple(self.state))

        done = distance_to_goal < 0.7 or  self.current_step >= self.max_steps
        info = {}
        # print(reward)
        return self.state, reward, done, info
    # def step(self, action):
    #     self.current_step += 1
    #     self.state = self.state + action
    #     self.state = np.clip(self.state, 0, 20)
    #     reward=0

    #     distance_to_goal = np.linalg.norm(self.goal - self.state)
    #     max_distance=np.linalg.norm(self.goal - self.start)
    #     if distance_to_goal < 0.7:
    #        reward+=5
    #     elif self._is_collision_with_obstacle() :
    #             reward-=2
    #     elif self._is_close_to_the_boundary() :
    #             reward-=1
        


    #     done = distance_to_goal < 0.7 or  self.current_step >= self.max_steps
    #     print(distance_to_goal,done)

    #     if tuple(self.state) in self.visited_states:  # Penalty for repeated states
    #             reward -= 0.5

    #     reward-=0.01
    #     reward-=distance_to_goal/max_distance
    #     self.visited_states.add(tuple(self.state))

        
    #     info = {}
    #     return self.state, reward, done, info

        


    def _is_collision_with_obstacle(self):
        buffer=0.2
        """Check if the agent's current position collides with any rectangular obstacle."""
        for obstacle in self.obstacles:
            x_in_range = obstacle['bottom_left'][0]-buffer <= self.state[0] <= (obstacle['bottom_left'][0] + obstacle['width'] + buffer)
            y_in_range = obstacle['bottom_left'][1] - buffer <= self.state[1] <= (obstacle['bottom_left'][1] + obstacle['height']+buffer)
            if x_in_range and y_in_range:
                return True
        return False
    
    def _is_close_to_the_boundary(self):
        """Check if the agent's current position is dangerously close to the boundary."""
        return (self.state[0]-0)<0.3 or (20-self.state[0])<0.3 or (20-self.state[1])<0.3 or (self.state[1]-0)<0.3


    def render(self, trajectory=None):
        """Render the environment with rectangular obstacles and the agent's trajectory."""
        plt.figure(figsize=(8, 6))

        # Plot the obstacles
        for obstacle in self.obstacles:
            rect = plt.Rectangle(obstacle['bottom_left'], obstacle['width'], obstacle['height'], color='black', alpha=0.7)
            plt.gca().add_patch(rect)

        # Plot the start point
        plt.scatter(self.start[0], self.start[1], color='green', s=100, label='Start')

        # Plot the goal point
        plt.scatter(self.goal[0], self.goal[1], color='red', s=100, label='Goal')

        # Plot a translucent circle around the goal area
        goal_area = plt.Circle(self.goal, radius=0.7, color='red', alpha=0.3, label='Goal Area', fill=True)
        plt.gca().add_patch(goal_area)

        # Plot the trajectory if available
        if trajectory is not None:
            trajectory = np.array(trajectory)
            plt.plot(trajectory[:, 0], trajectory[:, 1], marker='o', color='blue', label='Agent Path')

        plt.xlim(0, 20)
        plt.ylim(0, 20)
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title('Agent Trajectory with Rectangular Obstacles')
        plt.legend()
        plt.grid(True)
        plt.show()
