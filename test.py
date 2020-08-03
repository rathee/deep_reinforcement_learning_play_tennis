import torch
from collections import deque
import matplotlib.pyplot as plt
from unityagents import UnityEnvironment
import numpy as np
import time
from ddpg_agent import Agent
import os

env = UnityEnvironment(file_name="Tennis.app")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])

agent1 = Agent(state_size=state_size, action_size=action_size, random_seed=42,
               actor_filepath = 'trained_weights/checkpoint_actor1.pth',
               critic_filepath = 'trained_weights/checkpoint_critic1.pth')
agent2 = Agent(state_size=state_size, action_size=action_size, random_seed=42, memory = agent1.memory,
               actor_filepath = 'trained_weights/checkpoint_actor2.pth',
               critic_filepath = 'trained_weights/checkpoint_critic2.pth')

env_info = env.reset(train_mode=False)[brain_name]     # reset the environment
states = env_info.vector_observations                  # get the current state (for each agent)


for i in range(10):
    scores = np.zeros(num_agents)  # initialize the score (for each agent)
    while True:
        action1 = agent1.act(states[0], add_noise = False)
        action2 = agent2.act(states[1], add_noise = False)
        actions = [action1, action2]
        env_info = env.step(actions)[brain_name]           # send all actions to tne environment
        next_states = env_info.vector_observations         # get next state (for each agent)
        rewards = env_info.rewards                         # get reward (for each agent)
        dones = env_info.local_done                        # see if episode finished
        scores += env_info.rewards                         # update the score (for each agent)
        states = next_states                               # roll over states to next time step
        if np.any(dones):                                  # exit loop if episode finished
            break

    print("Episode %d best score is %f" % (i, np.mean(scores)))