
from unityagents import UnityEnvironment
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

PATH = ''
if os.path.isfile('checkpoint_actor1.pth') and os.access('checkpoint_actor1.pth', os.R_OK):
    agent1 = Agent(state_size=state_size, action_size=action_size, random_seed=42, actor_filepath = 'checkpoint_actor1.pth',
               critic_filepath = 'checkpoint_critic1.pth')
    agent2 = Agent(state_size=state_size, action_size=action_size, random_seed=42, memory = agent1.memory,
               actor_filepath = 'checkpoint_actor2.pth', critic_filepath = 'checkpoint_critic2.pth')
else:
    agent1 = Agent(state_size=state_size, action_size=action_size, random_seed=42)
    agent2 = Agent(state_size=state_size, action_size=action_size, random_seed=42, memory=agent1.memory)

STOP_NOISE_AFTER_EP = 300

def ddpg(n_episodes=10000, print_every=100, solved_score = 0.5, train_mode = True, consec_episodes = 100):
    scores_deque = deque(maxlen=print_every)
    scores = []
    best_score = float('-inf')
    add_noise = True
    for i_episode in range(1, n_episodes + 1):

        if i_episode > STOP_NOISE_AFTER_EP:
            add_noise = False
        env_info = env.reset(train_mode=train_mode)[brain_name]  # reset environment
        states = env_info.vector_observations  # get current state for each agent
        agent1.reset()
        agent2.reset()
        scores_episode = np.zeros(num_agents)
        start_time = time.time()

        steps = 0
        while True:
            steps += 1
            action1 = agent1.act(states[0], add_noise)
            action2 = agent2.act(states[1], add_noise)
            actions = [action1,action2]
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations  # get next state
            rewards = env_info.rewards  # get reward
            dones = env_info.local_done
            agent1.step(states[0], action1, rewards[0], next_states[0], dones[0], steps)
            agent2.step(states[1], action2, rewards[1], next_states[1], dones[1], steps)
            states = next_states
            scores_episode += rewards
            if np.any(dones):
                break

            #print(rewards)
        episode_best_score = np.max(scores_episode)
        scores_deque.append(episode_best_score)
        scores.append(episode_best_score)

        if episode_best_score > best_score:
            best_score = episode_best_score
            best_episode = i_episode
            if train_mode:
                torch.save(agent1.actor_local.state_dict(), 'checkpoint_actor1.pth')
                torch.save(agent1.critic_local.state_dict(), 'checkpoint_critic1.pth')
                torch.save(agent2.actor_local.state_dict(), 'checkpoint_actor2.pth')
                torch.save(agent2.critic_local.state_dict(), 'checkpoint_critic2.pth')

        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))

        if np.mean(scores_deque) >= solved_score:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode,np.mean(scores_deque)))
            # if train_mode:
            #     torch.save(agent1.actor_local.state_dict(), 'checkpoint_actor1.pth')
            #     torch.save(agent1.critic_local.state_dict(), 'checkpoint_critic1.pth')
            #     torch.save(agent2.actor_local.state_dict(), 'checkpoint_actor2.pth')
            #     torch.save(agent2.critic_local.state_dict(), 'checkpoint_critic2.pth')
            break

    return scores

scores = ddpg()


fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()
