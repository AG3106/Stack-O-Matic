from tetris_env import TetrisEnv as env
import matplotlib.pyplot as plt
import random
import math
from collections import namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
from model import DQN
from model import ReplayMemory

#set device (gpu if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

#Transition maps state, action pair to next state and reward
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


#Parameters for DQN
BATCH_SIZE = 128
GAMMA = 0.99
EPSILON_START = 0.9
EPSILON_END = 0.1
EPSILON_DECAY = 30000
TAU = 0.005
lr =1e-4

#Setting up Tetris environment
env = env(6,12)
n_actions = env.n_actions #using env.action_space.n since action size is discrete
state = env.clear()
n_observations = (env.width * env.height)

#Creating Policy Network, Target Network
policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict()) # making weights of target network equal of that of policy network

#Setting Optimizer and memory of capacity 100000
optimizer = optim.AdamW(policy_net.parameters(), lr=lr)
memory = ReplayMemory(100000)
steps_done = 0

# select_action function selects action to be taken with epsilon greedy algorithm

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPSILON_END + (EPSILON_START - EPSILON_END) * math.exp(-1. * steps_done / EPSILON_DECAY) #Exponentially decaying epsilon-greedy exploration
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1,1)
    else:
        return torch.tensor([[random.randint(0,5)]], device=device) #random action selection

# For visualizing rewards and scores over training episodes
episode_reward = []
episode_score = []
def plot(arr,yaxis_name):
    plt.figure(1)
    total_reward = torch.tensor(arr, dtype=torch.float)
    plt.title('Result')
    plt.xlabel('Episode')
    plt.ylabel(yaxis_name)
    plt.plot(total_reward.numpy())
    plt.show()

# optimize function to optimize the DQN
def optimize():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device,
                                  dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values

    expected_state_action_values = reward_batch + GAMMA * next_state_values  # bellman equation

    criterion = nn.SmoothL1Loss()  # Huber Loss
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))  # unsqueeze for changing dimension

    optimizer.zero_grad()
    loss.backward()  # backpropogation

    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

max_treward = 0
max_score = 0

#Training
for i_episode in range(10000):
    print()
    print(i_episode)
    print()
    treward = 0
    state = env.clear()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    while True:
        #env.view()
        action = select_action(state)
        observation, rewardn, terminated, score = env.step(action.item())
        reward = torch.tensor([rewardn], device=device)
        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float, device=device).unsqueeze(0)
        treward += rewardn
        memory.push(state, action, next_state, reward)
        state = next_state
        optimize()
        # updating target network slowly
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
        target_net.load_state_dict(target_net_state_dict)

        if terminated:
            if treward >= max_treward:
                max_treward = treward
                print('GOOD EPISODE: \n Total Reward:', max_treward)
                torch.save(policy_net.state_dict(), './policy_net_max.pth')  # saving Policy Network with maximum reward
                max_score = score
            episode_reward.append(treward)
            episode_score.append(score)
            break

print('Finished Training')
torch.save(policy_net.state_dict(), './policy_net_final.pth')
plot(episode_score, yaxis_name= 'Score')
plot(episode_reward,yaxis_name = 'Total Reward')
print("Max Total reward: ", max_treward, max_score)
print('Final Total reward: ', episode_reward[-1])