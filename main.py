import gym
import numpy as np
import torch
import random
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import copy
from collections import deque
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 초기화 : BUF_SIZE = 10000, UPDATE_TARGET = 500 , GAMMA = 0.99, eps_coeff = 0.995

BUF_SIZE = 10000
BATCH_SIZE = 128
UPDATE_TARGET = 700  # 500씩 만큼 target network 업데이트
GAMMA = 0.99

seed = 17
np.random.seed(seed)
random.seed(seed)
env = gym.make('MountainCar-v0')

env.seed(seed)
torch.manual_seed(seed)

class Buffer:
    def __init__(self, cap):
        # cap : buffer size = 10000 (capacity)
        self.cap = cap
        self.mem = []
        self.pos = -1  # 마지막으로 기록된 mem 요소의 위치

    def add(self, element):
        if len(self.mem) < self.cap:
            self.mem.append(None)
        new_pos = (self.pos + 1) % self.cap
        self.mem[new_pos] = element
        self.pos = new_pos

    def sample(self, batch_size):
        return random.sample(self.mem, batch_size)

    def __len__(self):
        return len(self.mem)

    def __getitem__(self, item):
        return self.mem[(self.pos + 1 + item) % self.cap]


class Model(nn.Module):
    # input dim(state dim) = 2, output dim(action dim) : 3
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DQN:
    def __init__(self):
        self.network = Model().to(device)
        self.target_network = copy.deepcopy(self.network).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=5e-4)

    def train(self, batch):
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.from_numpy(np.array(states)).float().to(device)
        actions = torch.from_numpy(np.array(actions)).to(device).unsqueeze(1)
        rewards = torch.from_numpy(np.array(rewards)).float().to(device).unsqueeze(1)
        next_states = torch.from_numpy(np.array(next_states)).float().to(device)
        dones = torch.from_numpy(np.array(dones)).to(device).unsqueeze(1)

        with torch.no_grad():  # Double DQN
            argmax = self.network(next_states).detach().max(1)[1].unsqueeze(1)
            target = rewards + (GAMMA * self.target_network(next_states).detach().gather(1, argmax)) * (~dones)

        Q_current = self.network(states).gather(1, actions.type(torch.int64))
        self.optimizer.zero_grad()
        # loss = F.mse_loss(target, Q_current)
        loss = F.mse_loss(target, Q_current)

        loss.backward()
        self.optimizer.step()

    def act(self, state):
        state = torch.tensor(state).to(device).float()
        with torch.no_grad():
            Q_values = self.network(state.unsqueeze(0))
        return np.argmax(Q_values.cpu().data.numpy())

    def update_target(self):
        self.target_network = copy.deepcopy(self.network)


def transform_state(state):
    state = (np.array(state) + np.array((1.2, 0.0))) / np.array((1.8, 0.07))
    result = []
    result.extend(state)
    return np.array(state)


def eps_greedy(env, dqn, state, eps):
    if random.random() < eps:
        return env.action_space.sample()
    return dqn.act(state)


dqn = DQN()
buf = Buffer(BUF_SIZE)

episodes = 1500
eps = 1
eps_coeff = 0.995
dqn_updates = 0
output_period = 100

rews = deque(maxlen=output_period)  # 각 에피소드에 대한 보상을 여기에 사용한다.
rews_all = [None] * episodes  # 그래프에서 나타낼 rewards를 모두 rews_all에 저장

for i in range(1, episodes + 1):
    state = transform_state(env.reset())
    done = False

    total_reward = 0
    while not done:
        action = eps_greedy(env, dqn, state, eps)
        next_state, reward, done, _ = env.step(action)
        next_state = transform_state(next_state)
        total_reward += reward
        reward += 300 * (GAMMA * abs(next_state[1]) - abs(state[1]))
        buf.add((state, action, reward, next_state, done))
        if len(buf) >= BATCH_SIZE:
            dqn.train(buf.sample(BATCH_SIZE))
            dqn_updates += 1
        if not dqn_updates % UPDATE_TARGET:
            dqn.update_target()
        state = next_state

    eps *= eps_coeff
    rews.append(total_reward)
    rews_all[i - 1] = total_reward

    mean_r = np.mean(rews)
    max_r = np.max(rews)
    min_r = np.min(rews)
    print(f'\repisode {i}, eps = {eps}, mean = {mean_r}, min = {min_r}, max = {max_r}', end="")
    if not i % output_period:
        print(f'\repisode {i}, eps = {eps}, mean = {mean_r}, min = {min_r}, max = {max_r}')

plt.figure(figsize=(15, 7))
plt.plot(range(1, episodes + 1), rews_all)
plt.ylabel('reward')
plt.xlabel('iteration')
plt.show()