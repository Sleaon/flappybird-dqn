import collections
import random
import torchvision.models as models
from timm.models.vision_transformer import vit_tiny_patch16_224_in21k
import numpy as np
import torch
import torch.nn.functional as F
import copy

class ReplayBuffer:
    ''' 经验回放池 '''

    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # 队列,先进先出

    def add(self, state, action, reward, next_state, done):  # 将数据加入buffer
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):  # 从buffer中采样数据,数量为batch_size
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):  # 目前buffer中数据的数量
        return len(self.buffer)


class Cnn(torch.nn.Module):
    def __init__(self):
        super(Cnn, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 8, 4, 2)
        self.conv2 = torch.nn.Conv2d(32, 64, 4, 2, 1)
        self.conv3 = torch.nn.Conv2d(64, 64, 3, 1, 1)
        self.pooling = torch.nn.MaxPool2d(2)
        self.pooling2 = torch.nn.MaxPool2d(2, 2, 1)

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.pooling(self.conv1(x)))
        x = F.relu(self.pooling2(self.conv2(x)))
        x = F.relu(self.pooling2(self.conv3(x)))
        x = x.view(batch_size, -1)
        return x


class Qnet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, int(hidden_dim/2))
        self.bn1 = torch.nn.BatchNorm1d(int(hidden_dim/2))
        self.fc2 = torch.nn.Linear(int(hidden_dim/2), hidden_dim)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.bn3 = torch.nn.BatchNorm1d(hidden_dim)
        self.fc4 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class DQN:
    ''' DQN算法 '''

    def __init__(self, state_dim, action_dim, hidden_dim=128, learning_rate=2e-3, gamma=0.98,
                 epsilon=0.01, target_update=10, device="cpu"):
        self.action_dim = action_dim
        self.q_net = Qnet(state_dim, hidden_dim,
                          self.action_dim).to(device)  # Q网络
        # 目标网络
        self.target_q_net = Qnet(state_dim, hidden_dim,
                                 self.action_dim).to(device)
        # 使用Adam优化器
        self.optimizer = torch.optim.Adam(self.q_net.parameters(),
                                          lr=learning_rate)
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略
        self.target_update = target_update  # 目标网络更新频率
        self.count = 0  # 计数器,记录更新次数
        self.device = device

    def take_action(self, state):  # epsilon-贪婪策略采取动作
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
            action = self.q_net(state)
            action = action.argmax().item()
        return action

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)  # Q值
        # 下个状态的最大Q值
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(
            -1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones
                                                                )  # TD误差目标
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数
        self.optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        dqn_loss.backward()  # 反向传播更新参数
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())  # 更新目标网络
        self.count += 1

    def save_model(self):
        torch.save(self.q_net.state_dict(), 'qnet.params')
        torch.save(self.target_q_net.state_dict(), 'target_qnet.params')

    def load_model(self) -> bool:
        try:
            self.q_net.load_state_dict(torch.load('qnet.params'))
            self.q_net.load_state_dict(torch.load('qnet.params'))
            return True
        except Exception:
            return False


class DQNCNN:
    ''' DQN算法，图像作为输入，前置一个resnet'''

    def __init__(self, hidden_dim=128, learning_rate=2e-3, gamma=0.98,
                 epsilon=0.1, target_update=10, device="cpu"):
        images_model = models.resnet50(pretrained=True).to(device)
        for param in images_model.parameters():
            if hasattr(param, 'requires_grad'):
                param.requires_grad = False
        # children = list(vit_model.children())
        q = Qnet(2048,hidden_dim,2).to(device)
        images_model.fc = q
        for param in q.parameters():
            param.requires_grad = True

        # self.q_net = torch.nn.Sequential(
        #     *children[:-1],
        #     Qnet(768, hidden_dim,2)).to(device)
        self.q_net = images_model

        # 目标网络
        # self.target_q_net = torch.nn.Sequential(
        #     *children[:-1],
        #     Qnet(768, hidden_dim,2)).to(device)
        self.target_q_net = copy.deepcopy(images_model)

        # 使用Adam优化器
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略
        self.target_update = target_update  # 目标网络更新频率
        self.count = 0  # 计数器,记录更新次数
        self.device = device

    def take_action(self, state):  # epsilon-贪婪策略采取动作
        if np.random.random() < self.epsilon:
            action = np.random.choice([0, 1], p=[0.9, 0.1]).item()
        else:
            # state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
            state = torch.tensor(np.array(state), dtype=torch.float).to(self.device)
            action = self.q_net(state)
            action = action.argmax().item()
        return action

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'],dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)  # Q值
        # 下个状态的最大Q值
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)  # TD误差目标
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数
        self.optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        dqn_loss.backward()  # 反向传播更新参数
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())  # 更新目标网络
        self.count += 1

    def save_model(self):
        torch.save(self.q_net.state_dict(), 'qnet.params')
        torch.save(self.target_q_net.state_dict(), 'target_qnet.params')

    def load_model(self) -> bool:
        try:
            self.q_net.load_state_dict(torch.load('qnet.params'))
            self.q_net.load_state_dict(torch.load('qnet.params'))
            return True
        except Exception:
            return False
