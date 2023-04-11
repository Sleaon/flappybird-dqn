#!/usr/bin/env python
from __future__ import print_function

import sys
import os
import cv2

from DQN import *

sys.path.append("game/")
import game.wrapped_flappy_bird as game
from tqdm import tqdm


os.environ["SDL_VIDEODRIVER"] = "dummy"

class Train():
    def __init__(self, game_state, actions_dim, input_size=(224, 224), hidden_dim=256, batch_size=32, lr=2e-3,
                 num_episodes=1000, gamma=0.98, epsilon=1, target_update=10,
                 buffer_size=10000, minimal_size=500):
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        self.game_state = game_state
        self.actions_dim = actions_dim
        self.input_size = input_size
        self.batch_size = batch_size
        self.num_episodes = num_episodes
        self.minimal_size = minimal_size
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
            "cpu")
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.agent = DQNCNN(hidden_dim, lr, gamma, epsilon,
                            target_update, self.device)
        if self.agent.load_model():
            print("model load success")
        self.count = 0

    def get_evn(self, action: int):
        do = np.zeros([self.actions_dim], dtype=np.int64)
        do[action] = 1
        next_stats_colored, reward, terminal = self.game_state.frame_step(do)
        next_stats = cv2.cvtColor(cv2.resize(next_stats_colored, self.input_size), cv2.COLOR_RGB2BGR)
        next_stats = np.transpose(next_stats, (2, 0, 1))
        next_stats = np.expand_dims(next_stats, 0)
        return next_stats, reward, terminal

    def update(self, current_stats, next_stats, action, reward, terminal):
        current_stats = np.squeeze(current_stats, 0)
        next_stats = np.squeeze(next_stats, 0)
        self.replay_buffer.add(current_stats, action, reward, next_stats, terminal)
        if self.replay_buffer.size() > self.minimal_size:
            b_s, b_a, b_r, b_ns, b_d = self.replay_buffer.sample(self.batch_size)
            transition_dict = {
                'states': b_s,
                'actions': b_a,
                'next_states': b_ns,
                'rewards': b_r,
                'dones': b_d
            }
            self.agent.update(transition_dict)

    def manual_callback(self, action, current_stats_raw, next_stats_raw, reward, terminal):
        next_stats = cv2.cvtColor(cv2.resize(next_stats_raw, self.input_size), cv2.COLOR_RGB2BGR)
        next_stats = np.transpose(next_stats, (2, 0, 1))
        next_stats = np.expand_dims(next_stats, 0)
        current_stats = cv2.cvtColor(cv2.resize(current_stats_raw, self.input_size), cv2.COLOR_RGB2BGR)
        current_stats = np.transpose(current_stats, (2, 0, 1))
        current_stats = np.expand_dims(current_stats, 0)
        self.update(current_stats, next_stats, action, reward, terminal)
        if terminal:
            self.count += 1
            print("now %d %%" % (self.count / 20 * 100))
        if self.count > 20:
            return False
        else:
            return True

    def human_demonstration(self):
        self.game_state.set_callback(self.manual_callback)
        self.game_state.manual()
        self.agent.save_model()

    def auto_train(self):
        return_list = []
        for i in range(100):
            with tqdm(total=int(self.num_episodes), desc='Iteration %d' % i) as pbar:
                self.agent.epsilon = self.agent.epsilon * 0.7 if self.agent.epsilon * 0.7 > 0.01 else 0.01
                for i_episode in range(int(self.num_episodes)):
                    episode_return = 0
                    ii = 0
                    state, reward, done = self.get_evn(0)
                    while not done:
                        ii += 1
                        action = self.agent.take_action(state)
                        next_state, reward, done = self.get_evn(action)
                        episode_return += reward
                        self.update(state, next_state, action, reward, done)
                    return_list.append(episode_return)
                    if (i_episode + 1) % 10 == 0:
                        pbar.set_postfix({
                            'episode':
                                '%d' % (self.num_episodes * i + i_episode + 1),
                            'return':
                                '%.3f' % np.mean(return_list[-10:])
                        })
                    pbar.update(1)
            self.agent.save_model()


GAME = 'bird'  # the name of the game being played for log files
ACTIONS = 2  # number of valid actions
game_state = game.GameState()
train = Train(game_state, ACTIONS)
train.human_demonstration()
train.auto_train()
