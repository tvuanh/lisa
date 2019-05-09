import unittest
from collections import deque

import numpy as np

import gym

from lisa import rl


class TestIntegration(unittest.TestCase):

    def test_integration_copy_v0(self):
        episodes = 3000
        nplays = 10
        results = np.array([play_copy_v0(episodes) for _ in range(nplays)])
        success = results < episodes
        self.assertTrue(np.sum(success) > 0.7 * nplays)
        self.assertTrue(np.mean(results[success]) < 1400)

    def test_integration_frozen_lake_v0(self):
        episodes = 5000
        nplays = 20
        results = np.array([play_frozen_lake_v0(episodes) for _ in range(nplays)])
        success = results < episodes
        self.assertTrue(np.sum(success) > 0.7 * nplays)
        self.assertTrue(np.mean(results[success]) < 3700)

    def test_integration_frozen_lake_8x8_v0(self):
        episodes = 10000
        nplays = 1
        results = np.array(
            [play_frozen_lake_8x8_v0(episodes) for _ in range(nplays)])
        success = results < episodes
        self.assertTrue(np.sum(success) > 0.7 * nplays)
        self.assertTrue(np.mean(results[success]) < 15000)


def play_copy_v0(episodes=3000):
   env = gym.make('Copy-v0')

   states = range(6)
   actions = tuple(
       [(i, j, k) for i in (0, 1) for j in (0, 1) for k in range(5)]
   )
   Qtable = rl.QTable(states=states, actions=actions, gamma=0.8, minvisits=5)
   return execute_game(env, Qtable, episodes, target=25.0, penalty=0.0)


def play_frozen_lake_v0(episodes=1000):
   env = gym.make('FrozenLake-v0')

   states = range(16)
   actions = range(4)
   Qtable = rl.QTable(states=states, actions=actions, gamma=0.9, minvisits=5)
   return execute_game(env, Qtable, episodes, target=0.78, penalty=10.)


def play_frozen_lake_8x8_v0(episodes=1000):
   env = gym.make('FrozenLake8x8-v0')

   states = range(64)
   actions = range(4)
   Qtable = rl.QTable(states=states, actions=actions, gamma=1.0, minvisits=10)
   return execute_game(env, Qtable, episodes, target=0.5, penalty=10., verbose=True)


def execute_game(env, Qtable, episodes, target, penalty, verbose=False):
   performance = deque(maxlen=100)
   performance.append(0.)
   episode = 0
   while episode < episodes and np.mean(performance) < target:
       episode += 1
       state = env.reset()

       steps, rewards, done = 0, [], False
       while not done:
           steps += 1
           action = Qtable.predict(state)
           next_state, reward, done, _ = env.step(action)
           if done and reward <= 0:
               Qtable.fit(state, action, reward - penalty, next_state)
           else:
               Qtable.fit(state, action, reward, next_state)
           rewards.append(reward)
           state = next_state
       performance.append(np.sum(rewards))
       if verbose:
           print(
               "episode {} steps {} sum {} overall mean {}".format(
                   episode, steps, np.sum(rewards), np.mean(performance)
               )
           )
   return episode
