import unittest
from collections import deque

import numpy as np

import gym

from lisa import rl


class TestTranslator(unittest.TestCase):

    def setUp(self):
        self.translator = rl.Translator(values=("a", "c", "b"))

    def test_encode_valid(self):
        self.assertEqual(self.translator.encode("a"), 0)
        self.assertEqual(self.translator.encode("c"), 1)
        self.assertEqual(self.translator.encode("b"), 2)

    def test_encode_invalid(self):
        with self.assertRaises(ValueError):
            self.translator.encode("d")

    def test_decode_valid(self):
        self.assertEqual(self.translator.decode(0), "a")
        self.assertEqual(self.translator.decode(1), "c")
        self.assertEqual(self.translator.decode(2), "b")

    def test_decode_invalid(self):
        with self.assertRaises(ValueError):
            self.translator.decode(3)


class TestQTable(unittest.TestCase):

    def setUp(self):
        states = (0, 1, 2)
        actions = ("left", "right")
        self.table = rl.QTable(states, actions, gamma=1.0, minvisits=10)
        self.shape = (len(states), len(actions))

    def test_QTable_init(self):
        expected_Qmean = np.zeros(self.shape)
        np.testing.assert_array_equal(self.table.Qmean, expected_Qmean)

        expected_visits = np.ones(self.shape)
        np.testing.assert_array_equal(self.table.visits, expected_visits)

        expected_Sr2 = np.zeros(self.shape)
        np.testing.assert_array_equal(self.table.Sr2, expected_Sr2)

        expected_Qvar = np.ones(self.shape) * np.inf
        np.testing.assert_array_equal(self.table.Qvar, expected_Qvar)


class TestIntegration(unittest.TestCase):

    def test_integration_copy_v0(self):
        episodes = 3000
        nplays = 10
        results = np.array([play_copy_v0(episodes) for _ in range(nplays)])
        success = results < episodes
        self.assertTrue(np.sum(success) > 0.7 * nplays)
        self.assertTrue(np.mean(results[success]) < 1300)

    def test_integration_frozen_lake_v0(self):
        episodes = 5000
        nplays = 1
        results = np.array([play_frozen_lake_v0(episodes) for _ in range(nplays)])
        success = results < episodes
        self.assertTrue(np.sum(success) > 0.7 * nplays)
        self.assertTrue(np.mean(results[success]) < 1300)


def play_copy_v0(episodes=3000):
   env = gym.make('Copy-v0')

   states = range(6)
   actions = tuple(
       [(i, j, k) for i in (0, 1) for j in (0, 1) for k in range(5)]
   )
   Qtable = rl.QTable(states=states, actions=actions, gamma=0.8)
   return execute_game(env, Qtable, episodes, target=25.0)


def play_frozen_lake_v0(episodes=1000):
   env = gym.make('FrozenLake-v0')

   states = range(16)
   actions = range(4)
   Qtable = rl.QTable(states=states, actions=actions, gamma=0.8, minvisits=10)
   return execute_game(env, Qtable, episodes, target=0.78)


def execute_game(env, Qtable, episodes, target):
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
           Qtable.fit(state, action, reward, next_state)
           rewards.append(reward)
           state = next_state
       performance.append(np.sum(rewards))
       print(
           "episode {} steps {} rewards {} total {}".format(
               episode, steps, rewards, np.sum(rewards
               )
           )
       )
   return episode
