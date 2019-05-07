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
        states = ("a", "b", "c")
        actions = ("left", "right")
        self.table = rl.QTable(states, actions, gamma=1.0, minvisits=10)
        self.shape = (len(states), len(actions))

    def assertTable(self, table, expected_Qmean, expected_visits, expected_Sr2,
                    expected_Qvar):
        np.testing.assert_array_equal(table.Qmean, expected_Qmean)
        np.testing.assert_array_equal(table.visits, expected_visits)
        np.testing.assert_array_equal(table.Sr2, expected_Sr2)
        np.testing.assert_array_equal(table.Qvar, expected_Qvar)

    def test_QTable_init(self):
        expected_Qmean = np.zeros(self.shape)
        expected_visits = np.zeros(self.shape)
        expected_Sr2 = np.zeros(self.shape)
        expected_Qvar = np.ones(self.shape) * np.inf
        self.assertTable(
            self.table, expected_Qmean, expected_visits, expected_Sr2,
            expected_Qvar)

    def test_QTable_fit(self):
        #
        self.table.fit(state="a", action="left", reward=10.0, next_state="b")
        expected_Qmean = np.array(
            [
                [10., 0.],
                [0., 0.],
                [0., 0.],
            ]
        )
        expected_visits = np.array(
            [
                [1, 0],
                [0, 0],
                [0, 0],
            ]
        )
        expected_Sr2 = np.array(
            [
                [100., 0.],
                [0., 0.],
                [0., 0.],
            ]
        )
        expected_Qvar = np.ones(self.shape) * np.inf
        self.assertTable(
            self.table, expected_Qmean, expected_visits, expected_Sr2,
            expected_Qvar)

        #
        self.table.fit(state="b", action="left", reward=10.0, next_state="c")
        expected_Qmean = np.array(
            [
                [10., 0.],
                [10., 0.],
                [0., 0.],
            ]
        )
        expected_visits = np.array(
            [
                [1, 0],
                [1, 0],
                [0, 0],
            ]
        )
        expected_Sr2 = np.array(
            [
                [100., 0.],
                [100., 0.],
                [0., 0.],
            ]
        )
        expected_Qvar = np.ones(self.shape) * np.inf
        self.assertTable(
            self.table, expected_Qmean, expected_visits, expected_Sr2,
            expected_Qvar)

        #
        self.table.fit(state="c", action="left", reward=10.0, next_state="a")
        expected_Qmean = np.array(
            [
                [10., 0.],
                [10., 0.],
                [20., 0.],
            ]
        )
        expected_visits = np.array(
            [
                [1, 0],
                [1, 0],
                [1, 0],
            ]
        )
        expected_Sr2 = np.array(
            [
                [100., 0.],
                [100., 0.],
                [400., 0.],
            ]
        )
        expected_Qvar = np.ones(self.shape) * np.inf
        self.assertTable(
            self.table, expected_Qmean, expected_visits, expected_Sr2,
            expected_Qvar)

        #
        self.table.fit(state="a", action="right", reward=-5.0, next_state="b")
        expected_Qmean = np.array(
            [
                [10., 5.],
                [10., 0.],
                [20., 0.],
            ]
        )
        expected_visits = np.array(
            [
                [1, 1],
                [1, 0],
                [1, 0],
            ]
        )
        expected_Sr2 = np.array(
            [
                [100., 25.],
                [100., 0.],
                [400., 0.],
            ]
        )
        expected_Qvar = np.ones(self.shape) * np.inf
        self.assertTable(
            self.table, expected_Qmean, expected_visits, expected_Sr2,
            expected_Qvar)

        #
        self.table.fit(state="b", action="right", reward=-5.0, next_state="c")
        expected_Qmean = np.array(
            [
                [10., 5.],
                [10., 15.],
                [20., 0.],
            ]
        )
        expected_visits = np.array(
            [
                [1, 1],
                [1, 1],
                [1, 0],
            ]
        )
        expected_Sr2 = np.array(
            [
                [100., 25.],
                [100., 225.],
                [400., 0.],
            ]
        )
        expected_Qvar = np.ones(self.shape) * np.inf
        self.assertTable(
            self.table, expected_Qmean, expected_visits, expected_Sr2,
            expected_Qvar)

        #
        self.table.fit(state="c", action="right", reward=-5.0, next_state="a")
        expected_Qmean = np.array(
            [
                [10., 5.],
                [10., 15.],
                [20., 5.],
            ]
        )
        expected_visits = np.array(
            [
                [1, 1],
                [1, 1],
                [1, 1],
            ]
        )
        expected_Sr2 = np.array(
            [
                [100., 25.],
                [100., 225.],
                [400., 25.],
            ]
        )
        expected_Qvar = np.ones(self.shape) * np.inf
        self.assertTable(
            self.table, expected_Qmean, expected_visits, expected_Sr2,
            expected_Qvar)

        #
        self.table.fit(state="a", action="left", reward=10.0, next_state="b")
        expected_Qmean = np.array(
            [
                [17.5, 5.],
                [10., 15.],
                [20., 5.],
            ]
        )
        expected_visits = np.array(
            [
                [2, 1],
                [1, 1],
                [1, 1],
            ]
        )
        expected_Sr2 = np.array(
            [
                [725., 25.],
                [100., 225.],
                [400., 25.],
            ]
        )
        expected_Qvar = np.array(
            [
                [112.5, np.inf],
                [np.inf, np.inf],
                [np.inf, np.inf],
            ]
        )
        self.assertTable(
            self.table, expected_Qmean, expected_visits, expected_Sr2,
            expected_Qvar)


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
               "episode {} steps {} rewards {} total {}".format(
                   episode, steps, rewards, np.sum(rewards
                   )
               )
           )
   return episode
