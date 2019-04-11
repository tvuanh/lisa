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


class TestIntegration(unittest.TestCase):

    def test_integration(self):
        episodes = 3000
        nplays = 10
        results = np.array([play_copy_v0(episodes) for _ in range(nplays)])
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

   performance = deque(maxlen=100)
   performance.append(0.)

   episode = 0
   while episode < episodes and np.mean(performance) < 25.:
       episode += 1
       state = env.reset()

       steps, rewards, done = 0, [], False
       while not done:
           steps += 1
           action = Qtable.predict(state)
           next_state, reward, done, _ = env.step(action)
            # use shifted reward to update the Q table
           Qtable.fit(state, action, reward + 0.5, next_state)
           rewards.append(reward)
           state = next_state
       performance.append(np.sum(rewards))
       # print(
       #     "episode {} steps {} rewards {} total {}".format(episode, steps, rewards, np.sum(rewards))
       # )

   return episode
