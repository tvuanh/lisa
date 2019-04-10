import numpy as np

from utils import encode_action, decode_action


class QTable(object):

    actions_space = range(20)

    def __init__(self, gamma=1., minvisits=10):
        # discount factor
        self.gamma = gamma
        # min number of visits per state-action
        self.minvisits = minvisits
        # Q(state, action) sample mean value
        self.Qmean = np.zeros(
            (6, len(self.actions_space))
            ) # 6 state-rows, 20 action-columns
        # number of visits per state-action
        self.visits = np.ones(self.Qmean.shape)
        # sum of squared future rewards
        self.Sr2 = np.zeros(self.Qmean.shape)
        # Q(state, action) sample variance
        self.Qvar = np.ones(self.Qmean.shape) * np.inf

    def predict(self, state):
        visits = self.visits[state, :]
        if np.min(visits) < self.minvisits:
            action = np.random.choice(self.actions_space)
        else:
            means, variances = self.Qmean[state, :], self.Qvar[state, :]
            sigmas = np.sqrt(np.divide(variances, visits))
            draws = means + sigmas * np.random.randn(len(self.actions_space))
            action = self.action_from_values(draws)
        return decode_action(action)

    def action_from_values(self, values):
        maxQstate = np.max(values)
        possible_actions = [a for a in self.actions_space if values[a] >= maxQstate]
        return np.random.choice(possible_actions)

    def fit(self, state, action, reward, next_state):
        future_reward = reward + self.gamma * np.max(self.Qmean[next_state, :])
        encoded_action = encode_action(action)
        # update mean, sum squared rewards and variance in exact order
        self.update_mean(state, encoded_action, future_reward)
        self.update_sum_squared_rewards(state, encoded_action, future_reward)
        self.update_variance(state, encoded_action)
        self.visits[state, encoded_action] += 1

    def update_mean(self, state, action, future_reward):
        visits = self.visits[state, action]
        Qm = self.Qmean[state, action]
        self.Qmean[state, action] += (future_reward - Qm) / visits

    def update_sum_squared_rewards(self, state, action, future_reward):
        self.Sr2[state, action] += future_reward * future_reward

    def update_variance(self, state, action):
        visits = self.visits[state, action]
        if visits > 1:
            sr2 = self.Sr2[state, action]
            Qm = self.Qmean[state, action]
            self.Qvar[state, action] = min(
                (sr2 - visits * Qm * Qm) / (visits - 1),
                np.inf
            )
