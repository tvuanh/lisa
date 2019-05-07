import numpy as np


class Translator(object):

    def __init__(self, values):
        self.encoders, self.decoders = dict(), dict()
        for v, c in zip(values, range(len(values))):
            self.encoders[v] = c
            self.decoders[c] = v

    def encode(self, value):
        if value in self.encoders.keys():
            return self.encoders[value]
        else:
            raise ValueError("Requested value {} is not found".format(value))

    def decode(self, encoded_value):
        if encoded_value in self.decoders.keys():
            return self.decoders[encoded_value]
        else:
            raise ValueError(
                "Requested value {} to be decoded is not found".format(
                    encoded_value)
            )


class QTable(object):

    def __init__(self, states, actions, gamma=1., minvisits=10):
        self.states_translator = Translator(states)
        self.actions_translator = Translator(actions)
        # discount factor
        self.gamma = gamma
        # min number of visits per state-action
        self.minvisits = minvisits
        # Q(state, action) sample mean value
        self.Qmean = np.zeros(
            (
                len(self.states_translator.encoders),
                len(self.actions_translator.encoders)
            )
        ) # state-rows, action-columns
        # number of visits per state-action
        self.visits = np.zeros(self.Qmean.shape)
        # sum of squared future rewards
        self.Sr2 = np.zeros(self.Qmean.shape)
        # Q(state, action) sample variance
        self.Qvar = np.ones(self.Qmean.shape) * np.inf

    def predict(self, state):
        encoded_state = self.states_translator.encode(state)
        encoded_actions = list(self.actions_translator.encoders.values())
        visits = self.visits[encoded_state, :]
        if np.min(visits) < self.minvisits:
            encoded_action = np.random.choice(encoded_actions)
        else:
            means = self.Qmean[encoded_state, :]
            variances = self.Qvar[encoded_state, :]
            sigmas = np.sqrt(np.divide(variances, visits))
            draws = means + sigmas * np.random.randn(len(encoded_actions))
            encoded_action = self.action_from_values(draws)
        return self.actions_translator.decode(encoded_action)

    def fit(self, state, action, reward, next_state):
        encoded_state = self.states_translator.encode(state)
        encoded_action = self.actions_translator.encode(action)
        encoded_next_state = self.states_translator.encode(next_state)
        future_reward = reward + self.gamma * np.max(
            self.Qmean[encoded_next_state, :])
        # update visits, mean, sum squared rewards and variance in exact order
        # update visits
        self.visits[encoded_state, encoded_action] += 1
        visits = self.visits[encoded_state, encoded_action]
        # update mean
        Qm = self.Qmean[encoded_state, encoded_action]
        self.Qmean[encoded_state, encoded_action] += (future_reward - Qm) / visits
        # update sum squared rewards
        self.Sr2[encoded_state, encoded_action] += future_reward * future_reward
        # update the variance
        if visits > 1:
            sr2 = self.Sr2[encoded_state, encoded_action]
            Qm = self.Qmean[encoded_state, encoded_action]
            self.Qvar[encoded_state, encoded_action] = min(
                (sr2 - visits * Qm * Qm) / (visits - 1), np.inf
            )

    def action_from_values(self, values):
        maxQstate = np.max(values)
        encoded_actions = self.actions_translator.encoders.values()
        possible_actions = [a for a in encoded_actions if values[a] >= maxQstate]
        return np.random.choice(possible_actions)
