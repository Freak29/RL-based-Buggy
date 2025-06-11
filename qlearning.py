import numpy as np
import random

class QLearningAgent:
    def __init__(self, state_size, action_size, lr=0.1, gamma=0.9, epsilon=1.0, decay=0.995):
        self.q_table = {}
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = 0.01
        self.decay = decay
        self.action_size = action_size

    def get_qs(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_size)
        return self.q_table[state]

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        return np.argmax(self.get_qs(state))

    def learn(self, state, action, reward, next_state, done):
        q_values = self.get_qs(state)
        future_q = 0 if done else np.max(self.get_qs(next_state))
        q_values[action] += self.lr * (reward + self.gamma * future_q - q_values[action])
        if done:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)
