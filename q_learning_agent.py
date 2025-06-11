import numpy as np
import random

class QLearningAgent:
    def __init__(self, state_size, action_size, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.1):
        self.q_table = {}
        self.action_size = action_size
        self.alpha = alpha      # learning rate
        self.gamma = gamma      # discount factor
        self.epsilon = epsilon  # exploration rate
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

    def get_qs(self, state):
        if state not in self.q_table:
            self.q_table[state] = [0] * self.action_size
        return self.q_table[state]

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        return np.argmax(self.get_qs(state))

    def learn(self, state, action, reward, next_state, done):
        old_q = self.get_qs(state)[action]
        next_max_q = max(self.get_qs(next_state))
        new_q = old_q + self.alpha * (reward + self.gamma * next_max_q * (1 - int(done)) - old_q)
        self.q_table[state][action] = new_q

        if done:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
