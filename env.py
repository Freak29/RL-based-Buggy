import numpy as np
import random

class Discrete:
    def __init__(self, n):
        self.n = n

    def sample(self):
        return random.randint(0, self.n - 1)

class GridWorldEnv:
    def __init__(self, size=10):
        self.size = size
        self.grid_size = size  # For visualization
        self.action_space = Discrete(4)  # 0: up, 1: down, 2: left, 3: right
        self.reset()

    def reset(self):
        self.agent_pos = [0, 0]
        self.goal_pos = [self.size - 1, self.size - 1]
        self.goal = self.goal_pos  # For visualization
        self.walls = {(3, 3), (3, 4), (4, 3)}
        return self._get_state()

    def _get_state(self):
        return tuple(self.agent_pos)

    def step(self, action):
        x, y = self.agent_pos
        if action == 0 and y > 0: y -= 1  # up
        elif action == 1 and y < self.size - 1: y += 1  # down
        elif action == 2 and x > 0: x -= 1  # left
        elif action == 3 and x < self.size - 1: x += 1  # right

        if (x, y) not in self.walls:
            self.agent_pos = [x, y]

        reward = -1
        done = False
        if self.agent_pos == self.goal_pos:
            reward = 100
            done = True

        return self._get_state(), reward, done , {}
