import numpy as np
import random

# Simple action space class to mimic gym's Discrete
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

        self.static_walls = {(3, 3), (3, 4), (4, 3)}  # Permanent walls
        self.dynamic_walls = set()
        self.checkpoints = [[5, 2], [2, 6]]  # Optional checkpoints
        self.reached_checkpoints = set()
        self.total_steps = 0

        self.reset()

    def reset(self):
        self.agent_pos = [0, 0]
        self.reached_checkpoints = set()
        self.dynamic_walls = set()
        self.total_steps = 0

        while True:
            self.goal = [np.random.randint(0, self.size), np.random.randint(0, self.size)]
            if (
                self.goal != self.agent_pos
                and tuple(self.goal) not in self.walls
            ):
                break

        return self._get_state()

    def _get_state(self):
        return tuple(self.agent_pos)

    def update_dynamic_walls(self):
        # Randomly generate 2-3 new dynamic walls
        new_walls = set()
        while len(new_walls) < 3:
            x, y = random.randint(0, self.size - 1), random.randint(0, self.size - 1)
            pos = (x, y)
            if (
                pos != tuple(self.agent_pos)
                and pos != tuple(self.goal)
                and pos not in self.static_walls
                and pos not in [tuple(cp) for cp in self.checkpoints]
            ):
                new_walls.add(pos)
        self.dynamic_walls = new_walls

    def step(self, action):
        self.total_steps += 1

        # Move agent
        x, y = self.agent_pos
        if action == 0 and y > 0: y -= 1  # up
        elif action == 1 and y < self.size - 1: y += 1  # down
        elif action == 2 and x > 0: x -= 1  # left
        elif action == 3 and x < self.size - 1: x += 1  # right

        if (x, y) not in self.static_walls and (x, y) not in self.dynamic_walls:
            self.agent_pos = [x, y]

        # Update dynamic walls every 5 steps
        if self.total_steps % 5 == 0:
            self.update_dynamic_walls()

        # Reward system
        reward = -1
        done = False

        if self.agent_pos in self.checkpoints and tuple(self.agent_pos) not in self.reached_checkpoints:
            self.reached_checkpoints.add(tuple(self.agent_pos))
            reward += 30

        if self.agent_pos == self.goal and len(self.reached_checkpoints) == len(self.checkpoints):
            reward = 100
            done = True

        return self._get_state(), reward, done, {}

    @property
    def walls(self):
        return self.static_walls.union(self.dynamic_walls)
