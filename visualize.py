import pygame
import pickle
from env import GridWorldEnv

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GRAY = (100, 100, 100)

CELL_SIZE = 60

def draw_env(screen, env, agent_pos):
    screen.fill(WHITE)
    size = env.grid_size

    for x in range(size):
        for y in range(size):
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, BLACK, rect, 1)

            if (x, y) == tuple(env.goal):
                pygame.draw.rect(screen, GREEN, rect)
            elif (x, y) in env.walls:
                pygame.draw.rect(screen, GRAY, rect)
            elif (x, y) == tuple(agent_pos):
                pygame.draw.rect(screen, BLUE, rect)

def load_q_table(path="q_table.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)

def run_simulation():
    pygame.init()
    env = GridWorldEnv(size=10)
    screen_size = env.grid_size * CELL_SIZE
    screen = pygame.display.set_mode((screen_size, screen_size))
    pygame.display.set_caption("RL Grid Car - Trained")

    q_table = load_q_table()
    obs = env.reset()
    done = False

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        state = tuple(obs)
        action = max(range(4), key=lambda a: q_table.get(state, [0]*4)[a])  # best action
        obs, reward, done, _ = env.step(action)

        draw_env(screen, env, env.agent_pos)
        pygame.display.flip()
        pygame.time.wait(200)

        if done:
            pygame.time.wait(1000)
            obs = env.reset()
            done = False

if __name__ == "__main__":
    run_simulation()
