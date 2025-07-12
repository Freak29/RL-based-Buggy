import pygame
import pickle
from env import GridWorldEnv

# Constants
CELL_SIZE = 60
MAX_STEPS = 150
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Initialize Pygame and Mixer
pygame.init()
pygame.mixer.init()

# Load images and scale
car_img = pygame.image.load("assets/car.png")
car_img = pygame.transform.scale(car_img, (CELL_SIZE - 10, CELL_SIZE - 10))

tree_img = pygame.image.load("assets/tree.png")
tree_img = pygame.transform.scale(tree_img, (CELL_SIZE, CELL_SIZE))

goal_img = pygame.image.load("assets/goal.png")
goal_img = pygame.transform.scale(goal_img, (CELL_SIZE, CELL_SIZE))

# Load sounds
goal_sound = pygame.mixer.Sound("assets/goal.wav")
bump_sound = pygame.mixer.Sound("assets/bump.wav")

def draw_env(screen, env, agent_pos, reward, font):
    screen.fill(WHITE)
    size = env.grid_size

    for x in range(size):
        for y in range(size):
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)

            # Background images
            if (x, y) in env.walls:
                screen.blit(tree_img, rect.topleft)
            elif (x, y) == tuple(env.goal):
                screen.blit(goal_img, rect.topleft)
            else:
                pygame.draw.rect(screen, BLACK, rect, 1)  # grid

    # Draw the agent (car)
    car_rect = car_img.get_rect(center=(agent_pos[0] * CELL_SIZE + CELL_SIZE // 2,
                                        agent_pos[1] * CELL_SIZE + CELL_SIZE // 2))
    screen.blit(car_img, car_rect.topleft)

    # Display reward
    reward_text = font.render(f"Reward: {reward}", True, BLACK)
    screen.blit(reward_text, (10, env.grid_size * CELL_SIZE + 5))

def load_q_table(path="q_table.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)

def run_simulation():
    env = GridWorldEnv(size=10)
    screen_size = env.grid_size * CELL_SIZE
    screen = pygame.display.set_mode((screen_size, screen_size + 40))
    pygame.display.set_caption("RL Buggy Simulation - Trained")
    font = pygame.font.SysFont("Arial", 24)

    q_table = load_q_table()
    obs = env.reset()
    done = False
    total_reward = 0
    steps = 0

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        state = tuple(obs)
        action_values = q_table.get(state, [0] * 4)
        action = max(range(4), key=lambda a: action_values[a])
        prev_pos = obs
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        steps += 1

        # Sounds
        if reward == 100:
            goal_sound.play()
        elif obs == prev_pos:
            bump_sound.play()

        draw_env(screen, env, env.agent_pos, total_reward, font)
        pygame.display.flip()
        pygame.time.wait(200)

        if done or steps >= MAX_STEPS:
            pygame.time.wait(1000)
            obs = env.reset()
            done = False
            total_reward = 0
            steps = 0

if __name__ == "__main__":
    run_simulation()
