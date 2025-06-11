from env import GridWorldEnv
from qlearning import QLearningAgent
import matplotlib.pyplot as plt

env = GridWorldEnv()
agent = QLearningAgent(state_size=100, action_size=4)

episodes = 500
rewards = []

for ep in range(episodes):
    state = env.reset()
    total_reward = 0

    for _ in range(100):
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        agent.learn(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        if done:
            break

    rewards.append(total_reward)

# Plot
plt.plot(rewards)
plt.title("Reward per Episode")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.savefig("reward_plot.png")
print("Plot saved as reward_plot.png")

