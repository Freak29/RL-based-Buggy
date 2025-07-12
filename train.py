from env import GridWorldEnv
from q_learning_agent import QLearningAgent
import pickle

EPISODES = 20000          # Increased for better learning in dynamic setup
MAX_STEPS = 150          # Limit steps per episode

def train():
    env = GridWorldEnv(size=10)
    agent = QLearningAgent(state_size=(10, 10), action_size=4)

    for episode in range(EPISODES):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0

        visited = {}

        while not done and steps < MAX_STEPS:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)

            # Penalize for repeating states to avoid loops
            visited[next_state] = visited.get(next_state, 0) + 1
            repeat_penalty = -0.6 * (visited[next_state] - 1)
            reward += repeat_penalty

            agent.learn(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            steps += 1

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode+1}, Total reward: {total_reward}, Epsilon: {agent.epsilon:.3f}")

    # Save Q-table to file
    with open("q_table.pkl", "wb") as f:
        pickle.dump(agent.q_table, f)

    print("✅ Training complete and Q-table saved to 'q_table.pkl'")

if __name__ == "__main__":
    train()
