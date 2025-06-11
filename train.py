from env import GridWorldEnv
from q_learning_agent import QLearningAgent
import pickle

EPISODES = 1000  # you can increase this for better learning

def train():
    env = GridWorldEnv(size=10)
    agent = QLearningAgent(state_size=(10, 10), action_size=4)

    for episode in range(EPISODES):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode+1}, Total reward: {total_reward}, Epsilon: {agent.epsilon:.3f}")

    # Save Q-table to file
    with open("q_table.pkl", "wb") as f:
        pickle.dump(agent.q_table, f)

    print("✅ Training complete and Q-table saved to 'q_table.pkl'")

if __name__ == "__main__":
    train()
