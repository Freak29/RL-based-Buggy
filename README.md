# 🤖 RL-based Buggy

A simple Reinforcement Learning (Q-Learning) project where a virtual buggy (agent) learns to navigate a grid world environment with obstacles to reach a goal using trial and error.

---

## 🚀 Demo

After training, the agent learns the optimal path and reaches the destination consistently.

![Reward Plot](reward_plot.png)

🎥 **Demo Videos**

▶️ Before Learning (Exploration Phase)  
[📥 Download before_learning.mp4](media/before_learning.mp4)

▶️ After Learning (Optimal Path)  
[📥 Download after_learning.mp4](media/after_learning.mp4)

---

## 🧠 What’s Inside

This project uses a custom environment built with Python and visualized using `pygame`.  
The agent learns using **Q-learning** and stores its knowledge in a Q-table.

---

## 📁 Project Structure

RL-based-Buggy/

├── env.py # GridWorld environment (walls, goal, agent)

├── q_learning_agent.py # Agent with Q-table and learning logic

├── train.py # Training loop for the agent

├── visualize.py # Pygame simulation after training

├── q_table.pkl # Saved Q-table after training

├── reward_plot.png # Reward vs Episodes graph

├── main.py # (Optional) Launcher or entry script

├── .gitignore # Ignoring cache and binary files

└── README.md # You’re reading this now!



---

## 📦 Requirements

Install the required libraries in your virtual environment:

```bash
pip install pygame matplotlib numpy
Or use a requirements.txt:

pygame
matplotlib
numpy
🏁 How to Run
Clone the Repository

git clone https://github.com/freak29/RL-based-Buggy.git
cd RL-based-Buggy
Train the Agent

python train.py
Visualize the Learned Policy

python visualize.py
📚 Concepts Used
Q-Learning

Grid-based pathfinding

Exploration vs Exploitation (Epsilon-Greedy)

Reward shaping

Pygame visualization

🛠️ Future Improvements
Add dynamic obstacles

Implement Deep Q-Learning (DQN)

Port it to a real-world robot using Raspberry Pi or Arduino

🧑‍💻 Author
Vaibhav Shikhar Singh
📧 reachvaibhav29@gmail.com
🔗 GitHub: freak29

📄 License
This project is licensed under the MIT License.
---

### ✅ How to Use:
1. Open `README.md` locally or on GitHub.
2. Paste the full content above.
3. Commit & push it.

Let me know if you want me to generate a matching `requirements.txt`, `LICENSE`, or badges for GitHub (l
