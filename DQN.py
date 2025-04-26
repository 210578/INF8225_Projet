import random
import numpy as np
import pickle
import time
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --------------------- Q-Network ---------------------
class QNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# --------------------- Replay Buffer ---------------------
class ReplayMemory:
    def __init__(self, size):
        self.buffer = []
        self.max_size = size

    def add(self, transition):
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)
        return (
            torch.tensor(np.array(states), dtype=torch.float32, device=device),
            torch.tensor(actions, dtype=torch.int64, device=device),
            torch.tensor(rewards, dtype=torch.float32, device=device),
            torch.tensor(np.array(next_states), dtype=torch.float32, device=device),
            torch.tensor(dones, dtype=torch.float32, device=device),
        )

    def __len__(self):
        return len(self.buffer)

# --------------------- DQN Core Functions ---------------------
def select_action(model, state, epsilon, action_space):
    if random.random() > epsilon:
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            return model(state_tensor).argmax(dim=1).item()
    else:
        return action_space.sample()

def optimize(model, target_model, memory, optimizer, gamma, batch_size):
    if len(memory) < batch_size:
        return

    states, actions, rewards, next_states, dones = memory.sample(batch_size)

    with torch.no_grad():
        next_q_values = target_model(next_states)
        max_next_q = next_q_values.max(dim=1)[0]
        targets = rewards + gamma * max_next_q * (1 - dones)

    q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    loss = F.mse_loss(q_values, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def evaluate(model, env, episodes=10):
    scores = []
    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        total = 0
        while not done:
            action = select_action(model, state, epsilon=0.0, action_space=env.action_space)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total += reward
        scores.append(total)
    return np.mean(scores)

# --------------------- Training Loop ---------------------
def train_dqn(env_name="CartPole-v1", episodes=15000, buffer_size=10000, batch_size=64,
              gamma=0.99, lr=5e-4, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=1.0 / 3000):

    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    q_model = QNet(state_dim, action_dim).to(device)
    target_model = QNet(state_dim, action_dim).to(device)
    target_model.load_state_dict(q_model.state_dict())

    memory = ReplayMemory(buffer_size)
    optimizer = torch.optim.Adam(q_model.parameters(), lr=lr)

    epsilon = epsilon_start
    rewards = []

    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = select_action(q_model, state, epsilon, env.action_space)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            memory.add((state, action, reward, next_state, float(done)))
            state = next_state
            total_reward += reward

            if len(memory) >= 2000:
                optimize(q_model, target_model, memory, optimizer, gamma, batch_size)

        epsilon = max(epsilon_end, epsilon - epsilon_decay)
        rewards.append(total_reward)

        if ep % 50 == 0:
            target_model.load_state_dict(q_model.state_dict())

        print(f"[{env_name}] Ep {ep+1}/{episodes} | Reward: {total_reward:.1f} | Eps: {epsilon:.5f}", end="\r")

    env.close()
    return q_model, rewards

# --------------------- Run and Save ---------------------
def run_for_env(env_name):
    start_time = time.time()
    model, reward_log = train_dqn(env_name=env_name)
    duration = time.time() - start_time

    final_score = evaluate(model, gym.make(env_name), episodes=50)
    print(f"\n{env_name} - Final Avg Score (50 runs): {final_score:.2f}")
    print(f"{env_name} - Training Duration: {duration:.2f} sec")

    # Save
    os.makedirs('./results', exist_ok=True)
    torch.save(model.state_dict(), f"{env_name}_dqn_model.pth")
    with open(f"{env_name}_dqn_log.pkl", "wb") as f:
        pickle.dump({
            "rewards": reward_log,
            "final_score": final_score,
            "duration": duration
        }, f)

    # Plot
    plt.figure()
    plt.plot(reward_log, label="Reward")
    plt.plot(np.convolve(reward_log, np.ones(10)/10, mode="valid"), label="Avg(10)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(f"{env_name} - Training Progress")
    plt.legend()
    plt.grid()
    plt.savefig(f"./results/{env_name}_training_progress.png")
    plt.show()

# Train and evaluate on both environments
for env in ["LunarLander-v3"]:
    run_for_env(env)
