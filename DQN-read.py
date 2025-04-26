import pickle
import matplotlib.pyplot as plt
import numpy as np

# Load the experiment data from the pickle file
with open("LunarLander-v3_dqn_log.pkl", "rb") as f:
    log_data = pickle.load(f)

# Extract the reward log
reward_log = log_data["rewards"]

# Compute moving average with a window of 50
window_size = 50
moving_avg = np.convolve(reward_log, np.ones(window_size)/window_size, mode='valid')

# Plot the reward log and its moving average
plt.plot(reward_log, label="Rewards")
plt.plot(np.arange(window_size - 1, len(reward_log)), moving_avg, label="Moving average (window=50)")
plt.xlabel("Episode")
plt.ylabel("Reward")

plt.legend()
plt.grid()
plt.title("DQN on LunarLander-v3")

# Save and show the plot
plt.savefig('./Results/DQN_LunarLander-v3.png')  # Save before show
plt.show()

# Print final metrics
print(f"Final score = {log_data['final_score']}")
print(f"Training time = {log_data['duration']}")  # Fixed key
