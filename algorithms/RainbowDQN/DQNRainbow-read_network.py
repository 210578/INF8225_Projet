import pickle
import matplotlib.pyplot as plt
import numpy as np

# Load the experiment data from the Rainbow DQN pickle file
with open("./results/DQN_Double_Dueling_PER_N_steps_env_CartPole-v1_number_1_seed_42.pkl", "rb") as f:
    log_data = pickle.load(f)

# Extract the training rewards
reward_log = log_data["training_rewards"]

# Compute moving average with a window of 50
window_size = 50
moving_avg = np.convolve(reward_log, np.ones(window_size)/window_size, mode='valid')

# Plot the reward log and its moving average
plt.plot(reward_log, label="Episode Rewards")
plt.plot(np.arange(window_size - 1, len(reward_log)), moving_avg, label="Moving Average (window=50)")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.legend()
plt.grid()
plt.title("Rainbow DQN on CartPole-v1")

# Save and show the plot
plt.savefig('./Results/RainbowDQN_CartPole-v1.png')  # Save before show
plt.show()

# Print final metrics
print(f"Final evaluation score = {log_data['final_evaluation_score']}")
print(f"Training time (seconds) = {log_data['training_time_sec']}")
print(f"Total training steps = {log_data['total_steps']}")
print(f"Total episodes = {log_data['total_episodes']}")
