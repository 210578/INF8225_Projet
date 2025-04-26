import numpy as np
import torch
import pickle
import os
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
import time
from replay_buffer import *
from rainbow_dqn import DQN
from types import SimpleNamespace
import argparse


class Runner:
    def __init__(self, args, env_name, number, seed):
        self.args = args
        self.env_name = env_name
        self.number = number
        self.seed = seed       
        self.env = gym.make(self.env_name)
        self.env_evaluate = gym.make(self.env_name)  # Keep this for final evaluation
        # Reset the environments with the seed
        self.env.reset(seed=self.seed)
        self.env_evaluate.reset(seed=self.seed)
        self.env.action_space.seed(self.seed)
        self.env_evaluate.action_space.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        self.args.state_dim = self.env.observation_space.shape[0]
        self.args.action_dim = self.env.action_space.n
        self.args.episode_limit = self.env.spec.max_episode_steps  # Maximum number of steps per episode
        print(f"env={self.env_name}")
        print(f"state_dim={self.args.state_dim}")
        print(f"action_dim={self.args.action_dim}")
        print(f"episode_limit={self.args.episode_limit}")

        # Setup ReplayBuffer based on the args
        if args.use_per and args.use_n_steps:
            self.replay_buffer = N_Steps_Prioritized_ReplayBuffer(args)
        elif args.use_per:
            self.replay_buffer = Prioritized_ReplayBuffer(args)
        elif args.use_n_steps:
            self.replay_buffer = N_Steps_ReplayBuffer(args)
        else:
            self.replay_buffer = ReplayBuffer(args)
        
        self.agent = DQN(args)

        # Setup the algorithm string based on the arguments
        self.algorithm = 'DQN'
        if args.use_double and args.use_dueling and args.use_noisy and args.use_per and args.use_n_steps:
            self.algorithm = 'Rainbow_' + self.algorithm
        else:
            if args.use_double:
                self.algorithm += '_Double'
            if args.use_dueling:
                self.algorithm += '_Dueling'
            if args.use_noisy:
                self.algorithm += '_Noisy'
            if args.use_per:
                self.algorithm += '_PER'
            if args.use_n_steps:
                self.algorithm += "_N_steps"

        # Setup TensorBoard logging
        self.writer = SummaryWriter(log_dir=f'/export/tmp/sala/INF8225/runs/DQN/{self.algorithm}_env_{self.env_name}_number_{self.number}_seed_{self.seed}')

        # Changed to track training rewards instead of evaluation rewards
        self.training_rewards = []  # Record the rewards during training
        self.training_episode_counter = 0  # Count training episodes
        self.total_steps = 0  # Record the total steps during the training

        if args.use_noisy:
            self.epsilon = 0
        else:
            self.epsilon = self.args.epsilon_init
            self.epsilon_min = self.args.epsilon_min
            self.epsilon_decay = (self.args.epsilon_init - self.args.epsilon_min) / self.args.epsilon_decay_steps

    def evaluate_policy(self):
        # Keep this for final evaluation only
        self.agent.net.eval()  # Set the model in evaluation mode
        total_reward = 0
        for _ in range(self.args.evaluate_times):
            state, _ = self.env_evaluate.reset()
            done = False
            episode_reward = 0
            while not done:
                action = self.agent.choose_action(state, epsilon=0)  # No exploration during evaluation
                next_state, reward, terminated, truncated, _ = self.env_evaluate.step(action)
                done = terminated or truncated
                episode_reward += reward
                state = next_state
            total_reward += episode_reward
        average_reward = total_reward / self.args.evaluate_times
        print(f"Final Evaluation, Average Reward: {average_reward}")
        return average_reward

    def run(self):
        start_time = time.time()
        
        # Remove initial evaluation
        
        while self.total_steps < self.args.max_train_steps:
            state, _ = self.env.reset()
            done = False
            episode_steps = 0
            episode_reward = 0  # Track reward for this episode
            
            while not done:
                action = self.agent.choose_action(state, epsilon=self.epsilon)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                episode_steps += 1
                self.total_steps += 1
                episode_reward += reward  # Accumulate the reward

                # Decrease epsilon for exploration/exploitation
                if not self.args.use_noisy:
                    self.epsilon = self.epsilon - self.epsilon_decay if self.epsilon - self.epsilon_decay > self.epsilon_min else self.epsilon_min

                # Check termination condition
                if done and episode_steps != self.args.episode_limit:
                    if self.env_name == 'LunarLander-v2' and reward <= -100:
                        reward = -1  # Adjust reward for LunarLander
                    terminal = True
                else:
                    terminal = False

                # Store the transition in the replay buffer
                self.replay_buffer.store_transition(state, action, reward, next_state, terminal, done)
                state = next_state

                # Learn from stored experience
                if self.replay_buffer.current_size >= self.args.batch_size:
                    self.agent.learn(self.replay_buffer, self.total_steps)
            
            # Record episode reward after episode completion
            self.training_rewards.append(episode_reward)
            self.training_episode_counter += 1
            
            # Log to tensorboard
            self.writer.add_scalar('Training/Episode_Reward', episode_reward, self.training_episode_counter)
            self.writer.add_scalar('Training/Epsilon', self.epsilon, self.training_episode_counter)
            
            # Print progress periodically
            if self.training_episode_counter % 10 == 0:
                print(f"Episode {self.training_episode_counter}, Steps: {self.total_steps}, Reward: {episode_reward}, Epsilon: {self.epsilon:.3f}")

        # Perform final evaluation
        final_evaluation_reward = self.evaluate_policy()

        # Calculate total training time
        training_time = time.time() - start_time

        # Save logs after training
        os.makedirs('./results', exist_ok=True)
        log_path = f'./results/{self.algorithm}_env_{self.env_name}_number_{self.number}_seed_{self.seed}.pkl'
        with open(log_path, "wb") as f:
            pickle.dump({
                "training_rewards": self.training_rewards,  # Changed to save training rewards
                "final_evaluation_score": final_evaluation_reward,  # Final evaluation score
                "total_steps": self.total_steps,
                "total_episodes": self.training_episode_counter,
                "training_time_sec": training_time
            }, f)

        # Plot training rewards
        plt.figure()
        plt.plot(self.training_rewards, label="Training reward")
        if len(self.training_rewards) >= 10:
            avg_rewards = np.convolve(self.training_rewards, np.ones(10) / 10, mode='valid')
            plt.plot(range(9, len(avg_rewards) + 9), avg_rewards, label="Avg(10)")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title(f"{self.env_name} - {self.algorithm} - Training Progress")
        plt.legend()
        plt.grid()
        plt.savefig(f"./results/{self.algorithm}_{self.env_name}_training_progress.png")
        plt.show()

        # Save the model
        model_path = f"./results/{self.algorithm}_{self.env_name}_model.pth"
        torch.save(self.agent.net.state_dict(), model_path)


# Run training for multiple environments
if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameter Setting for DQN")
    parser.add_argument("--env_name", type=str, default="LunarLander-v3", help="Name of the Gym environment")

    parser.add_argument("--max_train_steps", type=int, default=int(4e5), help=" Maximum number of training steps")
    parser.add_argument("--epsilon_decay_steps", type=int, default=int(1e5), 
                    help="Number of steps over which to decay epsilon from init to min")
    parser.add_argument("--evaluate_freq", type=int, default=2000)

    parser.add_argument("--evaluate_times", type=int, default=1, help="Number of episodes for final evaluation")

    parser.add_argument("--episodes", type=int, default=5000, help="Number of training episodes")
    parser.add_argument("--buffer_capacity", type=int, default=int(1e4), help="Replay buffer capacity")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Number of neurons in hidden layer")

    parser.add_argument("--epsilon_init", type=float, default=1.0, help="Initial epsilon for exploration")
    parser.add_argument("--epsilon_min", type=float, default=0.1, help="Minimum value of epsilon")
    parser.add_argument("--epsilon_deca", type=float, default=1.0 / 3000, help="Epsilon decay rate per episode")

    parser.add_argument("--tau", type=float, default=0.005, help="soft update the target network")
    parser.add_argument("--use_soft_update", type=bool, default=True, help="Whether to use soft update")
    parser.add_argument("--target_update_freq", type=int, default=200, help="Update frequency of the target network(hard update)")
    parser.add_argument("--n_steps", type=int, default=5, help="n_steps")
    parser.add_argument("--alpha", type=float, default=0.6, help="PER parameter")
    parser.add_argument("--beta_init", type=float, default=0.4, help="Important sampling parameter in PER")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Learning rate Decay")
    parser.add_argument("--grad_clip", type=float, default=10.0, help="Gradient clip")

    parser.add_argument("--use_double", type=bool, default=True, help="Whether to use double Q-learning")
    parser.add_argument("--use_dueling", type=bool, default=True, help="Whether to use dueling network")
    parser.add_argument("--use_noisy", type=bool, default=False, help="Whether to use noisy network")
    parser.add_argument("--use_per", type=bool, default=True, help="Whether to use PER")
    parser.add_argument("--use_n_steps", type=bool, default=True, help="Whether to use n_steps Q-learning")

    args = parser.parse_args()

    env_list = ["LunarLander-v3"]

    for i, env_name in enumerate(env_list):
        runner = Runner(args=args, env_name=env_name, number=1, seed=42)
        runner.run()