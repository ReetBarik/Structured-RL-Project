import time
import os
import numpy as np
import warnings
import matplotlib.pyplot as plt

import gym
import gym_chrome_dino

from stable_baselines import A2C
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.common import set_global_seeds
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy

best_mean_reward, n_steps = -np.inf, 0

def callback(_locals, _globals):
    """
    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
    :param _locals: (dict)
    :param _globals: (dict)
    """
    global n_steps, best_mean_reward
    # Print stats every 1000 calls
    if (n_steps + 1) % 1000 == 0:
        # Evaluate policy performance
        x, y = ts2xy(load_results(log_dir), 'timesteps')
        if len(x) > 0:
            mean_reward = np.mean(y[-100:])
            print(x[-1], 'timesteps')
            print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))

            # New best model, you could save the agent here
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                # Example for saving best model
                print("Saving new best model")
                _locals['self'].save(log_dir + 'best_model.pkl')
    n_steps += 1
    # Returning False will stop training early
    return True


def make_env(env_id, rank, seed=0):
	"""
	Utility function for multiprocessed env.

	:param env_id: (str) the environment ID
	:param num_env: (int) the number of environment you wish to have in subprocesses
	:param seed: (int) the inital seed for RNG
	:param rank: (int) index of the subprocess
	"""
	def _init():
		env = gym.make(env_id)
		env.seed(seed + rank)
		env = Monitor(env, log_dir, allow_early_resets=True)
		env = DummyVecEnv([lambda: env])
		return env

	set_global_seeds(seed)
	return _init

def evaluate(model, num_steps=1000):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_steps: (int) number of timesteps to evaluate it
    :return: (float) Mean reward
    """
    episode_rewards = [[0.0] for _ in range(env.num_envs)]
    obs = env.reset()
    for i in range(num_steps):
      # _states are only useful when using LSTM policies
      actions, _states = model.predict(obs)
      # here, action, rewards and dones are arrays
      # because we are using vectorized env
      obs, rewards, dones, info = env.step(actions)
      
      # Stats
      for i in range(env.num_envs):
          episode_rewards[i][-1] += rewards[i]
          if dones[i]:
              episode_rewards[i].append(0.0)

    mean_rewards =  [0.0 for _ in range(env.num_envs)]
    n_episodes = 0
    for i in range(env.num_envs):
        mean_rewards[i] = np.mean(episode_rewards[i])     
        n_episodes += len(episode_rewards[i])   

    # Compute mean reward
    mean_reward = round(np.mean(mean_rewards), 1)
    print("Mean reward:", mean_reward, "Num episodes:", n_episodes)

    return mean_reward


def movingAverage(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def plot_results(log_folder, title='Learning Curve'):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    y = movingAverage(y, window=50)
    # Truncate x
    x = x[len(x) - len(y):]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title + " Smoothed")
    plt.show()



with warnings.catch_warnings():

	warnings.simplefilter("ignore")

	log_dir = "a2c/"
	os.makedirs(log_dir, exist_ok=True)

	# Create and wrap the environment
	env = gym.make("ChromeDino-v0")
	# Logs will be saved in log_dir/monitor.csv
	env = Monitor(env, log_dir, allow_early_resets=True)
	env = DummyVecEnv([lambda: env])
	
	# env_id = "ChromeDino-v0"
	# num_cpu = 4  # Number of processes to use
	# # Create the vectorized environment
	# env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])

	model = A2C(MlpPolicy, env, ent_coef=0.1, verbose=1, tensorboard_log="tensorboard/")

	# Train the agent
	model.learn(total_timesteps=5000000, callback=callback)

	# Plot results
	plot_results(log_dir)

	# Save the agent
	model.save("a2c_dino")
	del model  # delete trained model to demonstrate loading

	model = A2C.load("a2c_dino")
	mean_reward = evaluate(model, num_steps=10000)