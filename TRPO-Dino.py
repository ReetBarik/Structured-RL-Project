import gym
import gym_chrome_dino

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import TRPO

env = gym.make("ChromeDino-v0")
env = DummyVecEnv([lambda: env])

model = TRPO(MlpPolicy, env, verbose=1, tensorboard_log="tensorboard/")
model.learn(total_timesteps=10000)

# Enjoy trained agent
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
env.close()