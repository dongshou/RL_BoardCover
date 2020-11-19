from gym.wrappers import Monitor
from stable_baselines.common.noise import OrnsteinUhlenbeckActionNoise

from stable_baselines import DQN, DDPG,PPO2
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.ddpg.policies import MlpPolicy,CnnPolicy
import numpy as np
import time
from myenv import MyEnv
import matplotlib.pyplot as plt

def train(size):
    log = 'env/'
    env1 = Monitor(MyEnv(size), log, force=True)
    env = DummyVecEnv([lambda: env1])
    # the noise objects for DDPG
    n_actions = env.action_space.shape[-1]
    param_noise = None
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))
    model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log='./log')
    model.learn(total_timesteps=1e7)
    model.save("ddpg_mountain_{}".format(size))
    env.close()
    del model
    del env

def test(size):
    log = 'env/'
    env2 = Monitor(MyEnv(size), log, force=True)
    env = DummyVecEnv([lambda: env2])
    print('test')
    model = PPO2.load("ddpg_mountain_{}".format(size))
    obs = env.reset()
    env.render()
    while True:
        action, _states = model.predict(obs)
        # action = input("action:")
        obs, rewards, dones, info = env.step(action)
        print(int(action[0]),rewards,dones)
        env.render()
        time.sleep(0.5)

if __name__ == '__main__':
    train(8)
    test(8)