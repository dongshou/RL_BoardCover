from gym.wrappers import Monitor
from stable_baselines.common.noise import OrnsteinUhlenbeckActionNoise

from stable_baselines import DQN, DDPG
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.ddpg.policies import MlpPolicy,CnnPolicy
import numpy as np
import time
from cnn_env import MyEnv
import matplotlib.pyplot as plt

def train(size):
    log = 'env/'
    env1 = Monitor(MyEnv(size), log, force=True)
    env = DummyVecEnv([lambda: env1])
    # the noise objects for DDPG
    n_actions = env.action_space.shape[-1]
    param_noise = None
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))
    #
    model = DDPG(CnnPolicy, env, verbose=1, param_noise=param_noise, action_noise=action_noise)
    #
    model.learn(total_timesteps=10000)
    model.save("ddpg_mountain_{}_cnn".format(size))
    env.close()
    del env
    del model

def test(size):
    log = 'env/'
    env2 = Monitor(MyEnv(size), log, force=True)
    env = DummyVecEnv([lambda: env2])
    print('test')
    model = DDPG.load("ddpg_mountain_{}_cnn".format(size))
    obs = env.reset()
    env.render()
    while True:
        action, _states = model.predict(obs)
        # action = input("action:")
        print("action:",action)
        obs, rewards, dones, info = env.step(action)
        env.render(action)
        time.sleep(0.5)

if __name__ == '__main__':
    train(64)
    test(64)