import time

from gym.wrappers import Monitor
import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv


from myenv import MyEnv
import matplotlib.pyplot as plt

def train(size):
    log = 'env/'
    env1 = Monitor(MyEnv(size), log, force=True)
    env = DummyVecEnv([lambda: env1])
    # the noise objects for DDPG
    model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log='./log')
    model.learn(total_timesteps=int(1e7))
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
        env.render()
        time.sleep(0.5)
        if dones:
            print(rewards,dones)
            # time.sleep(3)



if __name__ == '__main__':
    train(4)
    test(4)