import time
from gym.wrappers import Monitor

from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import DQN
from stable_baselines.common.vec_env import DummyVecEnv


from myenv import MyEnv
import matplotlib.pyplot as plt

def train(size):
    log = 'env/'
    env1 = Monitor(MyEnv(size), log, force=True)
    env = DummyVecEnv([lambda: env1])
    # the noise objects for DDPG
    model = DQN(MlpPolicy, env, verbose=1, tensorboard_log='./log')
    model.learn(total_timesteps=int(1e7))
    model.save("dqn_mountain_{}".format(size))
    env.close()
    del model
    del env

def test(size):
    log = 'env/'
    env2 = Monitor(MyEnv(size), log, force=True)
    env = DummyVecEnv([lambda: env2])
    print('test')
    model = DQN.load("dqn_mountain_{}".format(8))
    obs = env.reset()
    env.render()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()



if __name__ == '__main__':
    # train(8)
    test(128)