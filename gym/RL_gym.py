from gym.wrappers import Monitor
from stable_baselines.common.vec_env import DummyVecEnv

from stable_baselines import DQN, DDPG
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.ddpg.policies import MlpPolicy

#
import time
from myenv import MyEnv

log ='env/'
env1 =Monitor(MyEnv(8),log,force=True)
env = DummyVecEnv([lambda :env1])
# env = gym.make('CartPole-v1')
#
# # the noise objects for DDPG
# # n_actions = env.action_space.shape[-1]
# # param_noise = None
# # action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))
#
# model = DDPG(MlpPolicy, env, verbose=1, param_noise=param_noise, action_noise=action_noise)
model = DDPG(MlpPolicy, env, verbose=1,tensorboard_log ='./log/')
model.learn(total_timesteps=10000)
model.save("ddpg_mountain")
# del model # remove to demonstrate saving and loading
# #
# model = DDPG.load("ddpg_mountain")
#
print('test')
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    print(int(action[0]))
    obs, rewards, dones, info = env.step(action)
    env.render()
    time.sleep(0.5)
