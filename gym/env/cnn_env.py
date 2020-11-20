"""
作者：冬兽
时间：2020年11月13日
使用gym完成残缺棋盘覆盖的强化学习方案,创建相关的环境
可能存在的问题：
（1）可观测状态只有当前点的上下左右8个点，太少
"""
import numpy as np
import random
import gym
from gym import spaces,core
from gym.envs.classic_control import rendering


class MyEnv(gym.Env):
    def __init__(self,size):
        super(MyEnv,self).__init__()
        self.size = size  #棋盘大小
        self.edege_size = 512
        self.loc =[0,0]
        self.target = np.ones(shape=[self.size,self.size])
        # self.state = self.init_state()
        self.action_space = spaces.Box(low=np.array([0,0,1]),high=np.array([self.size-0.1,self.size-0.1,4]),dtype=np.float64)  # 动作空间,离散0：不动作，。。。。
        self.observation_space = spaces.Box(low=-1,high=1,shape=(self.size,self.size,1),dtype=np.float64) # 状态空间
        print(self.observation_space.shape)
        self.viewer = rendering.Viewer(self.edege_size+30,self.edege_size+30)

    def init_state(self):
        state = np.zeros(shape=(self.size,self.size,1))
        x,y = random.randint(0,self.size-1),random.randint(0,self.size-1)
        state[x][y][0] = 1
        return state

    def reset(self):
        """
        重置环境的状态，返回观察
        :return:
        """
        self.state = self.init_state()
        self.loc=[0,0]
        self.done = False
        obs = self.get_observation()
        return obs

    def get_observation(self,x=0,y=0):
        """
        返回全部棋盘
        :param x:
        :param y:
        :return:
        """
        return self.state


    def step(self,action):
        """
        推进一个时间步长，返回observation，reward，done，info
        :param action:
        :return:
        """
        reward = 0
        x = int(action[0])
        y = int(action[1])
        self.loc = [x,y]
        act = int(action[2])
        reward = self.update(x,y,act)
        reward += -0.3
        if  (self.state==self.target).all():
            reward = 100
        # print("(x,y):",self.loc,"action:",act)
        return self.state,reward,self.done,{}


    def update(self,x,y,action):
        """
        更新状态并返回奖励
        :param x:
        :param y:
        :param action:
        :return:
        """
        reward =0
        if action ==1:
            if x-1>=0 and y-1>=0:
               if self.state[x][y][0]  ==0 and self.state[x-1][y][0] ==0 and self.state[x][y-1][0] ==0:
                   self.state[x][y][0]  = 1
                   self.state[x-1][y][0] = 1
                   self.state[x][y-1][0]  = 1
                   reward = 1
               else:
                   reward = 0.01
            else:
               reward = -1
        elif action ==2:
            if x-1>=0 and y+1<self.size:
                if  self.state[x][y][0]  ==0 and  self.state[x-1][y][0]  ==0 and  self.state[x][y+1][0]  ==0:
                    self.state[x][y][0]  =1
                    self.state[x-1][y][0]  =1
                    self.state[x][y+1][0]  =1
                    reward =1
                else:
                    reward = 0.01
            else:
                reward =-1
        elif action==3:
            if y-1>=0 and x+1<self.size:
                if  self.state[x][y][0]  ==0 and  self.state[x][y-1][0]  ==0 and  self.state[x+1][y][0]  ==0:
                    self.state[x][y][0]  =1
                    self.state[x][y-1][0]  =1
                    self.state[x+1][y][0]  =1
                    reward = 1
                else:
                    reward = 0.01
            else:
                reward = -1
        elif action ==4:
            if x+1<self.size and y+1<self.size:
                if  self.state[x][y][0]  ==0 and  self.state[x+1][y][0]  ==0 and  self.state[x][y+1][0]  ==0:
                    self.state[x][y][0]  =1
                    self.state[x][y+1][0]  =1
                    self.state[x+1][y][0]  =1
                    reward =1
                else:
                    reward = 0.01
            else:
                reward = -1

        return reward


    def render(self,mode="human",close=False):
        """
        重新绘制环境的一帧
        :param mode:
        :param close:
        :return:
        """
        counter =0
        w = self.edege_size/self.size
        for i in range(self.size+1):
            if i%2==0:
                c = (1,0,0)
            else:
                c = (0,0,0)
            self.viewer.draw_line((15,15+i*w),(15+self.edege_size,15+i*w),color = c)
        for j in range(self.size+1):
            if j%2==0:
                c = (1,0,0)
            else:
                c = (0,0,0)
            self.viewer.draw_line((15+j*w,15),(15+j*w,15+self.edege_size),color = c)
        # self.viewer.draw_polygon([(15+w,15+w),(15+2*w,15+w),(15+2*w,15+2*w),(15+w,15+2*w)])
        for i in range(self.size):
            for j in range(self.size):
                if self.state[i][j][0]==1:
                    counter +=1
                    self.viewer.draw_polygon([(15+j*w+1,15+self.edege_size-i*w-2),
                                              (15+(j+1)*w-2,15+self.edege_size-i*w-2),
                                              (15+(j+1)*w-2,15+self.edege_size-(i+1)*w+1),
                                              (15 + j* w+1, 15 +self.edege_size-(i+1) * w+1)])
        i,j = self.loc[0],self.loc[1]
        self.viewer.draw_polygon([(15 + j * w , 15 + self.edege_size - i * w),
                                  (15 + (j + 1) * w, 15 + self.edege_size - i * w),
                                  (15 + (j + 1) * w, 15 + self.edege_size - (i + 1) * w),
                                  (15 + j * w, 15 + self.edege_size - (i + 1) * w)],
                                 filled=False,
                                 color = (0,0,0.5),
                                 linewidth = 3)
        return self.viewer.render(return_rgb_array=mode == 'human')

    def close(self):
        if self.viewer:
            self.viewer.close()

if __name__ == '__main__':
    env = MyEnv(8)
    env.render()