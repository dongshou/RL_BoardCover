"""
作者：冬兽
时间：2020年11月13日
使用gym完成残缺棋盘覆盖的强化学习方案,创建相关的环境
可能存在的问题：
（1）可观测状态只有当前点的上下左右8个点，太少
"""
import time

import math
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
        self.action_space = spaces.Box(low=0,high=4.999,shape=(1,),dtype=np.float64)  # 动作空间,离散0：不动作，。。。。
        self.observation_space = spaces.Box(low=0,high=1,shape=(self.size*self.size,),dtype=np.float64) # 状态空间
        self.viewer = rendering.Viewer(self.edege_size+30,self.edege_size+30)

    def init_state(self):
        state = np.zeros(shape=[self.size,self.size])
        x,y = random.randint(0,self.size-1),random.randint(0,self.size-1)
        state[x][y] = 1
        return state

    def _center(self,start,side):
        """
        开始的坐标，以及边长side
        :param start:
        :param side:
        :return: 中心2*2的坐标
        """
        return (start[0]+int(side/2)-1,start[1]+int(side/2)-1)

    def reset(self):
        """
        重置环境的状态，返回观察
        :return:
        """
        self.state = self.init_state()
        self._center_list()  #获取一个位置列表[location1,location2,...]
        self.done = False
        self.reward =0
        self.index =0 #在self.loclist中的位置
        return np.reshape(self.state,(-1,))

    def _center_list(self):
        side = self.size
        self.loclist = []
        while side>=2:
            i = 0
            while i<self.size:
                j = 0
                while j<self.size:
                    self.loclist.append(self._center((i,j),side))
                    j+=side
                i+=side
            side = int(side/2)


    def step(self,action):
        """
        推进一个时间步长，返回observation，reward，done，info
        :param action:
        :return:
        """
        reward =0
        action =int(action[0])
        (x,y) = self.loclist[self.index]
        if action ==1 and self.state[x+1][y] ==0 and self.state[x][y+1] ==0 and self.state[x+1][y+1]==0:
            self.state[x+1][y]=1
            self.state[x][y+1] =1
            self.state[x+1][y+1] =1
        elif action ==2 and self.state[x][y] ==0 and self.state[x+1][y] ==0 and self.state[x+1][y+1] ==0:
            self.state[x][y]=1
            self.state[x+1][y]=1
            self.state[x+1][y+1] =1
        elif action ==3 and self.state[x][y] ==0 and self.state[x][y+1]==0 and self.state[x+1][y+1]==0:
            self.state[x][y]=1
            self.state[x][y+1] =1
            self.state[x+1][y+1] =1
        elif action ==4 and self.state[x][y]==0 and self.state[x][y+1]==0 and self.state[x+1][y]==0:
            self.state[x][y]=1
            self.state[x+1][y]=1
            self.state[x][y+1]=1

        _r = self._reward()
        reward = _r-self.reward
        self.reward = _r
        reward += -0.01
        self.index +=1
        if self.index>=len(self.loclist):
            self.done = True
        return np.reshape(self.state,(-1,)),reward,self.done,{}


    def _reward(self):
        """
        通过self.state与E(2)的点乘实现卷积操作，从而实现reward的计算，stride=[1,2,2,1],padding = VALID
        :return:
        """
        _re =0
        for (x,y) in self.loclist:
            _re += int(np.sum(self.state[x:x+2,y:y+2])/4)
        return _re


    def render(self,mode="human",close=False):
        """
        重新绘制环境的一帧
        :param mode:
        :param close:
        :return:
        """
        w = self.edege_size/self.size
        # 画格子
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

        # 画已经覆盖的格子
        for i in range(self.size):
            for j in range(self.size):
                if self.state[i][j] == 1:
                    self.viewer.draw_polygon([(15+j*w+1,15+self.edege_size-i*w-2),
                                              (15+(j+1)*w-2,15+self.edege_size-i*w-2),
                                              (15+(j+1)*w-2,15+self.edege_size-(i+1)*w+1),
                                              (15 + j* w+1, 15 +self.edege_size-(i+1) * w+1)])
        # 画正在覆盖的格子
        if self.index ==len(self.loclist):
            (i,j) = self.loclist[self.index-1]
        else:
            (i,j) = self.loclist[self.index]
        self.viewer.draw_polygon([(15 + j * w , 15 + self.edege_size - i * w),
                                  (15 + (j + 2) * w, 15 + self.edege_size - i * w),
                                  (15 + (j + 2) * w, 15 + self.edege_size - (i + 2) * w),
                                  (15 + j * w, 15 + self.edege_size - (i + 2) * w)],
                                 filled=False,
                                 color = (1,0,0),
                                 linewidth = 5)
        return self.viewer.render(return_rgb_array=mode == 'human')

    def close(self):
        if self.viewer:
            self.viewer.close()

if __name__ == '__main__':
    env = MyEnv(4)
    env.reset()
    env.render()
    print(env.loclist)
    while True:
        act = input("action:")
        obs,reward,dones,info = env.step(act)
        # print(reward,dones)
        env.render()
        if dones:
            env.reset()
            time.sleep(1)
            env.render()

