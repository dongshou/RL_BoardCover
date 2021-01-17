"""
作者：东寿
时间：2020年11月1日
文件描述：强化学习当中为环境交互相关的代码
"""
import numpy as np
import random
import math


class Agent():
    def __init__(self,size):
        self.size = size
        self.state = np.zeros(shape = [1,1,self.size,self.size])
        self.x,self.y = random.randint(0,self.size-1),random.randint(0,self.size-1)
        self.state[0][0][self.x][self.y] = 1

    def game_over(self):
        flag = True
        fail = False
        for i in range(self.size):
            for j in range(self.size):
                if self.state[0][0][i][j] != 1:
                    flag =False
                if self.state[0][0][i][j] == 2:
                    fail = True
        if flag :
            return "Sucess"
        else:
            if fail == True:
                return "Defeat"
        return "playing"

    def check_state(self,x,y):
        """
        检测点（x,y）上下左右四个点的状态
        :param x:                      ------
        :param y:                     | |1| |
        :param action:                -------
        :return:                     |2|0|3|
                                     -------
                                     | |4| |
        """
        def check_point(x,y):
            if x >= 0 and x<self.size and y>=0 and y<self.size :
                s = self.state[0][0][x - 1][y]
            else:
                s = -1
            return s

        state =[0]*5
        state[0] = self.state[0][0][x][y]
        state[1] = check_point(x-1,y)
        state[2] = check_point(x,y-1)
        state[3] = check_point(x,y+1)
        state[4] = check_point(x+1,y)
        return state


    def update_state(self,x,y,action):
        """
        输入参数x,y,以及在此位置上的action，更新状态S，返回next_s,reward,terminal
        :param x:
        :param y:
        :param action: 0:不采取任何动作，1代表缺失序号为1的空。。。。
            -----
            |1|2|
            -----
            |3|4|
            -----
        :return: self.state,reward
        """
        reward = 0
        # if action ==0:
        #     reward = 2
        # elif action ==1:
        #     if x-1>=0 and y-1>=0:
        #         if self.check_state(self.state[0][0][x][y],self.state[0][0][x-1][y],self.state[0][0][x][y-1]):
        #             reward = 2
        #         else:
        #             reward = -10
        #         self.state[0][0][x][y] += 1
        #         self.state[0][0][x - 1][y] +=1
        #         self.state[0][0][x][y - 1] += 1
        #     else:
        #         return self.state, -100, "defeat"
        # elif action == 2:
        #     if y+1<self.size-1 and x-1>=0:
        #         if self.check_state(self.state[0][0][x][y],self.state[0][0][x - 1][y],self.state[0][0][x][y+1]):
        #             reward = 2
        #         else:
        #             reward = -10
        #         self.state[0][0][x][y] += 1
        #         self.state[0][0][x - 1][y] +=1
        #         self.state[0][0][x][y +1] += 1
        #     else:
        #         return self.state, -100, "defeat"
        # elif action == 3:
        #     if x+1<self.size-1 and y-1>=0:
        #         if self.check_state(self.state[0][0][x][y],self.state[0][0][x +1][y],self.state[0][0][x][y-1]):
        #             reward = 2
        #         else:
        #             reward = -10
        #         self.state[0][0][x][y] += 1
        #         self.state[0][0][x +1][y] +=1
        #         self.state[0][0][x][y -1] += 1
        #     else:
        #         return self.state,-100,"defeat"
        # elif action == 4:
        #     if x+1<self.size-1 and y+1<self.size-1:
        #         if self.check_state(self.state[0][0][x][y],self.state[0][0][x + 1][y],self.state[0][0][x][y+1]):
        #             reward = 2
        #         else:
        #             reward= -10
        #         self.state[0][0][x][y] += 1
        #         self.state[0][0][x + 1][y] +=1
        #         self.state[0][0][x][y +1] += 1
        #     else:
        #         return  self.state,-100,"defeat"
        state = self.check_state(x,y)
        flag = False
        if action ==0:
            reward = 3
        if action ==1:
            if state[0] ==0 and state[1] ==0 and state[2] ==0:
                self.state[0][0][x][y],self.state[0][0][x-1][y],self.state[0][0][x][y-1] = 1,1,1
                reward =5
            else:
                # self.state[0][0][x][y] += 1
                # if state[1] != -1:
                #     self.state[0][0][x-1][y] +=1
                # if state[2] !=-1:
                #     self.state[0][0][x][y-1] +=1
                reward =-5
                flag = True
        if action ==2:
            if state[0] ==0 and state[1] ==0 and state[3] ==0:
                self.state[0][0][x][y],self.state[0][0][x-1][y],self.state[0][0][x][y+1] = 1,1,1
                reward =5
            else:
                # self.state[0][0][x][y] += 1
                # if state[1] != -1:
                #     self.state[0][0][x-1][y] +=1
                # if state[3] !=-1:
                #     self.state[0][0][x][y+1] +=1
                reward =-5
                flag = True
        if action == 3:
            if state[0] == 0 and state[2] == 0 and state[4] == 0:
                self.state[0][0][x][y], self.state[0][0][x +1][y], self.state[0][0][x][y -1] = 1, 1, 1
                reward = 5
            else:
                # self.state[0][0][x][y] += 1
                # if state[4] != -1:
                #     self.state[0][0][x +1][y] += 1
                # if state[2] != -1:
                #     self.state[0][0][x][y - 1] += 1
                reward = -5
                flag = True
        if action == 4:
            if state[0] == 0 and state[4] == 0 and state[3] == 0:
                self.state[0][0][x][y], self.state[0][0][x + 1][y], self.state[0][0][x][y + 1] = 1, 1, 1
                reward = 5
            else:
                # self.state[0][0][x][y] += 1
                # if state[4] != -1:
                #     self.state[0][0][x + 1][y] += 1
                # if state[3] != -1:
                #     self.state[0][0][x][y + 1] += 1
                reward = -5
                flag =True


        result = self.game_over()
        if flag:
            result = "Defeat"
        if result == "Sucess":
            reward +=100
        return self.state,reward,result


if __name__ == '__main__':
    s = Agent(16)
    state,reward,result = s.update_state(0,1,1)
    print(result)
