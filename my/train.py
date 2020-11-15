"""
time:2020年11月8日
作者：冬兽
这是一个使用ddpg和agent，使游戏和算法能够交互的类
"""

from my.RL_ddpg import DDPG
from my.agent import Agent
import torch

class Game():
    def __init__(self,size):
        self.size =size
        self.init_state(size)
        self.ddpg = DDPG(tau = 1.0)

    def init_state(self,size):
        self.agent = Agent(size)


    def playgame(self):
        game_state = "playing"
        self.init_state(self.size)
        i,j=0,0
        while i<self.size:
            while j<self.size:
                s = torch.from_numpy(self.agent.state).float()
                action = self.ddpg.predit(s)
                next_s,reward,game_state = self.agent.update_state(i,j,torch.argmax(action).item())
                next_s = torch.from_numpy(next_s).float()
                terminal = 0 if game_state == "playing" else 1
                actor_cost,critic_cost = self.ddpg.learn(s,action,reward,next_s,terminal)
                print("(i,j):(",i,",",j,") action:",torch.argmax(action).item(),"reward:",reward,"  actor_cost:",
                      actor_cost.item(),"critic_cost:",critic_cost.item(),"Game state:",game_state)
                if game_state !="Defeat" and torch.argmax(action).item() !=0:
                    j+=1
                elif game_state !='playing':
                    break
            if game_state != "Defeat" and torch.argmax(action).item() !=0:
                i += 1
            elif game_state != 'playing':
                break

if __name__ == '__main__':
    g = Game(8)
    epochs = 0
    while True:
        g.playgame()
        epochs+=1
        # if epochs %100==0:
        #     g.ddpg.save()
        #     g.ddpg.update_target()



