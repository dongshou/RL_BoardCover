"""
日期：2020年11月5日
作者：冬兽
手写ddpg网络
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import os

class Policy(nn.Module):
    def __init__(self,act_dim):
        super(Policy, self).__init__()
        # 1 input image channel, 8 output channels, 3*3 square convolution
        self.conv1 = nn.Conv2d(1,8,2)
        self.conv2 = nn.Conv2d(8,16,2)
        self.fc1 = nn.Linear(64,300)
        self.fc2 = nn.Linear(300,100)
        self.fc3 = nn.Linear(100,act_dim)

    def forward(self,x):
        v1 = F.max_pool2d(self.conv1(x),2,stride =1)
        v2 = F.max_pool2d(self.conv2(v1),2)
        a3 = v2.view(-1,self.num_flat_features(v2))
        # fc1 = torch.sigmoid(self.fc1(a3))
        # fc2 = torch.sigmoid(self.fc2(fc1))
        fc1 = self.fc1(a3)
        fc2 = self.fc2(fc1)
        result = torch.softmax(self.fc3(fc2),dim= 1)
        return result

    def num_flat_features(self,x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class Value(nn.Module):
    def __init__(self):
        super(Value, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        self.conv1 = nn.Conv2d(1, 8, 2)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.fc1 = nn.Linear(64+100, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 20)
        self.fc4 = nn.Linear(20,1)
        self.afc1 = nn.Linear(5,30)
        self.afc2 = nn.Linear(30,100)

    def forward(self, x,action):
        #action
        act1 = self.afc1(action)
        act2 = self.afc2(act1)
        #envirnment
        v1 = F.max_pool2d(self.conv1(x), 2,stride = 1)
        v2 = F.max_pool2d(self.conv2(v1), 2)
        a3 = v2.view(-1, self.num_flat_features(v2))
        all = torch.cat((a3,act2),dim =1)
        fc1 = self.fc1(all)
        fc2 = self.fc2(fc1)
        fc3 = self.fc3(fc2)
        result = self.fc4(fc3)
        return result

    def num_flat_features(self,x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class Model(nn.Module):
    def __init__(self,act_dim):
        super(Model,self).__init__()
        self.actor = Policy(act_dim)
        self.critic = Value()

    def policy(self,s):
        return self.actor.forward(s)

    def value(self,s,act):
        return self.critic.forward(s,act)

    def get_actor_param(self):
        return self.actor.parameters()


class DDPG():
    def __init__(self,
                model = None,
                gamma =0.99,
                tau = 0.001,
                actor_lr = 1e-3,
                critic_lr = 1e-3):
        """  DDPG algorithm

           Args:
               model (parl.Model): actor and critic 的前向网络.
                                   model 必须实现 get_actor_params() 方法.
               gamma (float): reward的衰减因子.
               tau (float): self.target_model 跟 self.model 同步参数 的 软更新参数
               actor_lr (float): actor 的学习率
               critic_lr (float): critic 的学习率
        """
        assert isinstance(gamma,float)
        assert isinstance(tau, float)
        assert isinstance(actor_lr, float)
        assert isinstance(critic_lr, float)
        self.gamma = gamma
        self.tau = tau
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

        if model == None:
            self.load_model()
        else:
            self.model = model
            self.target_model = copy.deepcopy(model)

        self.actor_optimizer = torch.optim.Adam(params=self.model.get_actor_param(), lr=self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(params=self.model.critic.parameters(), lr=self.critic_lr)


    def predit(self,s):
        return self.model.policy(s)

    def learn(self,s,action,reward,next_s,terminal=None):
        """
        使用ddpg算法来更新acotr 和 critic
        :param s: 当前状态
        :param action: 采取的动作
        :param reward: 获得的奖励
        :param next_s: 下一个状态
        :param terminal:
        :return:actor_cost,critic_cost
        """
        actor_cost = self._actor_learn(s)
        critic_cost = self._critic_learn(s,action,reward,next_s,terminal)

        return actor_cost,critic_cost

    def _actor_learn(self,s):
        self.actor_optimizer.zero_grad()
        cost1 = -torch.mean(self.model.critic(s,self.model.actor(s)))
        cost1.backward()
        self.actor_optimizer.step()
        return cost1

    def _critic_learn(self,s,action,reward,next_s,terminal):
        """

        :param s:
        :param action:
        :param reward:
        :param next_x:
        :param terminal:
        :return:
        """
        next_action = self.target_model.actor(next_s)
        next_Q = self.target_model.critic(next_s,next_action)
        # terminal = float(terminal)
        # target_Q = reward + (1.0-terminal)*self.gamma*self.tau*next_Q
        target_Q = reward + next_Q
        # target_Q.stop_gradient = False
        Q = self.model.critic(s,action)
        self.critic_optimizer.zero_grad()
        cost = F.mse_loss(Q,target_Q,reduction='mean')
        cost.backward()
        self.critic_optimizer.step()
        return cost

    def update_target(self,decay=None,):
        """
         self.target_model从self.model复制参数过来，可设置软更新参数
        :param decay:
        :param share_vars_parallel_executor:
        :return:
        """
        if decay ==None:
            decay = 1-self.tau
        new_param = self.model.state_dict()
        self.target_model.load_state_dict(new_param)
        self.target_model.eval()
        print("target模型参数更新成功")

    def save(self):
        # 保存模型
        if not os.path.exists('./log'):
            os.mkdir('./log')
        torch.save(self.model,'./log/model.pkl')
        torch.save(self.target_model,'./log/target_model.pkl')
        print("模型保存成功")

    def load_model(self):
        print('开始加载model')
        if os.path.exists('./log/model.pkl'):
            self.model = torch.load('./log/model.pkl')
        else:
            print('model.pkl not exist,init by "Model(5)"')
            self.model = Model(5)
        if os.path.exists('./log/target_model.pkl'):
            self.target_model = torch.load('./log/target_model.pkl')
        else:
            print("target_model load failed,init from 'self.model'")
            self.target_model = copy.deepcopy(self.model)
        print('model 加载成功')


if __name__ == '__main__':
    s= torch.randn((1,1,8,8))
    v = Value()
    p =DDPG()
    action = p.predit(s)
    q = v.forward(s,action)

    print(p.predit(s))
