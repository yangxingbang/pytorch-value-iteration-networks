import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter


class VIN(nn.Module):
    def __init__(self, config):
        super(VIN, self).__init__()
        self.config = config
        # l_i: Number of channels in input layer. Default: 2, i.e. obstacles image and goal image.
        # l_h: Number of channels in first convolutional layer. Default: 150, described in paper.
        # l_q: Number of channels in q layer (~actions) in VI-module. Default: 10, described in paper.
        self.h = nn.Conv2d(
            in_channels=config.l_i,
            out_channels=config.l_h,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            bias=True)
        self.r = nn.Conv2d(
            in_channels=config.l_h,
            out_channels=1,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            bias=False)
        # VI Transitions: 3 × 3 kernel
        self.q = nn.Conv2d(
            in_channels=1,
            out_channels=config.l_q,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            bias=False)
        # Reactive policy: FC, softmax
        self.fc = nn.Linear(in_features=config.l_q, out_features=8, bias=False)
        self.w = Parameter(
            torch.zeros(config.l_q, 1, 3, 3), requires_grad=True)
        self.sm = nn.Softmax(dim=1)

    def forward(self, input_view, state_x, state_y, k):
        """
        :param input_view: (batch_sz, imsize, imsize)
        :param state_x: (batch_sz,), 0 <= state_x < imsize
        :param state_y: (batch_sz,), 0 <= state_y < imsize
        :param k: number of iterations
        :return: logits and softmaxed logits
        """
        # 障碍物地图和目的地点地图经过两次卷积得到了奖赏R
        h = self.h(input_view)  # Intermediate output
        r = self.r(h)           # Reward
        # 奖赏经过一次卷积得到了Q的初始值
        q_init = self.q(r)           # Initial Q value from reward
        # 给Q取最大值得到值函数V
        v, _ = torch.max(q_init, dim=1, keepdim=True)

        # eval_q实际上实现了R,V经过P的RNN
        # TODO(yxb,20220729)： replace it with RNN
        def eval_q(r, v):
            # 给 奖赏 和 值函数 的拼接tensor 做 卷积
            return F.conv2d(
                # Stack reward with most recent value
                # 把r和v沿第1维的方向拼接
                torch.cat([r, v], 1),
                # Convolve r->q weights to r, and v->q weights for v. These represent transition probabilities
                torch.cat([self.q.weight, self.w], 1),
                stride=1,
                padding=1)

        # Update q and v values
        # 这个循环相当于执行了k-1次的RNN
        for i in range(k - 1):
            # 更新Q，更新V，更新的次数由自己设定
            q = eval_q(r, v)
            v, _ = torch.max(q, dim=1, keepdim=True)

        # 这是第K次的RNN
        q = eval_q(r, v)
        # q: (batch_sz, l_q, map_size, map_size)
        batch_sz, l_q, _, _ = q.size()
        # Attention: choose Q values for current state ？
        q_out = q[torch.arange(batch_sz), :, state_x.long(), state_y.long()].view(batch_sz, l_q)

        # Reactive policy: FC, softmax
        # logits一般表示未归一化以前的变量，一般表示即将喂给softmax的向量
        logits = self.fc(q_out)  # q_out to actions
        # 值迭代的收敛 和 最优策略 在哪儿体现？
        return logits, self.sm(logits)
