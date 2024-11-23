import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l


class Residual(nn.Module):  #@save
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        F.dropout(Y, p=0.2, training=self.training)
        return F.relu(Y)


def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    '''
    :param input_channels: 输入通道数
    :param num_channels: 输出通道数
    :param num_residuals: 包含几个残差层
    :param first_block: 是否为第一个块
    :return:
    '''
    blk = []
    for i in range(num_residuals):
        # 是当前块中的第一个但不是第一个块
        # 除了第一个大块的一个小块不会进，其他块中的第一个小块都进
        if i == 0 and not first_block:
            # 进来后，通道数变为num_channels，同时，特征矩阵长和宽减半
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            # 第二个块，通道数，特征矩阵的shape都不变
            blk.append(Residual(num_channels, num_channels))
    return blk
