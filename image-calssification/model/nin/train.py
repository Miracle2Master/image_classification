import torch
from torch import nn
from d2l import torch as d2l

from model.resnet.train import train
from script.model_tester import ModelTester
from script.model_trainer import ModelTrainer
from utils.data_loader import get_std_data


def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU())


net = nn.Sequential(
    nin_block(1, 96, kernel_size=11, strides=4, padding=0),
    nn.MaxPool2d(3, stride=2),
    nin_block(96, 256, kernel_size=5, strides=1, padding=2),
    nn.MaxPool2d(3, stride=2),
    nin_block(256, 384, kernel_size=3, strides=1, padding=1),
    nn.MaxPool2d(3, stride=2),
    nn.Dropout(0.5),
    # 标签类别数是10
    nin_block(384, 10, kernel_size=3, strides=1, padding=1),
    nn.AdaptiveAvgPool2d((1, 1)),
    # 将四维的输出转成二维的输出，其形状为(批量大小,10)
    nn.Flatten())
if __name__ == '__main__':

    trainer = ModelTrainer(net)

    lr, num_epochs, batch_size = 1e-4, 10, 256
    train_iter, test_iter = get_std_data(size=224)
    trainer.fast_train(train_iter,test_iter,"nin")

    tester = ModelTester(net)
    tester.get_ConfusionMatrix("nin")
