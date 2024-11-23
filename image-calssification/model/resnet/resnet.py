from torch import nn
import torch
from model.resnet.res_block import resnet_block
from script.model_trainer import ModelTrainer
from utils.data_loader import get_std_data

b3 = nn.Sequential(*resnet_block(64, 128, 2))


# 训练Fashion-mnist的resnet
def get_model():
    b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                       nn.BatchNorm2d(64), nn.ReLU(),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
    b3 = nn.Sequential(*resnet_block(64, 128, 2))
    b4 = nn.Sequential(*resnet_block(128, 256, 2))
    b5 = nn.Sequential(*resnet_block(256, 512, 2))
    net = nn.Sequential(b1, b2, b3, b4, b5,
                        # 使输出的特征矩变为1 * 1
                        nn.AdaptiveAvgPool2d((1, 1)),
                        # 拉成一维向量，添加全连接层得到十个结果
                        nn.Flatten(), nn.Linear(512, 10))
    return net


def get_model2():
    b1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                       nn.BatchNorm2d(64), nn.ReLU(),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
    b3 = nn.Sequential(*resnet_block(64, 128, 2))
    b4 = nn.Sequential(*resnet_block(128, 256, 2))
    b5 = nn.Sequential(*resnet_block(256, 512, 2))
    net = nn.Sequential(b1, b2, b3, b4, b5,
                        # 使输出的特征矩变为1 * 1
                        nn.AdaptiveAvgPool2d((1, 1)),
                        # 拉成一维向量，添加全连接层得到十个结果
                        nn.Flatten(), nn.Linear(512, 10))
    return net


if __name__ == '__main__':
    train_iter,test_iter = get_std_data(size=96)
    trainer = ModelTrainer(get_model(),num_epochs=10)
    trainer.fast_train(train_iter,test_iter,"resnet")

