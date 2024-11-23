import torch
from d2l.torch import get_fashion_mnist_labels
from torch import nn
from d2l import torch as d2l

from model.resnet.train import train
from script.model_tester import ModelTester
from script.model_trainer import ModelTrainer
from utils.data_loader import get_std_data


def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels,
                                kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))


def vgg(conv_arch):
    conv_blks = []
    in_channels = 1
    # 卷积层部分
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    return nn.Sequential(
        *conv_blks, nn.Flatten(),
        # 全连接层部分
        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 10))

ratio = 4
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
# print(small_conv_arch)
net = vgg(small_conv_arch)

train_iter, test_iter = get_std_data(size=224)
if __name__ == '__main__':
    lr, num_epochs, batch_size = 0.05, 10, 256
    # train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
    # train(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
    trainer = ModelTrainer(net)
    trainer.fast_train(train_iter,test_iter,"VGG")

    tester = ModelTester(net)
    tester.TestModel(test_iter)
    tester.calculate_f1()
    tester.calculate_recall()
    tester.calculate_precision()
    tester.get_ConfusionMatrix()
    # title=[]
    # for i in range(10):
    #     title.append(get_fashion_mnist_labels(torch.tensor([i]))[0])
    #
    # print(title)
