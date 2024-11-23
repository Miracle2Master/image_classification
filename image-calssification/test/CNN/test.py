import torch
from torch import nn
from d2l import torch as d2l

from model.resnet.train import train

net = nn.Sequential(
    # 这里使用一个11*11的更大窗口来捕捉对象。
    # 同时，步幅为4，以减少输出的高度和宽度。
    # 另外，输出通道的数目远大于LeNet
    nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1),
    nn.ReLU()
)

batch_size = 256
# train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)

if __name__ == '__main__':

    X= torch.normal(0, 1, (batch_size, 1, 28, 28))
    Y = net(X)
    print(Y.shape)
    # lr, num_epochs = 0.01, 10
    # train(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
