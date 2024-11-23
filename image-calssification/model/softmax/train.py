import torch
from torch import nn
from d2l import torch as d2l

from script.model_trainer import ModelTrainer
from utils.data_loader import get_std_data
from utils.file_operate import save_model

batch_size = 256
train_iter, test_iter = get_std_data()

# PyTorch不会隐式地调整输入的形状。因此，
# 我们在线性层前定义了展平层（flatten），来调整网络输入的形状
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


net.apply(init_weights)
loss = nn.CrossEntropyLoss(reduction='none')

trainer = torch.optim.SGD(net.parameters(), lr=0.1)
if __name__ == '__main__':
    model_trainer = ModelTrainer(net, 1, 0.1, trainer)
    model_trainer.train(train_iter, test_iter)
    save_model(net, "softmax")
