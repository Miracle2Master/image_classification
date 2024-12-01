import torch
from torch import nn
from d2l import torch as d2l

from script.model_trainer import ModelTrainer
from utils.data_loader import get_std_data

net = nn.Sequential(nn.Flatten(),
                    nn.Linear(784, 256),
                    nn.ReLU(),
                    nn.Linear(256, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights);

batch_size, lr, num_epochs = 256, 0.1, 10
loss = nn.CrossEntropyLoss(reduction='none')
opti = torch.optim.SGD(net.parameters(), lr=lr)

train_iter, test_iter = get_std_data(size=28)
if __name__ == '__main__':
    trainer = ModelTrainer(net, optimizer=opti)
    trainer.train(train_iter, test_iter, "mlp")
