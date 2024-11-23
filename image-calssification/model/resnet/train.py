import os

from d2l import torch as d2l
from d2l.torch import evaluate_accuracy_gpu
from torch import nn

from model.resnet.resnet import get_model
from utils.data_loader import get_data
from utils.paint import Points, paint


def train(net, train_iter, test_iter, num_epochs, lr, device):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    timer, num_batches = d2l.Timer(), len(train_iter)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    point_list = Points(3)

    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples
        metric = d2l.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()

            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            test_acc = evaluate_accuracy_gpu(net, test_iter)

            test_batch, test_point_nums = num_batches, 50
            # 训练一轮生成m=50 个点
            if (i + 1) % (test_batch // test_point_nums) == 0 or i == test_batch - 1:
                print(f"x = {epoch + (i + 1) / test_batch}")
                point_list.add(epoch + (i + 1) / test_batch,
                               (train_l, train_acc, test_acc))
            # animator.add(epoch + 1, (None, None, test_acc))
            # point_list.add(i, [train_l, train_acc, test_acc])
            print(f'batch loc：{i + 1}/{test_batch}，loss {train_l:.3f}, train acc {train_acc:.3f}, '
                  f'test acc {test_acc:.3f}')
            # 训练过得总的数据量*轮次 / 时间
            print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
                  f'on {str(device)}')
            if i == test_batch:
                break
        if optimizer.param_groups[0]["lr"] > 0.05:
            scheduler.step()
            print(f'lr = {optimizer.param_groups[0]["lr"]}')
        paint(point_list.get_val()[0], point_list.get_val()[1], print_pic=True, title="epochs=" + str(epoch + 1))


import torch

if __name__ == '__main__':
    net = get_model()
    X = torch.rand(size=(1, 1, 96, 96))
    train_iter, test_iter = get_data()

    # firstdata, ydata = next(iter(train_iter))
    # print(type(firstdata))
    # print(firstdata.shape)
    # for layer in net:
    #     X = layer(X)
    #     print(layer.__class__.__name__, "out:\t", X.shape)
    train(net, train_iter, test_iter, 10, 0.1, d2l.try_gpu())
    # 指定保存路径
    save_path = 'checkpoints/model2.pth'
    # 确保文件夹存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # 保存模型
    torch.save(net.state_dict(), save_path)
    # 一个迭代器中有235组数据，每组数据有256个图片
    # print(i, X.shape, y.shape)
# print(train_iter.__len__(), test_iter.__len__())
