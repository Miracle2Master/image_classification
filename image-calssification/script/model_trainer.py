import os

import math
import torch
from d2l import torch as d2l
from d2l.torch import evaluate_accuracy_gpu
from hyperopt import fmin, tpe, hp, Trials
from sklearn.model_selection import KFold
from torch import nn
from torch.utils.data import DataLoader, SubsetRandomSampler

from script.model_tester import ModelTester
from utils.data_loader import get_project_root
from utils.file_operate import get_trained_model
from utils.paint import Points, paint


class ModelTrainer:

    def __init__(self, net,
                 num_epochs=10,
                 lr=1e-4,
                 optimizer=None,
                 device=d2l.try_gpu(),
                 dropout=0.2,
                 weight_decay=1e-5):
        self.net = net
        self.num_epochs = num_epochs
        self.lr = lr
        self.device = device
        self.dropout = dropout
        if optimizer is None:
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        else:
            self.optimizer = optimizer

    def save_model(self, name):
        project_root = get_project_root()
        save_path = str(project_root) + f"/store/model/"
        # 确保目标文件夹存在
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(self.net.state_dict(), f"{save_path}{name}.pth")
        print(f"保存模型成功，保存到了{save_path}{name}")

    def evaluate_test_loss(self, test_iter):
        loss = nn.CrossEntropyLoss()
        test_loss_sum = 0.0
        self.net.eval()
        with torch.no_grad():
            for X, y in test_iter:
                X, y = X.to(self.device), y.to(self.device)
                y_hat = self.net(X)
                l = loss(y_hat, y)
                test_loss_sum = l.item()
        test_loss = test_loss_sum
        return test_loss

    def train(self, train_iter, test_iter, model_name, scheduler=None):
        def init_weights(m):
            if type(m) is nn.Linear or type(m) is nn.Conv2d:
                nn.init.xavier_uniform_(m.weight)
            if type(m) is nn.Dropout:
                m.p = self.dropout

        self.net.apply(init_weights)
        print('training on', self.device)
        self.net.to(self.device)
        loss = nn.CrossEntropyLoss()
        timer, num_batches = d2l.Timer(), len(train_iter)
        # timer, num_batches = d2l.Timer(), 10
        scheduler = None
        point_list = Points(3)

        for epoch in range(self.num_epochs):
            # Sum of training loss, sum of training accuracy, no. of examples
            metric = d2l.Accumulator(3)
            self.net.train()
            for i, (X, y) in enumerate(train_iter):
                timer.start()
                self.optimizer.zero_grad()
                X, y = X.to(self.device), y.to(self.device)
                y_hat = self.net(X)
                l = loss(y_hat, y)
                l.backward()
                self.optimizer.step()

                test_batch, print_point_nums = num_batches, 25
                # 训练一轮生成m=50 个点
                if (i + 1) % (test_batch // print_point_nums) == 0 or i == test_batch - 1:
                    with torch.no_grad():
                        metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
                        timer.stop()
                        train_l = metric[0] / metric[2]
                        # assert train_l > 5, "模型损失过大"
                        train_acc = metric[1] / metric[2]
                        test_acc = evaluate_accuracy_gpu(self.net, test_iter)
                    print(f"x = {epoch + (i + 1) / test_batch}")
                    point_list.add(epoch + (i + 1) / test_batch,
                                   (train_l, train_acc, test_acc))
                    print(f'batch loc：{i + 1}/{test_batch}，loss {train_l:.3f}, train acc {train_acc:.3f}, '
                          f'test acc {test_acc:.3f}')
                    # 训练过得总的数据量*轮次 / 时间
                    print(f'{metric[2] * self.num_epochs / timer.sum():.1f} examples/sec '
                          f'on {str(self.device)}')
                if i == test_batch:
                    break
            if (scheduler is not None) and self.optimizer.param_groups[0]["lr"] > 0.05:
                scheduler.step()
                print(f'lr = {self.optimizer.param_groups[0]["lr"]}')
            paint(point_list.get_val()[0], point_list.get_val()[1], dir_name=model_name, print_pic=True,
                  title="epochs=" + str(epoch + 1))
        self.save_model(f'{model_name}')

    def fast_train(self, train_iter, test_iter, model_name, scheduler=None):
        def init_weights(m):
            if type(m) is nn.Linear or type(m) is nn.Conv2d:
                nn.init.xavier_uniform_(m.weight)
            if type(m) is nn.Dropout:
                m.p = self.dropout

        self.net.apply(init_weights)
        print('training on', self.device)
        self.net.to(self.device)
        loss = nn.CrossEntropyLoss()
        timer, num_batches = d2l.Timer(), len(train_iter)
        # timer, num_batches = d2l.Timer(), 10
        scheduler = None
        point_list = Points(3)
        # L1惩罚系数
        l1_lambda = 1e-5
        for epoch in range(self.num_epochs):
            # Sum of training loss, sum of training accuracy, no. of examples
            metric = d2l.Accumulator(3)
            self.net.train()
            for i, (X, y) in enumerate(train_iter):
                timer.start()
                self.optimizer.zero_grad()
                X, y = X.to(self.device), y.to(self.device)
                y_hat = self.net(X)

                l = loss(y_hat, y)
                # # 添加L1正则化项
                # l1_norm = sum(p.abs().sum() for p in self.net.parameters())
                # l += l1_lambda * l1_norm
                l.backward()
                self.optimizer.step()
                test_batch = num_batches

                if i == test_batch - 1 or i == test_batch // 2:
                    with torch.no_grad():
                        metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
                        timer.stop()
                        train_l = metric[0] / metric[2]
                        train_acc = metric[1] / metric[2]
                        test_acc = evaluate_accuracy_gpu(self.net, test_iter)
                    print(f"x = {epoch + (i + 1) / test_batch}")
                    point_list.add(epoch + (i + 1) / test_batch,
                                   (train_l, train_acc, test_acc))
                    print(f'batch loc：{i + 1}/{test_batch}，loss {train_l:.3f}, train acc {train_acc:.3f}, '
                          f'test acc {test_acc:.3f}')
                    # 训练过得总的数据量*轮次 / 时间
                    print(f'{metric[2] * self.num_epochs / timer.sum():.1f} examples/sec '
                          f'on {str(self.device)}')
                if i == test_batch:
                    break
        # point_x = [self.num_epochs, self.num_epochs, self.num_epochs]
        # point_y = [train_l, train_acc, test_acc], point_x=point_x, point_y=point_y
        paint(point_list.get_val()[0], point_list.get_val()[1], dir_name=model_name, print_pic=True,
              title=model_name)
        self.save_model(f'{model_name}')

    def bayesian_fast_train(self, train_iter, test_iter, model_name, scheduler=None):
        def init_weights(m):
            if type(m) is nn.Linear or type(m) is nn.Conv2d:
                nn.init.xavier_uniform_(m.weight)
            if type(m) is nn.Dropout:
                m.p = self.dropout

        self.net.apply(init_weights)
        print('training on', self.device)
        self.net.to(self.device)
        loss = nn.CrossEntropyLoss()
        timer, num_batches = d2l.Timer(), len(train_iter)
        # timer, num_batches = d2l.Timer(), 10
        scheduler = None
        point_list = Points(3)
        # L1惩罚系数
        l1_lambda = 1e-5
        for epoch in range(self.num_epochs):
            # Sum of training loss, sum of training accuracy, no. of examples
            metric = d2l.Accumulator(3)
            self.net.train()
            for i, (X, y) in enumerate(train_iter):
                timer.start()
                self.optimizer.zero_grad()
                X, y = X.to(self.device), y.to(self.device)
                y_hat = self.net(X)
                l = loss(y_hat, y)
                # # 添加L1正则化项
                # l1_norm = sum(p.abs().sum() for p in self.net.parameters())
                # l += l1_lambda * l1_norm
                l.backward()
                self.optimizer.step()
                test_batch = num_batches
                if i == test_batch - 1 or i == test_batch // 2:
                    with torch.no_grad():
                        metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
                        timer.stop()
                        train_l = metric[0] / metric[2]
                        train_acc = metric[1] / metric[2]
                        test_acc = evaluate_accuracy_gpu(self.net, test_iter)
                    print(f"x = {epoch + (i + 1) / test_batch}")
                    point_list.add(epoch + (i + 1) / test_batch,
                                   (train_l, train_acc, test_acc))
                    print(f'batch loc：{i + 1}/{test_batch}，loss {train_l:.3f}, train acc {train_acc:.3f}, '
                          f'test acc {test_acc:.3f}')
                    # 训练过得总的数据量*轮次 / 时间
                    print(f'{metric[2] * self.num_epochs / timer.sum():.1f} examples/sec '
                          f'on {str(self.device)}')

        point_x = [self.num_epochs, self.num_epochs, self.num_epochs]
        point_y = [train_l, train_acc, test_acc]
        paint(point_list.get_val()[0], point_list.get_val()[1], dir_name=model_name, print_pic=True,
              title="epochs=" + str(epoch + 1), point_x=point_x, point_y=point_y)
        self.save_model(f'{model_name}')

    def re_fast_train(self, train_iter, test_iter, model_name, scheduler=None):
        self.net.load_state_dict(get_trained_model(model_name))
        model_tester = ModelTester(self.net)
        model_tester.TestModel(test_iter, self.device)
        print(f"准确率为:{model_tester.accuracy_score()}")
        self.fast_train(train_iter, test_iter, model_name, scheduler)

    def train_with_bayesian_optimization(self, train_iter, test_iter, name):
        def objective(args):
            lr = args['lr']
            dropout_rate = args['dropout_rate']
            num_epochs = args['num_epochs']
            # 这里可以添加其他超参数的设置
            self.lr = lr

            self.dropout = dropout_rate
            self.num_epochs = num_epochs

            self.bayesian_fast_train(train_iter, test_iter,
                                     model_name=f"{name}_{math.log(lr)}_{dropout_rate}_{num_epochs}")
            # 返回验证集上的损失，这里假设evaluate_test_loss是计算验证集损失的方法
            return self.evaluate_test_loss(test_iter)

        # 定义超参数空间
        space = {
            'lr': hp.choice('lr', [1e-5, 1e-4, 1e-3]),
            'dropout_rate': hp.uniform('dropout_rate', 0.0, 0.9),
            'num_epochs': hp.choice('num_epochs', [10, 15, 20]),
            # 可以在这里添加更多的超参数
        }

        # 运行贝叶斯优化
        trials = Trials()
        best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=10, trials=trials)

        print("Best parameters found: ", best)
        # 找到最好的参数，然后用最好的参数，训练模型，保存模型
        # 正常情况下，应该全部训练完成，保存最好的超参数，以及模型参数
        # space，是参数，会进行排列组合

    # TODO k折交叉验证未完成
    def train_by_fold(self, train_iter, test_iter, use_scheduler=False, num_folds=5):
        def init_weights(m):
            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                nn.init.xavier_uniform_(m.weight)

        kfold = KFold(n_splits=num_folds, shuffle=True, random_state=1)

        for fold, (train_ids, val_ids) in enumerate(kfold.split(train_iter.dataset)):
            print(f'Fold {fold + 1}')
            self.net.apply(init_weights)
            self.net.to(self.device)
            optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
            loss = nn.CrossEntropyLoss()
            timer, num_batches = d2l.Timer(), len(train_ids)
            scheduler = None
            if use_scheduler:
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

            point_list = Points(3)
            train_sampler = SubsetRandomSampler(train_ids)
            val_sampler = SubsetRandomSampler(val_ids)
            train_loader = DataLoader(train_iter.dataset, batch_size=train_iter.batch_size, sampler=train_sampler)
            val_loader = DataLoader(train_iter.dataset, batch_size=train_iter.batch_size, sampler=val_sampler)

            for epoch in range(self.num_epochs):
                metric = d2l.Accumulator(3)
                self.net.train()
                for i, (X, y) in enumerate(train_loader):
                    optimizer.zero_grad()
                    X, y = X.to(self.device), y.to(self.device)
                    y_hat = self.net(X)
                    l = loss(y_hat, y)
                    l.backward()
                    optimizer.step()
                    metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
                    train_l = metric[0] / metric[2]
                    train_acc = metric[1] / metric[2]
                    print(f'Epoch {epoch + 1}, Fold {fold + 1}, Batch {i + 1}/{num_batches}, '
                          f'loss {train_l:.3f}, train acc {train_acc:.3f}')
                    val_acc = evaluate_accuracy_gpu(self.net, val_loader)
                    print(f'Epoch {epoch + 1}, Fold {fold + 1}, Val Acc: {val_acc:.3f}')
                    if use_scheduler:
                        scheduler.step()
