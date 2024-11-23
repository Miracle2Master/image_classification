import os

import torch
import torchvision
from d2l.torch import d2l, get_fashion_mnist_labels
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

from utils.data_loader import get_project_root
from utils.file_operate import save_image


class ModelTester:
    def __init__(self, model):
        self.total = 0
        self.correct = 0
        self.model = model
        # 用于存储预测结果和实际标签
        self.y_pred = []
        self.y_true = []
        self.acc = 0
        self.labels = []
        for i in range(10):
            self.labels.append(get_fashion_mnist_labels(torch.tensor([i]))[0])

    def show_CM(self, test_loader, classes):
        # 将模型设置为评估模式
        self.model.eval()

        # 初始化混淆矩阵
        confusion_matrix = torch.zeros(len(classes), len(classes))

        # 测试模型并计算混淆矩阵
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                for t, p in zip(labels.view(-1), predicted.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1

        # 可视化混淆矩阵
        plt.figure(figsize=(10, 7))
        sns.heatmap(confusion_matrix.numpy(), annot=True, fmt='d', cmap='Blues', xticklabels=classes,
                    yticklabels=classes)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()

    def TestModel(self, test_iter, device=d2l.try_gpu()):
        # 不计算梯度，因为在评估模式下我们不会更新权重
        self.model.eval()
        with torch.no_grad():
            for X, y in test_iter:
                X, y = X.to(device), y.to(device)
                # 前向传播
                output = self.model(X)
                # 获取预测结果，通常选择最高概率的类别
                predictions = torch.argmax(output, dim=1)
                self.correct += (predictions == y).sum().item()
                self.total += y.size(0)
                self.y_pred.extend(predictions.tolist())
                self.y_true.extend(y.tolist())

    def accuracy_score(self):
        print(self.y_true)
        print(self.y_pred)
        print(len(self.y_true))
        return self.correct / self.total

    # 计算精确率
    def calculate_precision(self):
        return precision_score(self.y_true, self.y_pred, average='macro')

    # 计算召回率
    def calculate_recall(self):
        return recall_score(self.y_true, self.y_pred, average='macro')

    # 计算F1分数
    def calculate_f1(self):
        return f1_score(self.y_true, self.y_pred, average='macro')


    def get_ConfusionMatrix(self,name):
        conf_matrix = confusion_matrix(self.y_true, self.y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=self.labels)
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f"{name} Confusion Matrix", fontsize=10)

        # 设置x轴和y轴标签的字体大小
        plt.xticks(fontsize=7)
        plt.yticks(fontsize=7)
        plt.show()

        dir_path = get_project_root()
        dir_path = f'{dir_path}/ConfusionMatrix'
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        d2l.plt.savefig(dir_path + f"/{name}")
        plt.savefig(name + '.png')
