from d2l import torch as d2l
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import FashionMNIST


def get_data(batch_size=256, size=96):
    """
    加载数据
    :return: train_iter, test_iter
    """
    return d2l.load_data_fashion_mnist(batch_size, resize=size)


from pathlib import Path

# 获取当前文件的绝对路径
current_file_path = Path(__file__).resolve()

# 获取项目根目录
project_root = current_file_path.parent.parent


def get_std_data(batch_size=256, size=96):
    # # 数据预处理和加载
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]) ,# Fashion-MNIST的标准化
        transforms.Resize(size)
    ])

    print(str(project_root))
    train_dataset = FashionMNIST(root=str(project_root) + '/data', train=True, download=True, transform=transform)
    test_dataset = FashionMNIST(root=str(project_root) + '/data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def get_std_MNIST_data(batch_size=256, resize=96):
    # # 数据预处理和加载
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,),
                             transforms.Resize(resize))
    ])
    train_dataset = datasets.MNIST(root=str(project_root) + '/data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=str(project_root) + '/data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


# 修改数据加载器，使其在每次迭代时移动数据到指定设备
def to_device(data, device):
    """将数据移动到指定设备"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device)


# 包装原始的数据迭代器
class DeviceDataLoader():
    """将数据加载器中的数据移动到指定设备"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """每次迭代时，将数据移动到指定设备"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """返回原始数据加载器的长度"""
        return len(self.dl)


def predict(net, test_iter, device, n=6):
    test_iter = DeviceDataLoader(test_iter, device)
    """
    用于测试给定网络（net）在测试集（test_iter）上的预测效果，并显示预测结果。
    - net: 训练好的神经网络模型，负责进行预测
    - test_iter: 测试数据的迭代器，提供测试样本和标签
    - n: 要显示的图像数量，默认为 6
    """
    # 遍历测试数据迭代器中的每一批数据
    # for x, y in test_iter:
    x, y = next(iter(test_iter))
    # 获取真实标签
    trues = d2l.get_fashion_mnist_labels(y)

    # 使用网络进行预测，并获取预测标签
    preds = d2l.get_fashion_mnist_labels(net(x).argmax(dim=1))

    # 创建标题列表，每个标题包含真实标签和预测标签
    titles = [true + ',' + pred for true, pred in zip(trues, preds)]
    # print(trues,'\n', preds)

    x_cpu = x[:n].cpu().reshape((n, 96, 96))
    # 显示前 n 张图像，图像数据需要重塑为 (n, 28, 28) 的形状
    d2l.show_images(x_cpu, 1, n, titles=titles)
    d2l.plt.show()


def get_project_root() -> str:
    return str(project_root)


if __name__ == '__main__':
    root = get_project_root()
    print(root)
