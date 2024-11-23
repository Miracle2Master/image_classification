import torch
from d2l import torch as d2l
from d2l.torch import evaluate_accuracy_gpu

from model.resnet.resnet import get_model
from utils.data_loader import get_data
from utils.data_loader import predict

# import d2l
# train_iter, test_iter = get_data()

# import d2l
# train_iter, test_iter = get_data()

device = d2l.try_gpu()

# 获取模型和数据加载器
model = get_model().to(device)
train_iter, test_iter = get_data()

# 加载模型权重
model.load_state_dict(torch.load("../checkpoints/model.pth", weights_only=True))

if __name__ == '__main__':
    predict(model, test_iter, device)
    print(evaluate_accuracy_gpu(model.to(d2l.try_gpu()), test_iter))
