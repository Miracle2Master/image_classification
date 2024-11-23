from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from script import model_trainer
from script.model_tester import ModelTester
from utils.data_loader import get_std_data, get_project_root
from utils.file_operate import save_model, get_trained_model

# 配置参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
epochs = 21
lr = 1e-4
img_size = 28  # Fashion-MNIST 图片大小 28x28
patch_size = 7  # ViT的patch大小
num_patches = (img_size // patch_size) ** 2
embed_dim = 256  # 嵌入维度
num_heads = 8  # 注意力头数
num_layers = 6  # Transformer层数
num_classes = 10  # Fashion-MNIST 分类数

# 定义 Vision Transformer (ViT) 模型
class ViT(nn.Module):
    def __init__(self, num_classes=10, embed_dim=256, num_heads=8, num_layers=6, patch_size=7, img_size=28,
                 dropout=0.2):
        super(ViT, self).__init__()

        self.patch_size = patch_size
        self.img_size = img_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.dropout = dropout

        # 创建图像分块层
        self.patch_embed = nn.Conv2d(in_channels=1, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size)

        # self.positional_embeddings = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))

        # Transformer编码器部分
        self.encoder_layers = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,dropout=self.dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layers, num_layers=num_layers)

        # self.classifier = nn.Sequential(
        #     nn.Linear(embed_dim, 1024),
        #     nn.Dropout(dropout),
        #     nn.ReLU(),
        #     nn.Linear(1024, 512),
        #     nn.Dropout(dropout),
        #     nn.ReLU(),
        #     nn.Linear(512, 256),
        #     nn.Dropout(dropout),
        #     nn.ReLU(),
        #     nn.Linear(256, num_classes)
        # )
        self.classifier = nn.Linear(embed_dim, num_classes)
        # self.activation = nn.ReLU()
        # self.classifier2 = nn.Linear(512, num_classes)
        # self.classifier = nn.Linear(self.embed_dim, num_classes)

    def forward(self, x):
        # 将输入图像分块并嵌入
        x = self.patch_embed(x)  # [batch_size, embed_dim, num_patches, num_patches]
        x = x.flatten(2).transpose(1, 2)  # [batch_size, num_patches, embed_dim]
        # x = x+self.positional_embeddings

        # Transformer编码
        x = self.transformer_encoder(x)

        # 取最后的token进行分类
        x = x.mean(dim=1)  # [batch_size, embed_dim]
        x = self.classifier(x)

        return x


# 实例化模型并移动到设备（GPU/CPU）
model = ViT(num_classes=num_classes, embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers,
            patch_size=patch_size, img_size=img_size,dropout=0)
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()

weight_decay = 1e-4  # 这是一个常用的权重衰减值
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
# 获取项目根目录
project_root = get_project_root()
train_loader, test_loader = get_std_data(size=img_size)
if __name__ == '__main__':

    # for i, (X, y) in enumerate(train_loader):
    #     print(X.shape, y.shape)
    trainer = model_trainer.ModelTrainer(model, num_epochs=30)
    trainer.fast_train(train_loader, test_loader, model_name="vit_not_decay")
    # trainer.re_fast_train(train_loader, test_loader,"vit1_0")


    # model = model
    # model = model.to(device)
    # model.load_state_dict(get_trained_model("vit_pos"))
    # tester = ModelTester(model)
    # tester.TestModel(test_loader, device)
    # print(f"精确率{tester.accuracy_score()}")
    # print(f"f1分数{tester.calculate_f1()}")
    # print(f"召回率：{tester.calculate_recall()}")
    # tester.get_ConfusionMatrix("Vision Transformer pos")

