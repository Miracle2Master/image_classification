import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import math

from script import model_trainer
from utils.data_loader import get_std_data, get_project_root

project_root = get_project_root()
# 数据预处理和增强
transform = transforms.Compose([
    transforms.Resize(224),  # ViT模型要求输入大小通常是224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20)
])

# 加载Fashion-MNIST数据集
train_dataset = datasets.FashionMNIST(root=str(project_root)+'/data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root=str(project_root)+'/data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)


# ViT模型定义
class VisionTransformer(nn.Module):
    def __init__(self, image_size=224, patch_size=16, num_classes=10, dim=768, depth=12, heads=12, mlp_dim=3072,
                 dropout=0.1):
        super(VisionTransformer, self).__init__()

        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_dim = patch_size ** 2  # RGB 图像每个patch有3通道

        # Patch Embedding
        self.patch_embeddings = nn.Conv2d(in_channels=1, out_channels=dim, kernel_size=patch_size, stride=patch_size)

        # Positional Embedding
        self.positional_embeddings = nn.Parameter(torch.randn(1, self.num_patches, dim))

        # Transformer Encoder
        self.transformer_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout)
            for _ in range(depth)
        ])

        # 分类头
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        # Patch Embedding
        x = self.patch_embeddings(x)  # Shape: [batch_size, dim, num_patches, 1]
        x = x.flatten(2).transpose(1, 2)  # Shape: [batch_size, num_patches, dim]

        # Add positional embeddings
        x = x + self.positional_embeddings

        # Transformer Encoder
        for block in self.transformer_blocks:
            x = block(x)

        # Classifier (using the representation of the first token)
        x = x.mean(dim=1)  # Global average pooling
        x = self.mlp_head(x)

        return x


# 初始化模型
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = VisionTransformer().to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 学习率调度器
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)



if __name__ == '__main__':
    trainer = model_trainer.ModelTrainer(model, num_epochs=10)
    trainer.train(train_loader, test_loader,"MNIST-ViT2")

    print("Training complete!")
    # X =torch.randn(1, 3, 224, 224)
    # for layer in model:
    #     X = layer(X)
    #     print(layer.__class__.__name__, "out:\t", X.shape)