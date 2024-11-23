import torch
from d2l.torch import evaluate_accuracy_gpu, get_fashion_mnist_labels
from d2l import torch as d2l

from model.vit.vit_train import ViT
from script.model_tester import ModelTester
from utils.data_loader import get_data, get_std_data
from utils.file_operate import get_trained_model

d2l.use_svg_display()
vit0_1 = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "batch_size": 64,
    "epochs": 21,
    "lr": 1e-4,
    "img_size": 28,  # Fashion-MNIST 图片大小 28x28
    "patch_size": 7,  # ViT的patch大小
    "num_patches": (28 // 7) ** 2,
    "embed_dim": 256,  # 嵌入维度
    "num_heads": 8,  # 注意力头数
    "num_layers": 6,  # Transformer层数
    "num_classes": 10  # Fashion-MNIST 分类数
}

# 配置参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_size = 28  # Fashion-MNIST 图片大小 28x28

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  #@save
    """绘制图像列表"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 图片张量
            ax.imshow(img.numpy())
        else:
            # PIL图片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes
def getTag(x, model):
    model = model.to(device)
    x = x.to(device)
    out = model(x)
    x=x.cpu()
    # 假设 get_fashion_mnist_labels 返回的是张量
    titles = get_fashion_mnist_labels(torch.argmax(out, dim=1).cpu())
    # titles = torch.tensor(titles).numpy()  # 确保 titles 是张量，并且在调用 .cpu() 时它是张量

    # 然后传给 show_images 函数
    show_images(x.reshape(5, 1, 28, 28), 1, 5, titles=titles)
    return out


if __name__ == '__main__':
    model = ViT(vit0_1)
    model = model.to(device)
    model.load_state_dict(get_trained_model("vit"))
    _, test_iter = get_std_data(size=img_size,batch_size=5)

    # for i in range(10):
    #     sample = next(iter(test_iter))
    #     x = sample[0]
    #     print(x.shape)
    #     getTag(x, model)
    #     x_tag = sample[1]

    tester = ModelTester(model)
    tester.TestModel(test_iter, device)
    print(f"精确率{tester.accuracy_score()}")
    print(f"f1分数{tester.calculate_f1()}")
    print(f"召回率：{tester.calculate_recall()}")
    tester.get_ConfusionMatrix("Vision Transformer")

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model2 = get_model().to(device)
# train_iter, test_iter = get_data().to(device)
# model2.load_state_dict(torch.load("../checkpoints/model2.pth"))
# if __name__ == '__main__':
#     pass
