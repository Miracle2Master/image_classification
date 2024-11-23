from d2l import torch as d2l
import math
import numpy as np
import os

from pathlib import Path

from utils.data_loader import get_project_root

# 获取当前文件的绝对路径
current_file_path = Path(__file__).resolve()

# 获取项目根目录
project_root = current_file_path.parent.parent
if __name__ == '__main__':
    print(project_root)


def paint(x, y, dir_name, xlabel='x', ylabel='y', figsize=(4.5, 2.5), title='title',
          legend=None,
          print_pic=False, loc=str(project_root) + "/img/",
          point_x=None, point_y=None):
    if legend is None:
        legend = ['train_loss', 'train_acc', 'test_acc']
    d2l.plot(x, y, xlabel=xlabel,
             ylabel=ylabel, figsize=figsize,
             legend=legend)
    d2l.plt.title(title)
    tag = ["loss", "tr", "te"]
    pos = ["right", "right", "left"]

    va = ['top', 'top', 'bottom']
    if point_x is not None and point_y is not None:
        for i, (tag, px, py, p, v) in enumerate(zip(tag, point_x, point_y, pos, va)):
            d2l.plt.text(px, py, f'({tag}:{py:.2f})', fontsize=10, ha=p, va=v)
    if print_pic:
        dir_path = get_project_root()+"/img/"
        # 检查目录是否存在，如果不存在则创建
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        d2l.plt.savefig(dir_path + title)
        print(f"保存训练过程成功{dir_path}{title}")
    d2l.plt.show()


'''
    添加点，最后绘图
'''


class pointgenerate:
    def __init__(self):
        self.x = []
        self.y = []

    def add(self, x, y):
        self.x.append(x)
        self.y.append(y)

    def get_val(self):
        return self.x, self.y


class Points:
    def __init__(self, n):
        self.points = [pointgenerate() for _ in range(n)]
        self.n = n

    def add(self, x, *args):
        for i in range(self.n):
            # print(args[0][i])
            self.points[i].add(x, args[0][i])

    def get_val(self):
        ans = []
        x_val = []
        for i in range(self.n):
            x_val, y_val = self.points[i].get_val()
            ans.append(np.array(y_val))
        return np.array(x_val), ans

#
# points = Points(3)
# points.add(1, [2, 3, 4])
# points.add(2, [4, 3, 2])
# print(points.get_val())  # 输出每个 PointGenerate 实例的 y 值的 NumPy 数组列表
#
# paint(points.get_val()[0], points.get_val()[1])
