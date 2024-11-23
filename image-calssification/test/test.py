from d2l import torch as d2l
import math
import numpy as np

from utils.paint import paint, pointgenerate, Points

index = 0
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)


# print(test_iter)
# y_ax = []
#
def normal(x, mu, sigma):
    p = 1 / math.sqrt(2 * math.pi * sigma ** 2)
    return p * np.exp(-0.5 / sigma ** 2 * (x - mu) ** 2)


if __name__ == '__main__':
    x = np.arange(-7, 7, 0.01)
    y = np.arange(0, 14, 0.01)
    z = np.arange(7, 21, 0.01)

    # 均值和标准差对
    params = [(0, 1), (0, 2), (3, 1)]
    # d2l.plot(x, [normal(x, mu, sigma) for mu, sigma in params], xlabel='x',
    #          ylabel='p(x)', figsize=(4.5, 2.5),
    #          legend=[f'mean {mu}, std {sigma}' for mu, sigma in params])
    # d2l.plt.savefig('test.png')
    # d2l.plt.show()

    from pathlib import Path

    # 获取当前文件的绝对路径
    current_file_path = Path(__file__).resolve()

    # 获取项目根目录
    project_root = current_file_path.parent.parent

    print("项目根目录:", project_root)

    print(type(x))
    print(type(y))
    # paint(x, [y,z], xlabel='x', title='normal',
    #       legend='2x')

    a = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    points = Points(3)
    for x in a:
        points.add(x, [x ** 2, x ** 3, x ** 4])
    y = points.get_val()
    print(a)
    print(y)
    paint(np.array(a), y, xlabel='x', title='normal', legend=['x^2', 'x^3', 'x^4'])
