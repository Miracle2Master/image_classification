from hyperopt import fmin, tpe, hp, Trials


def objective(args):
    # 目标函数，接收超参数字典
    x = args['x']
    b = args['b']
    return x ** 2 + b ** 3  # 一个简单的二次函数，最小值在 x=2


# 定义超参数空间
space = {
    'x': hp.uniform('x', -10, 10),  # x 的值在 [-10, 10] 范围内均匀分布
    'b': hp.uniform('b', -10, 10)
}
# 创建 Trials 对象来记录结果
trials = Trials()
# 运行贝叶斯优化
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=50, trials=trials)
print("Best parameters found: ", best)
