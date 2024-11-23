from script.model_tester import ModelTester
from utils.data_loader import get_std_data
from utils.file_operate import get_trained_model
from torch import nn

if __name__ == '__main__':
    model = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
    model.load_state_dict(get_trained_model("softmax"))
    _,test_iter = get_std_data()
    tester = ModelTester(model)
    tester.TestModel(test_iter)
    print(f"测试准确率：{tester.accuracy_score():.2f}")
    print(f"F1分数为：{tester.calculate_f1():.2f}")
    print(f"精准率为：{tester.calculate_precision():.2f}")
    print(f"召回率为：{tester.calculate_recall():.2f}")
