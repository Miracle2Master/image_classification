from script.model_tester import ModelTester
from utils.data_loader import get_std_data
from utils.file_operate import get_trained_model
from train2 import get_model
from d2l import torch as d2l

model = get_model()
model = model.to(d2l.try_gpu())
model.load_state_dict(get_trained_model("densenet"))
_, test_iter = get_std_data(size=96)
tester = ModelTester(model)
tester.TestModel(test_iter, d2l.try_gpu())
print(tester.accuracy_score())
print(f"f1分数{tester.calculate_f1()}")
print(f"召回率：{tester.calculate_recall()}")

tester.get_ConfusionMatrix("denseNet")
