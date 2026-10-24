"""将课程骨架连接到已完成的手写实现；只修改 Lab2 的五个模板。"""
from pathlib import Path

root=Path(__file__).resolve().parent
folder=root/'代码模版'
items={
 'task1_framework.py':('task1', '''def normal_equation_fit(x_train, y_train):
    theta = normal_fit(design(x_train), y_train)
    return float(theta[0]), float(theta[1])
'''),
 'task2_framework_sgd.py':('task2', '''def gd_train(X, y, lr=0.05, epochs=3000):
    theta, hist = bgd(design(X), y, lr, epochs)
    return theta[:-1], float(theta[-1]), hist
'''),
 'task3_framework_sgd_lr.py':('task3', '''def gd_train(X, y, lr=0.05, epochs=3000):
    theta, hist = bgd(design(X), y, lr, epochs)
    return theta[:-1], float(theta[-1]), hist
'''),
 'task4_framework_ridge.py':('task4', '''def ridge_fit_closed_form(X, y, lam):
    # 与原模板一致：输入 X 已经在最后追加常数列。
    return ridge_fit(X, y, lam)
'''),
 'task_ext_poly_framework.py':('task_ext', '''def poly_features(x, degree):
    # 保留原模板接口：[1, x, ..., x^degree]（单变量）。
    x = np.asarray(x).reshape(-1)
    return np.column_stack([x ** k for k in range(degree + 1)])

def fit_closed_form(X, y):
    return normal_fit(X, y)
''')}

for filename,(task,body) in items.items():
    content='''"""已完成的实验入口；核心手写算法见 ../regression_core.py。
从任意工作目录执行均可，结果统一写入 Lab2/outputs。
拓展任务遵循实验报告，使用 winequality-white.csv。
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from regression_core import design, normal_fit, bgd, ridge_fit, mse
from run_all import TASK

BODY

if __name__ == '__main__':
    TASK()
'''.replace('TASK',task).replace('BODY',body)
    (folder/filename).write_text(content,encoding='utf-8')
print('Completed five template entry points.')
