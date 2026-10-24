"""已完成的实验入口；核心手写算法见 ../regression_core.py。
从任意工作目录执行均可，结果统一写入 Lab2/outputs。
拓展任务遵循实验报告，使用 winequality-white.csv。
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from regression_core import design, normal_fit, bgd, ridge_fit, mse
from run_all import task_ext

def poly_features(x, degree):
    # 保留原模板接口：[1, x, ..., x^degree]（单变量）。
    x = np.asarray(x).reshape(-1)
    return np.column_stack([x ** k for k in range(degree + 1)])

def fit_closed_form(X, y):
    return normal_fit(X, y)


if __name__ == '__main__':
    task_ext()
