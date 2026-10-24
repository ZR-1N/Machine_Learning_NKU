"""已完成的实验入口；核心手写算法见 ../regression_core.py。
从任意工作目录执行均可，结果统一写入 Lab2/outputs。
拓展任务遵循实验报告，使用 winequality-white.csv。
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from regression_core import design, normal_fit, bgd, ridge_fit, mse
from run_all import task1

def normal_equation_fit(x_train, y_train):
    theta = normal_fit(design(x_train), y_train)
    return float(theta[0]), float(theta[1])


if __name__ == '__main__':
    task1()
