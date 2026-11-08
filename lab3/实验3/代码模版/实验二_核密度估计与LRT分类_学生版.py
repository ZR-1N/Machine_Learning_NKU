"""任务2运行入口。"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from density_core import MyGaussianKernelDensity, MyKernelClassifier
from run_all import find_best_h
from run_all import run

if __name__ == '__main__':
    run('2')
