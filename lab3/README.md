# 实验三：参数估计与非参数估计

## 运行

依赖：NumPy、Matplotlib、Scikit-learn。安装版本要求见 `requirements.txt`。

从包含 `lab3/` 的项目根目录执行：

```powershell
cd lab3
python -m pip install -r requirements.txt
python run_all.py
```

单项实验：

```powershell
python task1.py
python task2.py
python task3.py
python task4.py
```

也可用 `python run_all.py --task 1` 指定任务，任务号为1、2、3、4。

## 文件

- `density_core.py`：高斯分类、MAP/LRT、高斯核密度与kNN密度估计。
- `run_all.py`：数据生成、交叉验证、评估与绘图。
- `task1.py` 至 `task4.py`：四项实验的运行入口。
- `data/`：两组数据与固定训练测试索引。
- `outputs/`：参数、错误率、逐折验证、预测、密度网格与图像。
- `实验报告.pdf`：实验原理、实现方法、实验结果与分析。

## 参数设置

每组数据500个样本、2个特征，随机种子42。高、低分离度分别为2.0和0.5。核分类采用五折交叉验证，候选带宽为 `[0.1, 0.5, 1, 1.5, 2]`。kNN密度估计取 `k=1,3,5`，不使用标签。

任务1、任务2报告全数据拟合错误率。留出实验采用350条训练、150条测试记录；核分类仅在训练集内部选择带宽。任务4比较完整协方差与对角协方差高斯模型。分类错误率的评价范围在结果文件中分别标明。
