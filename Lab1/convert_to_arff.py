# -*- coding: utf-8 -*-
"""
把空格分隔的 semeion.data 转换成 Weka 原生 ARFF 格式。
Weka 不能直接读空格分隔的 txt,ARFF 是它最标准的输入格式。
运行:python convert_to_arff.py  ->  生成 data/semeion.arff

ARFF 文件结构:
  @relation 数据集名
  @attribute 属性名 类型      (逐列声明,这里 256 个像素列 + 1 个标签列)
  @data
  数据行(逗号分隔)
"""
import numpy as np

raw = np.loadtxt('data/semeion.data')                    # (1593, 266) 原数据
X = raw[:, :256]                                         # 前 256 列:0/1 像素
y = np.argmax(raw[:, 256:], axis=1)                      # 后 10 列独热 -> 0~9 整数标签

lines = []
lines.append('@relation semeion_handwritten_digit')      # 数据集名称
lines.append('')
for i in range(256):                                     # 声明 256 个像素属性
    lines.append(f'@attribute p{i+1} numeric')
# 标签属性:nominal 类型,Weka 只有 nominal 属性才能当分类目标(class)
lines.append('@attribute class {0,1,2,3,4,5,6,7,8,9}')
lines.append('')
lines.append('@data')
for row, label in zip(X, y):                             # 每行:256 个像素 + 1 个标签
    lines.append(','.join(str(int(v)) for v in row) + f',{int(label)}')

with open('data/semeion.arff', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print('转换完成 -> data/semeion.arff')
