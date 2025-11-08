import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PyEMD import CEEMDAN
from PyEMD import EEMD
# 读取数据
df = pd.read_csv(r'D:\Users\三木\Desktop\shanghai.csv')

# 选择目标列
target_column = 'total'

# 提取目标信号
signal = df[target_column].values

# 应用CEEMDAN分解
ceemdan = EEMD()
imfs = ceemdan(signal)

# 创建一个DataFrame来存储IMFs
imfs_df = pd.DataFrame(index=df.index)
imfs_df[f'原始信号'] = signal

# 将每个IMF添加到DataFrame中
for i, imf in enumerate(imfs, start=1):
    imfs_df[f'IMF {i}'] = imf

# 保存到Excel
output_file = r'D:\Users\三木\Desktop\EEMD\shanghai分解结果.xlsx'
with pd.ExcelWriter(output_file) as writer:
    imfs_df.to_excel(writer)

print(f"CEEMDAN分解结果已保存至: {output_file}")

# # 如果还需要绘图部分，可以保留下面的代码
# # 绘制原始信号和每个IMF
# plt.rcParams['font.sans-serif'] = ['SimSun']
# plt.rcParams['font.serif'] = ['Times New Roman']
# plt.rcParams['axes.unicode_minus'] = False
#
# plt.figure(figsize=(16, 12))
# plt.subplot(len(imfs) + 1, 1, 1)
# plt.plot(signal, label='原始信号', color='blue')
# plt.legend(loc='right')
#
# for i, imf in enumerate(imfs):
#     plt.subplot(len(imfs) + 1, 1, i + 2)
#     plt.plot(imf, label=f'IMF {i + 1}', color='green')
#     plt.legend(loc='right')
#
# for ax in plt.gcf().axes:
#     ax.tick_params(axis='y', labelsize=8)
#
# plt.tight_layout(h_pad=2)
# plt.show()