import numpy as np
import pandas as pd


data = pd.read_csv('Train_data.csv', header=None)  # 无表头数据
numpy_array = data.values  # 转换为(4000, 31)的numpy数组


means = np.mean(numpy_array, axis=0)




cov_matrix = np.cov(numpy_array, rowvar=False, bias=False)

print("均值向量形状:", means.shape)
print("协方差矩阵形状:", cov_matrix.shape)


df_means = pd.DataFrame(means.reshape(1, -1),
                       columns=[f'Col_{i}' for i in range(31)],
                       index=['Mean'])
df_cov = pd.DataFrame(cov_matrix,
                     columns=[f'Col_{i}' for i in range(31)],
                     index=[f'Col_{i}' for i in range(31)])

print("\n均值向量：")
print(df_means)
print("\n协方差矩阵前31x31部分：")
print(df_cov.iloc[:31,:31])
df_cov.to_csv("cov_data.csv", index=False)
df_means.to_csv("means_data.csv", index=False)