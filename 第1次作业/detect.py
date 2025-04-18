import csv
import math
from scipy.stats import chi2
import numpy as np
import matplotlib.pyplot as plt

D = 31
N = 4000
pi = math.pi

# 读取train_data.csv, test_data1.csv, test_data2.csv文件
with open("F:/Edit Files/Pycharm Files/模式识别/第1次作业/Train_data.csv", "r", encoding="UTF-8") as f:
    train_data = [[0.0] * D for _ in range(N)]
    for row in range(N):
        lines = f.readline().strip().split(',')
        for col in range(D):
            train_data[row][col] = float(lines[col])
with open("F:/Edit Files/Pycharm Files/模式识别/第1次作业/Test_data1.csv", "r", encoding="UTF-8") as f:
    test_data1 = [[0.0] * D for _ in range(N)]
    for row in range(N):
        lines = f.readline().strip().split(',')
        for col in range(D):
            test_data1[row][col] = float(lines[col])
with open("F:/Edit Files/Pycharm Files/模式识别/第1次作业/Test_data2.csv", "r", encoding="UTF-8") as f:
    test_data2 = [[0.0] * D for _ in range(N)]
    for row in range(N):
        lines = f.readline().strip().split(',')
        for col in range(D):
            test_data2[row][col] = float(lines[col])

# 计算均值向量
ave = [0.0] * D
for i in range(N):
    for j in range(D):
        ave[j] = ave[j] + train_data[i][j]
for i in range(D):
    ave[i] = ave[i] / N
with open("ave.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(ave)
print(f"均值向量：{ave}")

# 计算中心化数据
center_data = train_data
for i in range(N):
    for j in range(D):
        center_data[i][j] = center_data[i][j]-ave[j]

# 计算协方差矩阵
cov = [[0.0] * D for _ in range(D)]
for i in range(D):
    for j in range(D):
        for k in range(N):
            cov[i][j] = cov[i][j] + center_data[k][i] * center_data[k][j]
        cov[i][j] = cov[i][j] / (N - 1)
with open('cov.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(cov)
print(f"协方差矩阵：{cov}")

# 求协方差矩阵的行列式
det_cov = 1.0
cov1 = [[0.0] * D for _ in range(D)]
for i in range(D):
    for j in range(D):
        cov1[i][j] = cov[i][j]
for i in range(D):
    if cov1[i][i] == 0:
        for k in range(i + 1, D):
            if cov1[k][i] != 0:
                cov1[i], cov1[k] = cov1[k], cov1[i]
                det_cov *= -1
                break
            else:
                det_cov = 0.0
    for j in range(i + 1, D):
        factor = cov1[j][i] / cov1[i][i]
        for k in range(i, D):
            cov1[j][k] -= factor * cov1[i][k]
for i in range(D):
    det_cov *= cov1[i][i]
print("协方差矩阵的行列式为：", det_cov)

# 构造增广矩阵
augmented_cov = [[0.0] * D * 2 for _ in range(D)]
for i in range(D):
    for j in range(D*2):
        if j < D:
            augmented_cov[i][j] = cov[i][j]
        else:
            if j - D == i:
                augmented_cov[i][j] = 1.0

# 改进行初等变换消元
for i in range(D):
    # 选择主元行
    max_row = i
    for k in range(i + 1, D):
        if abs(augmented_cov[k][i]) > abs(augmented_cov[max_row][i]):
            max_row = k
            # 交换行
            augmented_cov[i], augmented_cov[max_row] = augmented_cov[max_row], augmented_cov[i]
    diagonal = augmented_cov[i][i]
    for j in range(2*D):
        augmented_cov[i][j] /= diagonal
    for k in range(D):
        if k != i:
            factor = augmented_cov[k][i]
            for j in range(2*D):
                augmented_cov[k][j] -= factor * augmented_cov[i][j]

# 构造逆矩阵
inverse_cov = [[0.0] * D for _ in range(D)]
for i in range(D):
    for j in range(D):
        inverse_cov[i][j] = augmented_cov[i][j+D]
print(f"协方差逆矩阵：{inverse_cov}")

# 计算马氏距离
mahalanobis1 = [0.0] * N
for i in range(N):
    vec = test_data1[i]
    for n in range(D):
        vec[n] -= ave[n]
    for j in range(D):
        for k in range(D):
            mahalanobis1[i] += vec[j] * inverse_cov[j][k] * vec[k]
with open("mahalanobis1.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(mahalanobis1)
print(f"测试数据1马氏距离：{mahalanobis1}")

mahalanobis2 = [0.0] * N
for i in range(N):
    vec = test_data2[i]
    for n in range(D):
        vec[n] -= ave[n]
    for j in range(D):
        for k in range(D):
            mahalanobis2[i] += vec[j] * inverse_cov[j][k] * vec[k]
with open("mahalanobis2.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(mahalanobis2)
print(f"测试数据2马氏距离：{mahalanobis2}")

# 求概率密度值
pdd1 = [0.0] * N
for i in range(N):
    pdd1[i] = math.exp(mahalanobis1[i] / (-2)) / math.sqrt(((2 * pi) ** D) * det_cov)
with open("pdd1.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(pdd1)
print(f"测试数据1概率密度值：{pdd1}")
pdd2 = [0.0] * N
for i in range(N):
    pdd2[i] = math.exp(mahalanobis2[i] / (-2)) / math.sqrt(((2 * pi) ** D) * det_cov)
with open("pdd2.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(pdd2)
print(f"测试数据2概率密度值：{pdd2}")
pddc = math.exp(chi2.ppf(0.99, D) / (-2)) / math.sqrt(((2 * pi) ** D) * det_cov)
print(pddc)

# 计算卡方分布的99%区间上限
alpha = 0.01  # 99%置信区间
token1 = []
for i, distance in enumerate(mahalanobis1):
    if distance > chi2.ppf(0.99, D):
        token1.append(i)
print(token1)
token2 = []
for i, distance in enumerate(mahalanobis2):
    if distance > chi2.ppf(0.99, D):
        token2.append(i)
print(token2)

# 绘制折线图
# 创建 x 轴数据
x1 = list(range(len(mahalanobis1)))
# 绘制一般的点（蓝色）
plt.plot(x1, mahalanobis1, label="Mahalanobis Distance", color="blue", linewidth=1, marker="o")
# 绘制 token1 中的点（红色）
for idx in token1:
    plt.scatter(idx, mahalanobis1[idx], color="red", zorder=5)  # zorder 确保点在最上层
# 绘制红色虚线
plt.axhline(y=chi2.ppf(0.99, D), color="red", linestyle="--", label=f"Chi2 Critical Value (99%) = {chi2.ppf(0.99, D):.2f}")
# 添加图例
plt.legend()
# 添加标题和轴标签
plt.title("Mahalanobis Distance Plot1")
plt.xlabel("Index")
plt.ylabel("Mahalanobis Distance")
# 显示网格
plt.grid(True)
# 显示图形
plt.show()

# 创建 x 轴数据
x2 = list(range(len(mahalanobis2)))
# 绘制一般的点（蓝色）
plt.plot(x2, mahalanobis2, label="Mahalanobis Distance", color="blue", linewidth=1, marker="o")
# 绘制 token1 中的点（红色）
for idx in token2:
    plt.scatter(idx, mahalanobis2[idx], color="red", zorder=5)  # zorder 确保点在最上层
# 绘制红色虚线
plt.axhline(y=chi2.ppf(0.99, D), color="red", linestyle="--", label=f"Chi2 Critical Value (99%) = {chi2.ppf(0.99, D):.2f}")
# 添加图例
plt.legend()
# 添加标题和轴标签
plt.title("Mahalanobis Distance Plot2")
plt.xlabel("Index")
plt.ylabel("Mahalanobis Distance")
# 显示网格
plt.grid(True)
# 显示图形
plt.show()
