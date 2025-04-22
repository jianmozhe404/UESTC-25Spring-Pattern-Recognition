import csv
import math
import numpy as np
from scipy.stats import chi2
import matplotlib.pyplot as plt

D = 31
N = 4000
wid = 25
pi = math.pi

# 读取train_data.csv, test_data1.csv, test_data2.csv文件
with open("test\Train_data.csv", "r", encoding="UTF-8") as f:
    train_data = [[0.0] * D for _ in range(N)]
    for row in range(N):
        lines = f.readline().strip().split(',')
        for col in range(D):
            train_data[row][col] = float(lines[col])
with open("test\Test_data1.csv", "r", encoding="UTF-8") as f:
    test_data1 = [[0.0] * D for _ in range(N)]
    for row in range(N):
        lines = f.readline().strip().split(',')
        for col in range(D):
            test_data1[row][col] = float(lines[col])
with open("test\Test_data2.csv", "r", encoding="UTF-8") as f:
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
center_data = [[0.0] * D for _ in range(N)]
for i in range(N):
    for j in range(D):
        center_data[i][j] = train_data[i][j]
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
mahalanobis = [0.0] * N
for i in range(N):
    vec = train_data[i]
    for n in range(D):
        vec[n] -= ave[n]
    for j in range(D):
        for k in range(D):
            mahalanobis[i] += vec[j] * inverse_cov[j][k] * vec[k]
with open("mahalanobis.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(mahalanobis)
print(f"训练数据马氏距离：{mahalanobis}")

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

# # 高斯窗口拟合train_data的PDF
# train_pdf = [0.0] * N
# for n in range(N):
#     for n0 in range(N):
#         sub_vec = [0.0] * D
#         md = 0.0
#         for i in range(D):
#             sub_vec[i] = train_data[n][i] - train_data[n0][i]
#         for j in range(D):
#             for k in range(D):
#                 md += sub_vec[j] * inverse_cov[j][k] * sub_vec[k]
#         train_pdf[n] += ((math.exp(md / (-2))) / math.sqrt(((2 * pi) ** D) * det_cov))
# print(f"train_data的pdf为：{train_pdf}")

train_data = np.array(train_data)
# 定义高斯窗函数
def gaussian_kernel(x, x_i, sigma):
    return (1 / ((2 * math.pi) ** (D / 2) * sigma ** D)) * math.exp(-0.5 * np.sum((x - x_i) ** 2) / sigma ** 2)

# 选择合适的带宽（需要根据数据调整）
sigma = 2  # 带宽参数

# 计算每个数据点的概率密度
pdf_values = np.zeros(N)
for i in range(N):
    x = train_data[i]  # 当前数据点
    for j in range(N):
        x_i = train_data[j]  # 训练数据中的样本
        pdf_values[i] += gaussian_kernel(x, x_i, sigma)
    pdf_values[i] /= N

# 输出结果
print("Estimated PDF values:", pdf_values)

# 将马氏距离从小到大排序，并获取对应的索引
sorted_indices = np.argsort(mahalanobis)
sorted_mahalanobis = np.array(mahalanobis)[sorted_indices]
sorted_pdf_values = np.array(pdf_values)[sorted_indices]

# 计算区间范围
min_mahalanobis = sorted_mahalanobis[0]
max_mahalanobis = sorted_mahalanobis[-1]
interval = (max_mahalanobis - min_mahalanobis) / wid

# 初始化区间范围和对应的概率密度和
interval_edges = np.linspace(min_mahalanobis, max_mahalanobis, wid+1)  # 20个区间需要21个边界
interval_pdf_sums = np.zeros(wid)  # 每个区间的概率密度值之和

# 遍历所有点，将它们分配到对应的区间，并累加概率密度值
for i in range(len(sorted_mahalanobis)):
    for j in range(wid):
        if interval_edges[j] <= sorted_mahalanobis[i] < interval_edges[j + 1]:
            interval_pdf_sums[j] += sorted_pdf_values[i]
            break

# 绘制新的曲线
plt.plot(interval_edges[:-1] + interval / 2, interval_pdf_sums, label="Total Probability Density", color="blue", linewidth=2, marker="o")
plt.title("Total Probability Density vs. Mahalanobis Distance Intervals")
plt.xlabel("Mahalanobis Distance Intervals")
plt.ylabel("Total Probability Density")
plt.grid(True)
plt.legend()
plt.show()

# 找到最大概率密度值
max_pdf = np.max(interval_pdf_sums)

# 初始化下标列表
index1 = []

# 遍历 mahalanobis1 中的每个马氏距离
for i, md in enumerate(mahalanobis1):
    # 查找马氏距离所在的区间
    interval_index = np.searchsorted(interval_edges, md) - 1
    if interval_index < 0 or interval_index >= wid:
        # 如果马氏距离不落在任何一个区间内，直接记录下标
        index1.append(i)
    else:
        # 记录该区间内的概率密度
        pdf = interval_pdf_sums[interval_index]
        # 判断条件
        if pdf < 0.01 * max_pdf:
            index1.append(i)

# 输出结果
print("满足条件的下标列表:", index1)

# 初始化下标列表
index2 = []

# 遍历 mahalanobis1 中的每个马氏距离
for i, md in enumerate(mahalanobis2):
    # 查找马氏距离所在的区间
    interval_index = np.searchsorted(interval_edges, md) - 1
    if interval_index < 0 or interval_index >= wid:
        # 如果马氏距离不落在任何一个区间内，直接记录下标
        index2.append(i)
    else:
        # 记录该区间内的概率密度
        pdf = interval_pdf_sums[interval_index]
        # 判断条件
        if pdf < 0.01 * max_pdf:
            index2.append(i)

# 输出结果
print("满足条件的下标列表:", index2)