import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers

# ====================== 1. 生成线性可分数据（2000个点/类） ======================
np.random.seed(42)

# 生成正类样本（类别 +1）
X_pos = np.random.randn(2000, 2) + np.array([3, 3])  # 均值 [3, 3]
y_pos = np.ones(2000)  # 标签 +1

# 生成负类样本（类别 -1）
X_neg = np.random.randn(2000, 2) + np.array([-3, -3])  # 均值 [-3, -3]
y_neg = -np.ones(2000)  # 标签 -1

# 合并数据
X = np.vstack([X_pos, X_neg])  # 特征矩阵 (4000, 2)
y = np.hstack([y_pos, y_neg])  # 标签 (4000,)

# ====================== 2. 手动实现SVM（硬间隔） ======================
def linear_kernel(X):
    """线性核函数"""
    return np.dot(X, X.T)

# 计算核矩阵
K = linear_kernel(X)

# 构建二次规划问题的参数
n_samples = X.shape[0]
P = np.outer(y, y) * K  # P_{i,j} = y_i y_j K(x_i, x_j)
q = -np.ones(n_samples)  # q = [-1, -1, ..., -1]

# 转换为 cvxopt 的矩阵格式
P = matrix(P.astype(np.float64))
q = matrix(q.astype(np.float64))

# 不等式约束 (alpha_i >= 0)
G = matrix(-np.eye(n_samples))  # G = -I, h = 0
h = matrix(np.zeros(n_samples))

# 等式约束 (sum alpha_i y_i = 0)
A = matrix(y.reshape(1, -1).astype(np.float64))  # A = y^T
b = matrix(0.0)  # b = 0

# 求解二次规划
solution = solvers.qp(P, q, G, h, A, b)
alphas = np.array(solution['x']).flatten()  # 拉格朗日乘子 alpha

# 提取支持向量（alpha > 1e-5 视为支持向量）
support_vector_indices = alphas > 1e-5
support_vectors = X[support_vector_indices]
support_vector_labels = y[support_vector_indices]
support_vector_alphas = alphas[support_vector_indices]

# 计算权重 w = sum alpha_i y_i x_i
w = np.sum(support_vector_alphas * support_vector_labels * support_vectors.T, axis=1)

# 计算偏置 b（取所有支持向量的平均值）
b = np.mean(support_vector_labels - np.dot(support_vectors, w))

# ====================== 3. 预测函数 ======================
def predict(X_new):
    """预测新样本的类别"""
    return np.sign(np.dot(X_new, w) + b)

# ====================== 4. 可视化（绘制全部4000个点） ======================
plt.figure(figsize=(10, 8))

# 绘制所有数据点（4000个）
plt.scatter(X[:2000, 0], X[:2000, 1], c='blue', alpha=0.5, edgecolors='k', label='Class +1')
plt.scatter(X[2000:, 0], X[2000:, 1], c='red', alpha=0.5, edgecolors='k', label='Class -1')

# 绘制支持向量
plt.scatter(
    support_vectors[:, 0],
    support_vectors[:, 1],
    s=100, facecolors='none', edgecolors='black', linewidths=1.5, label='Support Vectors'
)

# 绘制决策边界 w^T x + b = 0
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
Z = predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

# 绘制决策边界和间隔
plt.contour(xx, yy, Z, colors='black', levels=[-1, 0, 1], linestyles=['--', '-', '--'], linewidths=2)
plt.xlabel("Feature 1", fontsize=12)
plt.ylabel("Feature 2", fontsize=12)
plt.title("Manual SVM Implementation (Hard Margin) - All 4000 Points", fontsize=14)
plt.legend(loc='best')
plt.grid(True, linestyle='--', alpha=0.3)
plt.show()