# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 18:20:03 2025

@author: 王晓琦
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cvxopt as cp
import seaborn as sns

# 读取数据
train_data = pd.read_csv("Train_data.csv")
test_data1 = pd.read_csv("Test_data1.csv")
test_data2 = pd.read_csv("Test_data2.csv")

# 得到数据的行数和列数
n = train_data.shape[0]  # 数据的样本数
m = train_data.shape[1]  # 数据的维度

# 数据标准化
mean = np.mean(train_data,axis = 0) # 按列取均值
std = np.std(train_data,axis = 0) #按列取方差
norm_train = (train_data - mean)/std
norm_test1 = (test_data1 - mean)/std
norm_test2 = (test_data2 - mean)/std

# 高斯核矩阵
r = 0.05
def RBF(xi,xj,r):
    xi = np.array(xi)
    xj = np.array(xj)
    dist = np.sum(xi**2, axis=1).reshape(-1, 1) + np.sum(xj**2, axis=1) - 2 * xi @ xj.T
    return np.exp(-r * dist)

# 计算训练数据对应的RBF核矩阵
K = RBF(norm_train,norm_train,r)
print(np.shape(K))

# cvxopt解决凸优化问题，得到α
def calcu_alpha(K, C):
    P = cp.matrix(K)
    q = cp.matrix(np.zeros((n,1)))  
    G = cp.matrix(np.vstack((-np.eye(n), np.eye(n)))) 
    h = cp.matrix(np.vstack((np.zeros((n, 1)), C * np.ones((n, 1))))) 
    A = cp.matrix(np.ones((1, n)))  
    b = cp.matrix(1.0)
    solution = cp.solvers.qp(P, q, G, h, A, b)
    alpha = np.array(solution['x']).flatten()  # 展平为一维数组
    return alpha

# 计算训练数据对应的拉格朗日乘子
C = 0.01 # 1/(mu*4000)
alpha = calcu_alpha(K, C)

# 针对所有满足不等式约束的alpha，逐个计算偏离值，取平均值作为最终结果
def calcu_b(alpha, xtrain, C, K):
    index_sv = np.where(alpha > 1e-5)[0]  
    alpha_sv = alpha[index_sv]
    index_b = np.where((alpha > 1e-5) & (alpha < C))[0]  
    if len(index_b)>0:    # 若存在满足0<α<C的支持向量，计算偏离距离b，取平均值
        b_ = [alpha_sv @ K[k, index_sv] 
             for k in index_b]
        b = sum(b_)/len(b_)  # 取偏离距离b平均值
    return b

# 计算训练模型中的偏离距离b
b = calcu_b(alpha, norm_train, C, K)
print(b)

#计算支持向量
index_sv = np.where(alpha > 1e-5)[0]  
alpha_sv = alpha[index_sv]
svecs = norm_train.iloc[index_sv,:]

#求得分类决策函数
def dec_function(x_test,svecs,alpha_sv,b,r):
    K_test = RBF(x_test,svecs, r)
    f =  K_test @ alpha_sv - b 
    return f

#计算决策值
f= dec_function(norm_train,svecs,alpha_sv,b,r)
f1 = dec_function(norm_test1,svecs,alpha_sv,b,r)
f2 = dec_function(norm_test2,svecs,alpha_sv,b,r)

#异常点数量
anomalies1 = np.where(f1 < 0, -1, 1)# 正常 +1 / 异常 -1
anomalies2 = np.where(f2 < 0, -1, 1)
anomalies1_in_first_2000 = np.sum(anomalies1[:2000] == -1)
anomalies1_in_last_2000 = np.sum(anomalies1[-2000:] == -1)
anomalies2_in_first_2000 = np.sum(anomalies2[:2000] == -1)
anomalies2_in_last_2000 = np.sum(anomalies2[-2000:] == -1)
print('接下来是对test1数据的检测结果:')
print(f"前 2000 个数据中被标记为异常的数量: {anomalies1_in_first_2000}/2000")
print(f"后 2000 个数据中被标记为异常的数量: {anomalies1_in_last_2000}/2000")
print('接下来是对test2数据的检测结果:')
print(f"前 2000 个数据中被标记为异常的数量: {anomalies2_in_first_2000}/2000")
print(f"后 2000 个数据中被标记为异常的数量: {anomalies2_in_last_2000}/2000")

# 训练数据可视化
plt.figure(figsize=(12, 6))
sns.lineplot(x = train_data.index, y = f, label = 'train')
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Train_data")
plt.ylabel("f")
plt.title("train")
plt.show()


# 测试数据1可视化
plt.figure(figsize=(12, 6))
sns.lineplot(x = test_data1.index, y = f1, label = 'test1')
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("test_data1")
plt.ylabel("f1")
plt.title("test1")
plt.legend()


# 测试数据2可视化
plt.figure(figsize=(12, 6))
sns.lineplot(x = test_data2.index, y = f2, label = 'test2')
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("test_data2")
plt.ylabel("f2")
plt.title("test2")
plt.legend()

    
    
    
    
    
    
    
    