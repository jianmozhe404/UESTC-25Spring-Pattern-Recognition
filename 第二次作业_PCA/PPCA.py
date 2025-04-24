# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
train_data = pd.read_csv('data\data\Train_data.csv')
test_data1 = pd.read_csv('test\Test_data1.csv')
test_data2 = pd.read_csv('test\Test_data2.csv')
print(f"{train_data.shape} {test_data1.shape} {test_data2.shape}")

# %% [markdown]
# #### 数据预处理
# 
# + 对train_data进行标准化
# + 同样的参数应用到test_data

# %%
from sklearn.preprocessing import StandardScaler

# 训练集标准化
scaler = StandardScaler()
standard_train_data = scaler.fit_transform(train_data)  # 计算均值、标准差
standard_test_data1 = scaler.transform(test_data1)       # 使用训练集的均值、标准差
standard_test_data2 = scaler.transform(test_data2)

# %% [markdown]
# #### PPCA
# 
# **重要的假设**:
# + 潜在变量$z$服从标准高斯分布
# + $x|z$ 是多元高斯分布---训练数据符合这一特征

# %%
n_samples, n_features = standard_train_data.shape
# 样本协方差矩阵
S = (standard_train_data.T @ standard_train_data) / n_samples
# 对样本进行特征分解
def ppca_fit(S,d=10):
    """ 对S矩阵进行特征分解, 提取前d个特征 """
    eigenvalues, eigenvectors = np.linalg.eigh(S) #默认是升序排列的
    #print(eigenvalues[::-1])
    #print(eigenvectors[:,::-1])
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:,::-1]
    sigma_sq = np.mean(eigenvalues[d:])
    print(sigma_sq)
    lambda_d = np.diag(eigenvalues[:d] - sigma_sq)
    U_d = eigenvectors[:, :d]
    W = U_d @ np.sqrt(lambda_d)
    #print(W)
    return W,sigma_sq,eigenvectors, eigenvalues
d = 15
W, sigma_sq, U_d, eigenvalues = ppca_fit(S, d)
print(f"噪声方差σ² = {sigma_sq:.4f}")

# %% [markdown]
# #### 异常检测:
# 
# **核心思想**:
# PPCA假设正常数据服从一个由​**低维潜在变量**和 **高斯噪声**​共同生成的高斯分布。异常检测的目标是：​找到偏离该分布的样本，即在该分布下概率密度极低的点。
# 
# + ​关键假设：正常数据集中分布在由主成分（W）张成的低维子空间附近，并受到方差为 $σ^2$的高斯噪声干扰。
# + ​异常定义：若某个测试样本的生成概率显著低于正常数据，则判定为异常。

# %%
def compute_nll(x_test,W,sigma_sq):
    """
    计算PPCA的对数似然
    """
    D = W.shape[0]
    d = W.shape[1]
    C = W @ W.T + sigma_sq * np.eye(D)

    inv_C = np.linalg.inv(C)
    log_det_C = np.log(np.linalg.det(C))
    
    #计算马氏距离的平方项
    mahalanobis_term = x_test.T @ inv_C @ x_test

    nll = 0.5 * (D * np.log(2*np.pi) + log_det_C + mahalanobis_term)

    return nll

test_scores1 = [compute_nll(x,W,sigma_sq) for x in standard_test_data1]
test_scores2 = [compute_nll(x,W,sigma_sq) for x in standard_test_data2]
train_scores = [compute_nll(x,W,sigma_sq) for x in standard_train_data]




# %%
threshold = np.percentile(train_scores,80)
test_anomalies1 = test_data1[test_scores1 > threshold]
test1_idx = test_data1.index[test_scores1 > threshold]
print(f"异常样本数：{len(test_anomalies1)}")
print(f"异常索引:{test1_idx[:]}")

# %%
test_anomalies2 = test_data2[test_scores2 > threshold]
test2_idx = test_data2.index[test_scores2 > threshold]
print(f"异常样本数：{len(test_anomalies2)}")
print(f"异常索引:{test2_idx[:250]}")

# %% [markdown]
# ### 结果可视化

# %%



