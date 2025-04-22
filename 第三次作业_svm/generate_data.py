import numpy as np
import pandas as pd

# 设置随机种子以确保结果可重现
np.random.seed(42)

# 定义总样本数和类别数
total_samples = 5000
n_classes = 3
samples_per_class = total_samples // n_classes

# 定义三个类别的均值和方差
class_params = {
    'class1': {'mean': [0, 0], 'cov': [[1, 0], [0, 1]]},
    'class2': {'mean': [5, 5], 'cov': [[2, 0], [0, 2]]},
    'class3': {'mean': [-5, 5], 'cov': [[1.5, 0], [0, 1.5]]}
}

# 生成训练数据
data = []
labels = []

for class_idx, (class_name, params) in enumerate(class_params.items()):
    # 为每个类别生成数据
    class_data = np.random.multivariate_normal(
        mean=params['mean'],
        cov=params['cov'],
        size=samples_per_class
    )
    data.append(class_data)
    labels.extend([class_idx] * samples_per_class)

# 合并所有数据
X = np.vstack(data)
y = np.array(labels)

# 创建DataFrame并保存到CSV文件
df = pd.DataFrame(X, columns=['feature1', 'feature2'])
df['label'] = y

# 保存训练数据
df.to_csv('train_data.csv', index=False)

print(f"训练数据已生成并保存到 train_data.csv")
print(f"总样本数: {len(df)}")
print(f"类别分布:")
print(df['label'].value_counts())

# ====================== 生成测试数据 ======================
# 更新随机种子，确保测试数据与训练数据不同
np.random.seed(43)

# 定义测试集总样本数和每个类别的样本数
test_total_samples = 2000
test_samples_per_class = test_total_samples // n_classes

# 生成测试数据
test_data = []
test_labels = []

for class_idx, (class_name, params) in enumerate(class_params.items()):
    # 为每个类别生成测试数据
    class_data = np.random.multivariate_normal(
        mean=params['mean'],
        cov=params['cov'],
        size=test_samples_per_class
    )
    test_data.append(class_data)
    test_labels.extend([class_idx] * test_samples_per_class)

# 合并所有测试数据
X_test = np.vstack(test_data)
y_test = np.array(test_labels)

# 创建DataFrame并保存到CSV文件
test_df = pd.DataFrame(X_test, columns=['feature1', 'feature2'])
test_df['label'] = y_test

# 保存测试数据
test_df.to_csv('test_data.csv', index=False)

print(f"\n测试数据已生成并保存到 test_data.csv")
print(f"测试样本数: {len(test_df)}")
print(f"测试集类别分布:")
print(test_df['label'].value_counts()) 