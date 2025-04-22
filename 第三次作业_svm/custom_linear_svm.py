import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib as mpl
from matplotlib.font_manager import FontProperties
import os

# 设置字体路径 - 尝试查找Windows系统的中文字体
font_path = None
possible_font_paths = [
    r'C:\Windows\Fonts\simhei.ttf',  # 黑体
    r'C:\Windows\Fonts\simsun.ttc',   # 宋体
    r'C:\Windows\Fonts\msyh.ttc',     # 微软雅黑
    r'C:\Windows\Fonts\simkai.ttf'    # 楷体
]

for path in possible_font_paths:
    if os.path.exists(path):
        font_path = path
        print(f"找到中文字体: {path}")
        break

if font_path:
    chinese_font = FontProperties(fname=font_path)
else:
    print("警告: 未找到系统中文字体文件，将使用默认字体")
    chinese_font = FontProperties()

# 自定义线性SVM类
class LinearSVM:
    def __init__(self, learning_rate=0.01, lambda_param=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iterations = n_iterations
        self.w = None
        self.b = None
        
    def _init_weights_bias(self, X):
        # 初始化权重和偏置
        n_features = X.shape[1]
        self.w = np.zeros(n_features)
        self.b = 0
    
    def _hinge_loss(self, X, y):
        # 计算铰链损失 max(0, 1 - y * (w·x + b))
        n_samples = X.shape[0]
        distances = 1 - y * (np.dot(X, self.w) + self.b)
        # 不考虑负距离（正确分类且距离足够远的样本）
        distances = np.maximum(0, distances)
        # 铰链损失
        hinge_loss = self.lambda_param * (np.sum(self.w ** 2)) + np.sum(distances) / n_samples
        return hinge_loss
    
    def _gradient_descent(self, X, y):
        n_samples = X.shape[0]
        
        # 对于所有样本，计算dw和db
        for i in range(n_samples):
            distance = 1 - y[i] * (np.dot(X[i], self.w) + self.b)
            
            if distance <= 0:  # 正确分类且边距足够
                dw = 2 * self.lambda_param * self.w
                db = 0
            else:  # 错误分类或边距不足
                dw = 2 * self.lambda_param * self.w - y[i] * X[i]
                db = -y[i]
            
            # 更新权重和偏置
            self.w -= self.lr * dw
            self.b -= self.lr * db
    
    def fit(self, X, y):
        # 初始化权重和偏置
        self._init_weights_bias(X)
        
        # 将标签转换为-1和1（如果不是的话）
        y_ = np.where(y <= 0, -1, 1)
        
        # 梯度下降
        for _ in range(self.n_iterations):
            self._gradient_descent(X, y_)
        
        return self
    
    def predict(self, X):
        # 计算决策函数
        decision = np.dot(X, self.w) + self.b
        # 根据决策函数值返回预测类别
        y_pred = np.where(decision <= 0, -1, 1)
        return y_pred

# 实现一对一分类器
class OneVsOneLinearSVM:
    def __init__(self, learning_rate=0.01, lambda_param=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iterations = n_iterations
        self.classifiers = []
        self.class_pairs = []
        
    def fit(self, X, y):
        # 获取所有唯一类别
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        
        # 为每对类别创建和训练一个二分类器
        for i in range(n_classes):
            for j in range(i+1, n_classes):
                # 获取当前两个类别
                class_i, class_j = self.classes[i], self.classes[j]
                
                # 筛选出这两个类别的数据
                mask = np.logical_or(y == class_i, y == class_j)
                X_pair = X[mask]
                y_pair = y[mask]
                
                # 将标签转换为二分类标签（-1和1）
                y_binary = np.where(y_pair == class_i, -1, 1)
                
                # 创建并训练分类器
                clf = LinearSVM(
                    learning_rate=self.lr, 
                    lambda_param=self.lambda_param, 
                    n_iterations=self.n_iterations
                )
                clf.fit(X_pair, y_binary)
                
                # 保存分类器和对应的类别对
                self.classifiers.append(clf)
                self.class_pairs.append((class_i, class_j))
        
        return self
    
    def predict(self, X):
        n_samples = X.shape[0]
        n_classes = len(self.classes)
        
        # 投票矩阵
        votes = np.zeros((n_samples, n_classes))
        
        # 对每个二分类器进行预测和投票
        for clf, (class_i, class_j) in zip(self.classifiers, self.class_pairs):
            predictions = clf.predict(X)
            
            # 根据预测结果投票
            for k in range(n_samples):
                if predictions[k] == -1:  # 预测为class_i
                    idx_i = np.where(self.classes == class_i)[0][0]
                    votes[k, idx_i] += 1
                else:  # 预测为class_j
                    idx_j = np.where(self.classes == class_j)[0][0]
                    votes[k, idx_j] += 1
        
        # 返回得票最多的类别
        return self.classes[np.argmax(votes, axis=1)]

# 读取数据
print("正在读取数据...")
df = pd.read_csv('train_data.csv')
X = df[['feature1', 'feature2']].values
y = df['label'].values

# 数据归一化
print("正在进行数据归一化...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
print("正在划分训练集和测试集...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

# 创建并训练自定义一对一线性SVM模型
print("正在训练自定义一对一线性SVM模型...")
ovo_svm = OneVsOneLinearSVM(learning_rate=0.01, lambda_param=0.001, n_iterations=1000)
ovo_svm.fit(X_train, y_train)

# 在测试集上评估模型
print("正在评估模型...")
y_pred = ovo_svm.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print(f"模型在拆分测试集上的准确率: {accuracy:.4f}")

# 计算混淆矩阵
def confusion_matrix(y_true, y_pred, classes):
    n_classes = len(classes)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for i in range(len(y_true)):
        true_idx = np.where(classes == y_true[i])[0][0]
        pred_idx = np.where(classes == y_pred[i])[0][0]
        cm[true_idx, pred_idx] += 1
    return cm

# 打印混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred, ovo_svm.classes)
print("混淆矩阵:")
print(conf_matrix)

# 可视化决策边界
print("正在可视化决策边界...")
def plot_decision_boundary(X, y, model, scaler):
    plt.figure(figsize=(12, 10))
    
    # 设置网格范围
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    # 预测网格点的类别
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_scaled = scaler.transform(grid)
    Z = model.predict(grid_scaled)
    Z = Z.reshape(xx.shape)
    
    # 绘制决策边界
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
    plt.contour(xx, yy, Z, colors='k', linewidths=0.5, antialiased=True)
    
    # 绘制散点图
    colors = ['#FF9999', '#66B2FF', '#99FF99']
    labels = ['类别一', '类别二', '类别三']
    for i, cls in enumerate(model.classes):
        mask = y == cls
        plt.scatter(X[mask, 0], X[mask, 1], c=colors[i], 
                    label=labels[i], edgecolors='k', alpha=0.7)
    
    # 添加标题和标签
    plt.title('一对一线性SVM决策边界可视化', fontsize=14, fontproperties=chinese_font)
    plt.xlabel('特征一', fontsize=12, fontproperties=chinese_font)
    plt.ylabel('特征二', fontsize=12, fontproperties=chinese_font)
    
    # 设置图例
    plt.legend(prop=chinese_font, fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 保存图像
    plt.savefig('一对一线性决策边界.png', dpi=300, bbox_inches='tight')
    plt.show()

# 调用函数绘制决策边界
plot_decision_boundary(X, y, ovo_svm, scaler)

# 分析和可视化每个二分类器
print("\n分析自定义一对一分类器的二分类器：")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 定义颜色和标签
colors = ['#FF9999', '#66B2FF', '#99FF99']
labels = ['类别一', '类别二', '类别三']

# 使用原始数据（非标准化）绘制分类器
for i, ((class_i, class_j), clf) in enumerate(zip(ovo_svm.class_pairs, ovo_svm.classifiers)):
    ax = axes[i]
    
    # 获取这两个类别的数据
    mask = np.logical_or(y == class_i, y == class_j)
    X_pair = X[mask]
    y_pair = y[mask]
    
    # 设置网格范围
    x_min, x_max = X_pair[:, 0].min() - 1, X_pair[:, 0].max() + 1
    y_min, y_max = X_pair[:, 1].min() - 1, X_pair[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                        np.arange(y_min, y_max, 0.1))
    
    # 预测网格点的类别
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_scaled = scaler.transform(grid)
    Z = clf.predict(grid_scaled)
    Z = Z.reshape(xx.shape)
    
    # 绘制决策边界
    ax.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdBu)
    
    # 绘制决策线
    ax.contour(xx, yy, Z, colors='k', linewidths=1.0)
    
    # 绘制两个类别的数据点
    idx_i = np.where(ovo_svm.classes == class_i)[0][0]
    idx_j = np.where(ovo_svm.classes == class_j)[0][0]
    
    for cls, color, label in zip([class_i, class_j], 
                                [colors[idx_i], colors[idx_j]],
                                [labels[idx_i], labels[idx_j]]):
        class_mask = y_pair == cls
        ax.scatter(X_pair[class_mask, 0], X_pair[class_mask, 1], 
                  c=color, label=label, edgecolors='k', alpha=0.7)
    
    # 绘制决策边界方程
    w = clf.w
    b = clf.b
    
    # 转换回原始特征空间（非标准化的权重）
    w_orig = w / scaler.scale_
    b_orig = b - np.sum(w * scaler.mean_ / scaler.scale_)
    
    decision_eq = f"决策边界: {w_orig[0]:.4f}*x1 + {w_orig[1]:.4f}*x2 + {b_orig:.4f} = 0"
    ax.set_title(f'{labels[idx_i]} vs {labels[idx_j]}\n{decision_eq}', 
                fontproperties=chinese_font, fontsize=10)
    ax.set_xlabel('特征一', fontproperties=chinese_font)
    ax.set_ylabel('特征二', fontproperties=chinese_font)
    ax.legend(prop=chinese_font)
    ax.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('分别训练的二分类器.png', dpi=300, bbox_inches='tight')
plt.show()




# =================== 在外部测试集上评估模型 ===================
print("\n=================== 在外部测试集上评估模型 ===================")
print("正在读取外部测试数据...")
test_df = pd.read_csv('test_data.csv')
X_external = test_df[['feature1', 'feature2']].values
y_external = test_df['label'].values

# 使用训练数据的归一化器对测试数据进行标准化
print("对测试数据进行归一化...")
X_external_scaled = scaler.transform(X_external)

# 使用模型进行预测
print("在外部测试集上进行预测...")
y_external_pred = ovo_svm.predict(X_external_scaled)

# 计算准确率
external_accuracy = np.mean(y_external_pred == y_external)
print(f"模型在外部测试集上的准确率: {external_accuracy:.4f}")

# 计算并打印混淆矩阵
external_conf_matrix = confusion_matrix(y_external, y_external_pred, ovo_svm.classes)
print("外部测试集上的混淆矩阵:")
print(external_conf_matrix)

# 计算每个类别的准确率
print("\n每个类别的准确率:")
for i, cls in enumerate(ovo_svm.classes):
    mask = y_external == cls
    class_accuracy = np.mean(y_external_pred[mask] == y_external[mask])
    print(f"类别{i+1}的准确率: {class_accuracy:.4f}")

# 可视化外部测试数据和决策边界
plt.figure(figsize=(12, 10))
    
# 设置网格范围
x_min, x_max = X_external[:, 0].min() - 1, X_external[:, 0].max() + 1
y_min, y_max = X_external[:, 1].min() - 1, X_external[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                    np.arange(y_min, y_max, 0.1))

# 预测网格点的类别
grid = np.c_[xx.ravel(), yy.ravel()]
grid_scaled = scaler.transform(grid)
Z = ovo_svm.predict(grid_scaled)
Z = Z.reshape(xx.shape)

# 绘制决策边界
plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
plt.contour(xx, yy, Z, colors='k', linewidths=0.5, antialiased=True)

# 绘制测试数据点
for i, cls in enumerate(ovo_svm.classes):
    mask = y_external == cls
    plt.scatter(X_external[mask, 0], X_external[mask, 1], c=colors[i], 
                label=labels[i], edgecolors='k', alpha=0.7)

# 标记错误分类的点
for i in range(len(y_external)):
    if y_external[i] != y_external_pred[i]:
        plt.scatter(X_external[i, 0], X_external[i, 1], s=100, 
                    facecolors='none', edgecolors='red', linewidth=1.5)

# 添加标题和标签
plt.title('外部测试数据的分类结果和决策边界', fontsize=14, fontproperties=chinese_font)
plt.xlabel('特征一', fontsize=12, fontproperties=chinese_font)
plt.ylabel('特征二', fontsize=12, fontproperties=chinese_font)

# 设置图例
plt.legend(prop=chinese_font, fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)

# 保存图像
plt.savefig('外部测试数据和决策边界.png', dpi=300, bbox_inches='tight')
plt.show() 