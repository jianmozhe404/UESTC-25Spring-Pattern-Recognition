import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, recall_score, precision_score
from scipy.stats import chi2

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号


# 读取检测结果
def load_results(file_path):
    return pd.read_csv(file_path)


# 绘制归一化概率密度的折线图
def plot_normalized_probability(results):
    plt.figure(figsize=(12, 6))
    plt.plot(results.index, results["Normalized_Probability"], label="归一化概率密度", color="orange", alpha=0.7)
    plt.xlabel("样本索引")
    plt.ylabel("归一化概率密度")
    plt.title("归一化概率密度变化")
    plt.legend()
    plt.show()


# 绘制检测结果折线图（马氏距离和异常点）
def plot_results(results, true_labels):
    plt.figure(figsize=(12, 6))
    plt.plot(results.index, results["Mahalanobis_Distance"], label="马氏距离", color="green", alpha=0.7)
    plt.scatter(results[results["Is_Anomaly"]].index, results[results["Is_Anomaly"]]["Mahalanobis_Distance"],
                color="red", label="检测到的异常", zorder=5)
    plt.axhline(y=chi2.ppf(0.95, df=31), color="red", linestyle="--", label="阈值 (95%)")
    plt.xlabel("样本索引")
    plt.ylabel("马氏距离")
    plt.yscale('log')  # 将纵坐标改为指数形式
    plt.title("异常检测结果")
    plt.legend()
    plt.show()


# 计算故障检测率、故障误报率和混淆矩阵
def evaluate_results(results, true_labels):
    predicted_labels = results["Is_Anomaly"].astype(int)
    cm = confusion_matrix(true_labels, predicted_labels)

    # 绘制混淆矩阵
    print("混淆矩阵：")
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["正常", "异常"], yticklabels=["正常", "异常"])
    plt.xlabel("预测值")
    plt.ylabel("真实值")
    plt.title("混淆矩阵")
    plt.show()

    # 计算故障检测率（召回率）和故障误报率（1 - 精确率）
    recall = recall_score(true_labels, predicted_labels)  # 故障检测率（召回率）
    precision = precision_score(true_labels, predicted_labels)  # 精确率
    print(f"故障检测率 (召回率): {recall:.4f}")
    print(f"故障误报率 (1 - 精确率): {1 - precision:.4f}")


# 主函数
if __name__ == "__main__":
    # 加载检测结果
    results = load_results("Result_data2.csv")

    # 假设真实结果是前一半数据正常，后一半数据异常
    half_length = len(results) // 2
    true_labels = np.concatenate([np.zeros(half_length), np.ones(len(results) - half_length)])

    # 绘制归一化概率密度的折线图
    plot_normalized_probability(results)

    # 绘制检测结果折线图（马氏距离和异常点）
    plot_results(results, true_labels)

    # 评估检测结果
    evaluate_results(results, true_labels)