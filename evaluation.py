import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from collections import deque # predict 函数内部需要

# 从 model_utils 导入所需的类和函数
from model_utils import LSTM, calculate_threshold, predict

# 解决matplotlib中文显示问题
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

def main():
    # %% 数据加载与预处理 (需要训练数据来拟合scaler和计算阈值)
    train_file_path = 'd:/lecture/25spring/模式识别/作业1/test/Train_data.csv'
    test1_file_path = 'd:/lecture/25spring/模式识别/作业1/test/Test_data1.csv'
    test2_file_path = 'd:/lecture/25spring/模式识别/作业1/test/Test_data2.csv'

    train_data_df = pd.read_csv(train_file_path)
    testdata1_df = pd.read_csv(test1_file_path)
    testdata2_df = pd.read_csv(test2_file_path)

    train_data_np = train_data_df.values
    testdata1_np = testdata1_df.values
    testdata2_np = testdata2_df.values

    # 使用在训练数据上拟合的 scaler
    scaler = StandardScaler()
    train_data_scaled = scaler.fit_transform(train_data_np)
    testdata1_scaled = scaler.transform(testdata1_np)
    testdata2_scaled = scaler.transform(testdata2_np)

    # %% 参数设置 (需要与训练时一致)
    input_size = train_data_scaled.shape[1]
    hidden_size = 256 # 保持与训练时一致
    num_layers = 3    # 保持与训练时一致
    window_size = 200 # 保持与训练时一致
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_load_path = 'd:/lecture/25spring/模式识别/作业1/lstm_best_model.pth' # 加载训练好的模型

    print(f"使用设备: {device}")
    print(f"输入维度: {input_size}")

    # 将 numpy 数组转换为 PyTorch 张量
    train_data_tensor = torch.tensor(train_data_scaled, dtype=torch.float32)
    testdata1_tensor = torch.tensor(testdata1_scaled, dtype=torch.float32)
    testdata2_tensor = torch.tensor(testdata2_scaled, dtype=torch.float32)

    # %% 加载模型
    model = LSTM(input_size, hidden_size, device, num_layers)
    try:
        model.load_state_dict(torch.load(model_load_path, map_location=device))
        print(f"模型状态已从 {model_load_path} 加载")
    except FileNotFoundError:
        print(f"错误: 找不到模型文件 {model_load_path}。请先运行 train.py 进行训练。")
        return
    except Exception as e:
        print(f"加载模型时出错: {e}")
        return

    model.to(device) # 确保模型在正确的设备上
    criterion = torch.nn.MSELoss() # 评估时也需要损失函数来计算误差

    # %% 计算阈值 (使用训练数据)
    print("开始计算阈值...")
    threshold, train_errors = calculate_threshold(model, train_data_tensor, criterion, window_size, device)

    if threshold is None:
        print("无法计算阈值，评估中止。")
        return

    # 可视化训练误差分布
    plt.figure(figsize=(10, 6))
    plt.hist(train_errors, bins=50, alpha=0.7, label='训练误差')
    plt.axvline(threshold, color='r', linestyle='--', label=f'阈值 ({threshold:.6f})')
    plt.legend(prop={'size': 12})
    plt.title('训练数据误差分布与阈值', fontsize=14)
    plt.xlabel('误差值', fontsize=12)
    plt.ylabel('频次', fontsize=12)
    plt.savefig('d:/lecture/25spring/模式识别/作业1/threshold_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

    # %% 评估模型
    model.eval() # 确保模型处于评估模式

    # 测试数据集1
    print("\n测试数据集1的结果:")
    errors1, result1 = predict(model, testdata1_tensor, criterion, threshold, window_size, device)
    # 注意：predict 返回的 result 长度与 testdata1_tensor 相同，errors 长度为 len(testdata1_tensor) - window_size
    # 计算异常点数时，应考虑 result 的总长度
    normal_count1 = result1.count(True)
    anomaly_count1 = len(result1) - normal_count1
    print(f"总数据点: {len(result1)}")
    print(f"正常数据点: {normal_count1}, 异常数据点: {anomaly_count1}")
    if len(result1) > 0:
        print(f"异常比例: {anomaly_count1/len(result1)*100:.2f}%")
    else:
        print("无数据点可计算比例")


    # 测试数据集2
    print("\n测试数据集2的结果:")
    errors2, result2 = predict(model, testdata2_tensor, criterion, threshold, window_size, device)
    normal_count2 = result2.count(True)
    anomaly_count2 = len(result2) - normal_count2
    print(f"总数据点: {len(result2)}")
    print(f"正常数据点: {normal_count2}, 异常数据点: {anomaly_count2}")
    if len(result2) > 0:
        print(f"异常比例: {anomaly_count2/len(result2)*100:.2f}%")
    else:
        print("无数据点可计算比例")


    # %% 可视化评估结果

    # 可视化两个测试数据集的异常检测结果
    plt.figure(figsize=(15, 10))
    feature_idx = 0  # 选择一个特征进行可视化

    # 数据集1的可视化
    plt.subplot(2, 1, 1)
    plt.plot(testdata1_scaled[:, feature_idx], label='测试数据1 (标准化)') # 使用标准化后的数据绘图
    anomaly_indices1 = [i for i, r in enumerate(result1) if not r]
    if anomaly_indices1:
        plt.scatter(anomaly_indices1, testdata1_scaled[anomaly_indices1, feature_idx],
                    color='red', label='异常点', s=10) # 调小点的大小以便观察
    plt.legend(prop={'size': 12})
    plt.title('测试数据集1异常检测结果', fontsize=16)
    plt.xlabel('时间步', fontsize=14)
    plt.ylabel(f'特征 {feature_idx} (标准化)', fontsize=14)

    # 数据集2的可视化
    plt.subplot(2, 1, 2)
    plt.plot(testdata2_scaled[:, feature_idx], label='测试数据2 (标准化)')
    anomaly_indices2 = [i for i, r in enumerate(result2) if not r]
    if anomaly_indices2:
        plt.scatter(anomaly_indices2, testdata2_scaled[anomaly_indices2, feature_idx],
                    color='red', label='异常点', s=10)
    plt.legend(prop={'size': 12})
    plt.title('测试数据集2异常检测结果', fontsize=16)
    plt.xlabel('时间步', fontsize=14)
    plt.ylabel(f'特征 {feature_idx} (标准化)', fontsize=14)

    plt.tight_layout()
    plt.savefig('d:/lecture/25spring/模式识别/作业1/both_datasets_anomaly_detection.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 比较两个数据集的异常比例
    if len(result1) > 0 and len(result2) > 0:
        labels = ['测试数据集1', '测试数据集2']
        normal_percentages = [normal_count1/len(result1)*100, normal_count2/len(result2)*100]
        anomaly_percentages = [anomaly_count1/len(result1)*100, anomaly_count2/len(result2)*100]

        plt.figure(figsize=(10, 6))
        x = np.arange(len(labels))
        width = 0.35

        plt.bar(x - width/2, normal_percentages, width, label='正常数据')
        plt.bar(x + width/2, anomaly_percentages, width, label='异常数据')

        plt.ylabel('百分比 (%)', fontsize=14)
        plt.title('两个测试数据集的异常检测比例对比')