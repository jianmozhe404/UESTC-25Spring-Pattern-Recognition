import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from collections import deque # predict 函数内部需要

# 从 model_utils 导入所需的类和函数
from model_utils import LSTMAutoencoder,AutoencoderDataset,predict_autoencoder,calculate_autoencoder_threshold
from model_utils import plot_confusion_matrix, calculate_metrics
# 解决matplotlib中文显示问题
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

def main():
    
    train_file_path = 'test/Train_data.csv'
    test1_file_path = 'test/Test_data1.csv'
    test2_file_path = 'test/Test_data2.csv'

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

    input_size = train_data_scaled.shape[1]
    hidden_size = 64 # 保持与训练时一致
    num_layers = 2    # 保持与训练时一致
    window_size = 100 # 保持与训练时一致
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_load_path = 'save_model/autoencoder_best_model.pth' # 加载训练好的模型

    print(f"使用设备: {device}")
    print(f"输入维度: {input_size}")

    # 将 numpy 数组转换为 PyTorch 张量
    train_data = torch.tensor(train_data_scaled, dtype=torch.float32)
    testdata1 = torch.tensor(testdata1_scaled, dtype=torch.float32)
    testdata2 = torch.tensor(testdata2_scaled, dtype=torch.float32)

    model = LSTMAutoencoder(input_size, hidden_size, device, num_layers)
    try:
        model.load_state_dict(torch.load(model_load_path, map_location=device))
        print(f"模型状态已从 {model_load_path} 加载")
    except FileNotFoundError:
        print(f"错误: 找不到模型文件 {model_load_path}。请先运行 train.py 进行训练。")
        return
    except Exception as e:
        print(f"加载模型时出错: {e}")
        return

    model.to(device) 
    criterion = torch.nn.MSELoss() 

    # %% 计算阈值 (使用训练数据)
    print("开始计算阈值...")
    threshold, train_errors = calculate_autoencoder_threshold(model, train_data, criterion, window_size, device)

    if threshold is None:
        print("无法计算阈值，评估中止。")
        return

    # 可视化训练误差分布
    plt.figure(figsize=(10, 6))
    plt.hist(train_errors, bins=50, alpha=0.7, label='训练重构误差')
    plt.axvline(threshold, color='r', linestyle='--', label=f'阈值 ({threshold:.6f})')
    plt.legend(prop={'size': 12})
    plt.title('训练数据重构误差分布与阈值', fontsize=14)
    plt.xlabel('重构误差', fontsize=12)
    plt.ylabel('频次', fontsize=12)
    plt.savefig('autoencoder_threshold_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()


    # 评估模型
    model.eval()

    # 测试数据集1model.eval()

    # 测试数据集1
    print("测试数据集1的结果:")
    errors1, result1 = predict_autoencoder(model, testdata1, criterion, threshold, window_size, device)
    normal_count1 = result1.count(True)
    anomaly_count1 = len(result1) - normal_count1
    print(f"正常数据点: {normal_count1}, 异常数据点: {anomaly_count1}")
    print(f"异常比例: {anomaly_count1/len(result1)*100:.2f}%")

    # 计算测试数据集1的评估指标
    metrics1 = calculate_metrics(result1, len(testdata1))
    print("\n测试数据集1的评估指标:")
    for metric_name, metric_value in metrics1.items():
        if metric_name != "混淆矩阵":
            print(f"{metric_name}: {metric_value:.4f}")
        else:
            print(f"{metric_name}:")
            for cm_name, cm_value in metric_value.items():
                print(f"  {cm_name}: {cm_value}")

    # 测试数据集2
    print("\n测试数据集2的结果:")
    errors2, result2 = predict_autoencoder(model, testdata2, criterion, threshold, window_size, device)
    normal_count2 = result2.count(True)
    anomaly_count2 = len(result2) - normal_count2
    print(f"正常数据点: {normal_count2}, 异常数据点: {anomaly_count2}")
    print(f"异常比例: {anomaly_count2/len(result2)*100:.2f}%")

    # 计算测试数据集2的评估指标
    metrics2 = calculate_metrics(result2, len(testdata2))
    print("\n测试数据集2的评估指标:")
    for metric_name, metric_value in metrics2.items():
        if metric_name != "混淆矩阵":
            print(f"{metric_name}: {metric_value:.4f}")
        else:
            print(f"{metric_name}:")
            for cm_name, cm_value in metric_value.items():
                print(f"  {cm_name}: {cm_value}")

    # ... 现有代码 ...

    # 在可视化部分添加混淆矩阵图
    # 绘制测试数据集1的混淆矩阵
    cm_plot1 = plot_confusion_matrix(metrics1, '测试数据集1的混淆矩阵')
    cm_plot1.savefig('confusion_matrix_dataset1.png', dpi=300, bbox_inches='tight')
    cm_plot1.show()

    # 绘制测试数据集2的混淆矩阵
    cm_plot2 = plot_confusion_matrix(metrics2, '测试数据集2的混淆矩阵')
    cm_plot2.savefig('confusion_matrix_dataset2.png', dpi=300, bbox_inches='tight')
    cm_plot2.show()

    # 添加评估指标对比图
    plt.figure(figsize=(12, 6))
    metrics_names = ["准确率", "精确率", "检测率(召回率)", "F1分数", "误检率"]
    metrics_values1 = [metrics1[name] for name in metrics_names]
    metrics_values2 = [metrics2[name] for name in metrics_names]

    x = np.arange(len(metrics_names))
    width = 0.35

    plt.bar(x - width/2, metrics_values1, width, label='测试数据集1')
    plt.bar(x + width/2, metrics_values2, width, label='测试数据集2')

    plt.ylabel('指标值', fontsize=14)
    plt.title('两个测试数据集的评估指标对比', fontsize=16)
    plt.xticks(x, metrics_names, fontsize=12, rotation=15)
    plt.legend(prop={'size': 12})

    plt.tight_layout()
    plt.savefig('metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()