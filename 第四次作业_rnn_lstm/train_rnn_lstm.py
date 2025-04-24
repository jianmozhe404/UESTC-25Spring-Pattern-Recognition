
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np 

# 从 model_utils 导入所需的类和函数
from model_utils import LSTM, slidingWindowDataset, train

def main():
    # %% 数据加载与预处理
    train_file_path = 'test/Train_data.csv'
    train_data_df = pd.read_csv(train_file_path)
    train_data_np = train_data_df.values

    scaler = StandardScaler()
    train_data_scaled = scaler.fit_transform(train_data_np)

    # %% 超参数设置
    input_size = train_data_scaled.shape[1]  # 从数据自动获取特征维度
    hidden_size = 128
    num_layers = 3
    batch_size = 64
    learning_rate = 1e-3
    num_epochs = 500 # 可以根据需要调整
    window_size = 100
    stride = 3
    patience = 50 # 早停耐心值
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_save_path = 'save_model/lstm_best_model.pth' # 明确模型保存路径

    print(f"使用设备: {device}")
    print(f"输入维度: {input_size}")

    # 将 numpy 数组转换为 PyTorch 张量
    train_data_tensor = torch.tensor(train_data_scaled, dtype=torch.float32) # 暂时放CPU，dataloader处理

    # %% 数据集和数据加载器
    dataset = slidingWindowDataset(train_data_tensor, window_size, stride)
    # pin_memory=True 可以加速 GPU 数据传输
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    # %% 模型、优化器、损失函数
    model = LSTM(input_size, hidden_size, device, num_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()

    print("开始训练...")
    trained_model, best_loss = train(model, dataloader, optimizer, criterion, num_epochs, device, patience)
    print(f"训练完成，最佳损失: {best_loss:.6f}")
    print(f"最佳模型已保存至: {model_save_path}")

if __name__ == "__main__":
    main()