import torch
# 修复matplotlib导入问题
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from collections import deque
from tqdm import tqdm

class RNN(torch.nn.Module):
    def __init__(self,input_dim,hidden_dim,device,num_layers=2):
        super(RNN,self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = False
        self.rnn = torch.nn.RNN(self.input_dim,self.hidden_dim,num_layers=num_layers,dropout=0.1,
                           bidirectional=self.bidirectional,batch_first=True)
       
        self.fc = torch.nn.Linear(hidden_dim,input_dim)  
        self.device = device
        
    def forward(self,X,h_prev=None):
        """
        X : (batch_size, seq_length, features)
        h_prev : (num_layers*num_directions, batch_size, hidden_dim)
        output : (batch_size, seq_length, hidden_dim*num_directions)
        h_n : (num_layers*num_directions, batch_size, hidden_dim)
        """
        batch_size = X.size(0)
        if h_prev is None:
            # 修正隐藏状态维度，考虑双向因素
            
            h_prev = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(self.device)
        
        output, h_n = self.rnn(X, h_prev)
        
        # 获取最后一个时间步的输出并通过全连接层
        out = self.fc(output[:, -1, :])
        return out

class LSTM(torch.nn.Module):
    def __init__(self,input_dim,hidden_dim,device,num_layers=3):
        super(LSTM,self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = False
        self.rnn = torch.nn.LSTM(self.input_dim,self.hidden_dim,num_layers=num_layers,dropout=0.1,
                           bidirectional=self.bidirectional,batch_first=True)
       
        self.fc = torch.nn.Linear(hidden_dim,input_dim)  
        self.device = device
        
    def forward(self,X,h_prev=None):
        """
        X : (batch_size, seq_length, features)
        h_prev : (num_layers*num_directions, batch_size, hidden_dim)
        output : (batch_size, seq_length, hidden_dim*num_directions)
        h_n : (num_layers*num_directions, batch_size, hidden_dim)
        """
        batch_size = X.size(0)
        if h_prev is None:
            
            h_prev = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(self.device)
            c_prev = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(self.device)
            hidden = (h_prev,c_prev)

        output, (h_n,c_n) = self.rnn(X, hidden)
        
        # 获取最后一个时间步的输出并通过全连接层
        out = self.fc(output[:, -1, :])
        return out

class slidingWindowDataset:
    def __init__(self,squeeze,window_size,stride):
        super(slidingWindowDataset,self).__init__()
        self.sq = squeeze
        self.window_size = window_size
        self.stride = stride
        self.num_samples = (len(squeeze) - window_size) // stride + 1

    def __len__(self):
        """
        返回一个样本的长度
        """
        return self.num_samples
    
    def __getitem__(self,idx):
        """ 
        输入: 一个样本的索引
        返回:
        1. 样本本身: (window_size,features)
        2. 样本标签: (1,features) 即下一个时间步的数据
        """
        start = idx*self.stride
        end = start + self.window_size
        sample = self.sq[start:end,:]
        label = self.sq[end]
        return torch.tensor(sample,dtype=torch.float32),torch.tensor(label,dtype=torch.float32) 

def train(model, dataloader, optimizer, criterion, num_epochs, device, patience=20):
    model.to(device)
    total_steps = len(dataloader) * num_epochs
    global_step = 0
    
    # 早停相关变量
    best_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = []
        
        for samples, signal in dataloader:
            samples = samples.to(device)
            signal = signal.to(device)
            
            optimizer.zero_grad()
            y_hat = model(samples)
            loss = criterion(y_hat, signal)
            loss.backward()
            optimizer.step()
            
            epoch_loss.append(loss.item())
            global_step += 1
            
            if global_step % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{global_step}/{total_steps}], Loss: {loss.item():.4f}')
        
        avg_loss = sum(epoch_loss) / len(epoch_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')
        
        # 早停检查
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            # 保存最佳模型
            torch.save(best_model_state, f'd:/lecture/25spring/模式识别/作业1/rnn_best_model.pth')
            print(f'Epoch [{epoch+1}]: 保存最佳模型，损失: {best_loss:.6f}')
        else:
            patience_counter += 1
            print(f'Epoch [{epoch+1}]: 损失未改善，耐心计数: {patience_counter}/{patience}')
            
        # 如果连续patience个epoch没有改善，则停止训练
        if patience_counter >= patience:
            print(f'早停: {patience}个epoch内损失未改善')
            break
    
    # 训练结束后，加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f'训练完成，加载最佳模型，最佳损失: {best_loss:.6f}')
    
    return model, best_loss

def calculate_threshold(model, train_data, criterion, window_size, device, sigma_multiplier=3):
    """
    基于训练数据计算异常检测的阈值
    
    参数:
    model: 训练好的模型
    train_data: 训练数据
    criterion: 损失函数
    window_size: 滑动窗口大小
    device: 计算设备
    sigma_multiplier: 标准差的乘数因子，默认为3（3-sigma原则）
    
    返回:
    threshold: 计算得到的阈值
    train_errors: 训练数据上的预测误差列表
    """
    model.eval()
    train_errors = []
    
    with torch.no_grad():
        for i in range(window_size, len(train_data)):
            x_train = torch.tensor(train_data[i-window_size:i], dtype=torch.float32).to(device)
            y_train = torch.tensor(train_data[i], dtype=torch.float32).to(device)
            y_hat = model(x_train.unsqueeze(0))
            error = criterion(y_hat, y_train).item()
            train_errors.append(error)
    
    # 计算阈值：均值 + sigma_multiplier*标准差
    threshold = np.mean(train_errors) + sigma_multiplier * np.std(train_errors)
    print(f"计算得到的阈值: {threshold}")
    
    return threshold, train_errors

def predict(model, test_data, criterion, threshold, window_size, device):
    """ 
    在测试集中测试模型的正确性
    
    参数:
    test_data: [num_samples, num_features] 
    threshold: 用于异常检测的阈值
    """
    test_data = torch.Tensor(test_data)
    model.eval()
    model.to(device)
    test_data = test_data.to(device)
    windows = deque(maxlen=window_size)
    
    result = []
    errors = []
    with torch.no_grad():
        for i, signal in enumerate(test_data):
            if i < window_size: 
                windows.append(signal)
                result.append(True)  # 前window_size个点默认为正常
                continue

            x = torch.stack(list(windows)).unsqueeze(0)  # 添加batch维度
            y_hat = model(x)
            error = criterion(y_hat, signal).item()
            errors.append(error)
            windows.append(signal)
            
            # 如果误差大于阈值，则判断为异常
            if error > threshold:
                result.append(False)  # 异常
            else:
                result.append(True)   # 正常
    
    return errors, result