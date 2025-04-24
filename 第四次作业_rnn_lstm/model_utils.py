import torch
# 修复matplotlib导入问题
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from collections import deque
from tqdm import tqdm

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"随机种子已设置为: {seed}")

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


class LSTMAutoencoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, device, num_layers=2):
        super(LSTMAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.device = device
        
        # 编码器
        self.encoder = torch.nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers=num_layers,
            dropout=0.2,
            batch_first=True
        )
        
        # 解码器
        self.decoder = torch.nn.LSTM(
            hidden_dim, 
            hidden_dim, 
            num_layers=num_layers,
            dropout=0.2,
            batch_first=True
        )
        
        # 输出层
        self.output_layer = torch.nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # 编码
        _, (h_n, c_n) = self.encoder(x)
        
        # 使用编码器的最后隐藏状态初始化解码器输入
        # 取最后一层的隐藏状态并重复seq_len次
        decoder_input = h_n[-1].unsqueeze(1).repeat(1, seq_len, 1)
        
        # 解码
        decoder_output, _ = self.decoder(decoder_input)
        
        # 输出层
        output = self.output_layer(decoder_output)
        
        return output

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

class AutoencoderDataset:
    def __init__(self, data, window_size, stride):
        super(AutoencoderDataset, self).__init__()
        self.data = data
        self.window_size = window_size
        self.stride = stride
        self.num_samples = (len(data) - window_size) // stride + 1

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        输入: 一个样本的索引
        返回:
        1. 样本本身: (window_size, features)
        2. 目标: 与样本相同 (window_size, features)
        """
        start = idx * self.stride
        end = start + self.window_size
        sample = self.data[start:end, :]
        # 自编码器的输入和目标相同
        return torch.tensor(sample, dtype=torch.float32), torch.tensor(sample, dtype=torch.float32)

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
            torch.save(best_model_state, f'save_model/rnn_best_model.pth')
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

def train_autoencoder(model, dataloader, optimizer, criterion, num_epochs, device, patience):
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
        
        for samples, targets in dataloader:
            samples = samples.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            reconstructed = model(samples)
            
            # 计算重构误差
            loss = criterion(reconstructed, targets)
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
            torch.save(best_model_state, f'save_model/autoencoder_best_model.pth')
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
    sigma_multiplier: 标准差的乘数因子
    
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

def calculate_autoencoder_threshold(model, train_data, criterion, window_size, device, sigma_multiplier=3):
    """
    基于训练数据计算自编码器异常检测的阈值
    """
    model.eval()
    reconstruction_errors = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(train_data) - window_size, 10), desc="计算阈值"):  # 使用步长10加速计算
            x_window = torch.tensor(train_data[i:i+window_size], dtype=torch.float32).unsqueeze(0).to(device)
            reconstructed = model(x_window)
            # 计算整个窗口的重构误差
            error = criterion(reconstructed, x_window).item()
            reconstruction_errors.append(error)
    
    # 计算阈值：均值 + sigma_multiplier*标准差
    threshold = np.mean(reconstruction_errors) + sigma_multiplier * np.std(reconstruction_errors)
    print(f"计算得到的阈值: {threshold}")
    
    return threshold, reconstruction_errors

 # %%
def predict_autoencoder(model, test_data, criterion, threshold, window_size, device):
    """
    使用自编码器进行异常检测
    """
    test_data = torch.Tensor(test_data)
    model.eval()
    model.to(device)
    
    result = []
    errors = []
    
    # 前window_size个点无法预测，默认为正常
    for i in range(window_size):
        result.append(True)
    
    with torch.no_grad():
        for i in tqdm(range(window_size, len(test_data)), desc="执行预测"):
            # 获取当前窗口
            x_window = test_data[i-window_size:i].unsqueeze(0).to(device)
            # 重构窗口
            reconstructed = model(x_window)
            # 计算重构误差
            error = criterion(reconstructed, x_window).item()
            errors.append(error)
            
            # 如果重构误差大于阈值，则判断为异常
            if error > threshold:
                result.append(False)  # 异常
            else:
                result.append(True)   # 正常
    
    return errors, result
    
       
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

def calculate_metrics(result, test_data_length, anomaly_start=None):
    """
    计算异常检测的评估指标
    
    参数:
    result: 预测结果列表，True表示正常，False表示异常
    test_data_length: 测试数据总长度
    anomaly_start: 异常数据开始的索引，如果为None则使用默认值(总长度-1999)
    
    返回:
    metrics: 包含各种评估指标的字典
    """
    if anomaly_start is None:
        anomaly_start = test_data_length - 1999
    
    # 创建真实标签
    true_labels = [True] * anomaly_start + [False] * (test_data_length - anomaly_start)
    
    # 确保预测结果和真实标签长度一致
    if len(result) < test_data_length:
        # 如果预测结果不足，用正常标签填充
        result = result + [True] * (test_data_length - len(result))
    elif len(result) > test_data_length:
        # 如果预测结果过多，截断
        result = result[:test_data_length]
    
    # 转换为二进制标签(0:异常, 1:正常)
    y_true = [1 if label else 0 for label in true_labels]
    y_pred = [1 if label else 0 for label in result]
    
    # 计算混淆矩阵
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # 计算评估指标
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0  # 检测率
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0  # 误检率
    
    metrics = {
        "准确率": accuracy,
        "精确率": precision,
        "检测率(召回率)": recall,
        "F1分数": f1,
        "误检率": false_alarm_rate,
        "混淆矩阵": {
            "真正例(TP)": tp,
            "假正例(FP)": fp,
            "真负例(TN)": tn,
            "假负例(FN)": fn
        }
    }
    
    return metrics

# 添加混淆矩阵可视化
def plot_confusion_matrix(metrics, title):
    cm = np.array([
        [metrics["混淆矩阵"]["真负例(TN)"], metrics["混淆矩阵"]["假正例(FP)"]],
        [metrics["混淆矩阵"]["假负例(FN)"], metrics["混淆矩阵"]["真正例(TP)"]]
    ])
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title, fontsize=16)
    plt.colorbar()
    
    classes = ['异常', '正常']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=12)
    plt.yticks(tick_marks, classes, fontsize=12)
    
    # 在格子中显示数字
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14)
    
    plt.ylabel('真实标签', fontsize=14)
    plt.xlabel('预测标签', fontsize=14)
    plt.tight_layout()
    return plt