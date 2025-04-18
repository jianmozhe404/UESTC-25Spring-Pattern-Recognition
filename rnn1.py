# %% [markdown]
# ## Pytorch 实现版本
# 
# 目的:
# 1. 跑通流程
# 2. 熟悉pytorch

# %%
import torch
# 修复matplotlib导入问题
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from collections import deque
from tqdm import tqdm

# 解决matplotlib中文显示问题
import matplotlib
# 方案1: 使用SimHei字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
matplotlib.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# 方案2: 使用微软雅黑字体(备选方案)
# from matplotlib import font_manager
# font_path = 'C:/Windows/Fonts/msyh.ttc'  # 微软雅黑字体路径
# font_manager.fontManager.addfont(font_path)
# plt.rcParams['font.family'] = ['Microsoft YaHei']
# plt.rcParams['axes.unicode_minus'] = False

# %%
#读取数据
train_file_path = 'test\Train_data.csv'
test1_file_path = 'test\Test_data1.csv'
test2_file_path = 'test\Test_data2.csv'

train_data = pd.read_csv(train_file_path)
testdata1 = pd.read_csv(test1_file_path)
testdata2 = pd.read_csv(test2_file_path)

# 将数据转换为 numpy 数组
train_data = train_data.values
testdata1 = testdata1.values
testdata2 = testdata2.values

# 数据标准化
scaler = StandardScaler()
train_data = scaler.fit_transform(train_data)  # 计算均值、标准差
testdata1 = scaler.transform(testdata1)       # 使用训练集的均值、标准差
testdata2 = scaler.transform(testdata2)

# 将 numpy 数组转换为 PyTorch 张量
train_data_tensor = torch.tensor(train_data, dtype=torch.float32, device=torch.device('cuda:0'))
testdata1_tensor = torch.tensor(testdata1, dtype=torch.float32, device=torch.device('cuda:0'))
testdata2_tensor = torch.tensor(testdata2, dtype=torch.float32, device=torch.device('cuda:0'))

# %%
print("Train data shape:", train_data_tensor.shape)
print("Test data 1 shape:", testdata1_tensor.shape)
print("Test data 2 shape:", testdata2_tensor.shape)

# %%
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

# %%
#滑窗切分样本
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

# %%
input_size = 31  # 特征维度
hidden_size = 256  # 隐藏层维度
num_layers = 3    # 隐藏层层数
batch_size = 64   # 批次大小
learning_rate = 1e-3
num_epochs = 500
window_size = 200
stride = 10
threshold = 1e-5
device = device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

dataset = slidingWindowDataset(train_data,window_size,stride)

#暂时选择不打乱数据
dataloader = torch.utils.data.DataLoader(dataset,batch_size,shuffle=False)

model = LSTM(input_size,hidden_size,device,num_layers)
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
#损失函数暂时选择均方误差
criterion = torch.nn.MSELoss()

# %%
# 修改训练函数，添加早停机制
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

# %%
dataset = slidingWindowDataset(train_data, window_size, stride)

# 训练模型
model.train()
model, best_loss = train(model, dataloader, optimizer, criterion, num_epochs, device, patience=50)

print(f"训练完成，最佳损失: {best_loss:.6f}")

# 保存模型
torch.save(model.state_dict(), 'd:/lecture/25spring/模式识别/作业1/rnn_model.pth')

# 计算阈值
threshold, train_errors = calculate_threshold(model, train_data, criterion, window_size, device)

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

# 评估模型 - 修改为同时测试两个数据集
model.eval()

# 测试数据集1
print("测试数据集1的结果:")
errors1, result1 = predict(model, testdata1, criterion, threshold, window_size, device)
normal_count1 = result1.count(True)
anomaly_count1 = len(result1) - normal_count1
print(f"正常数据点: {normal_count1}, 异常数据点: {anomaly_count1}")
print(f"异常比例: {anomaly_count1/len(result1)*100:.2f}%")

# 测试数据集2
print("\n测试数据集2的结果:")
errors2, result2 = predict(model, testdata2, criterion,threshold, window_size, device)
normal_count2 = result2.count(True)
anomaly_count2 = len(result2) - normal_count2
print(f"正常数据点: {normal_count2}, 异常数据点: {anomaly_count2}")
print(f"异常比例: {anomaly_count2/len(result2)*100:.2f}%")

# 可视化两个测试数据集的异常检测结果
plt.figure(figsize=(15, 10))

# 数据集1的可视化
plt.subplot(2, 1, 1)
feature_idx = 0  # 可以选择任意特征进行可视化
plt.plot(testdata1[:, feature_idx], label='测试数据1')

# 标记异常点
anomaly_indices1 = [i for i, r in enumerate(result1) if not r]
if anomaly_indices1:
    plt.scatter(anomaly_indices1, testdata1[anomaly_indices1, feature_idx], 
                color='red', label='异常点')

plt.legend(prop={'size': 12})
plt.title('测试数据集1异常检测结果', fontsize=16)
plt.xlabel('时间步', fontsize=14)
plt.ylabel('特征值', fontsize=14)

# 数据集2的可视化
plt.subplot(2, 1, 2)
plt.plot(testdata2[:, feature_idx], label='测试数据2')

# 标记异常点
anomaly_indices2 = [i for i, r in enumerate(result2) if not r]
if anomaly_indices2:
    plt.scatter(anomaly_indices2, testdata2[anomaly_indices2, feature_idx], 
                color='red', label='异常点')

plt.legend(prop={'size': 12})
plt.title('测试数据集2异常检测结果', fontsize=16)
plt.xlabel('时间步', fontsize=14)
plt.ylabel('特征值', fontsize=14)

plt.tight_layout()
plt.savefig('d:/lecture/25spring/模式识别/作业1/both_datasets_anomaly_detection.png', dpi=300, bbox_inches='tight')
plt.show()

# 比较两个数据集的异常比例
labels = ['测试数据集1', '测试数据集2']
normal_percentages = [normal_count1/len(result1)*100, normal_count2/len(result2)*100]
anomaly_percentages = [anomaly_count1/len(result1)*100, anomaly_count2/len(result2)*100]

plt.figure(figsize=(10, 6))
x = np.arange(len(labels))
width = 0.35

plt.bar(x - width/2, normal_percentages, width, label='正常数据')
plt.bar(x + width/2, anomaly_percentages, width, label='异常数据')

plt.ylabel('百分比 (%)', fontsize=14)
plt.title('两个测试数据集的异常检测比例对比', fontsize=16)
plt.xticks(x, labels, fontsize=12)
plt.legend(prop={'size': 12})

plt.tight_layout()
plt.savefig('d:/lecture/25spring/模式识别/作业1/anomaly_comparison.png', dpi=300, bbox_inches='tight')
plt.show()


