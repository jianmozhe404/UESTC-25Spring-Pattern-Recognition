# %% [markdown]
# ## Pytorch 实现版本
# 
# 目的:
# 1. 跑通流程
# 2. 熟悉pytorch

# %%
import torch
import matplotlib as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from collections import deque
from tqdm import tqdm

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
        self.rnn = torch.nn.RNN(self.input_dim,self.hidden_dim,num_layers=num_layers,dropout=0.1,
                           bidirectional= False,batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim,input_dim)
        self.device = device
        
    def forward(self,X,h_prev=None):
        """
        X : (sq_length,features)
        h_prev : (num_layers*D, Hout)
        output : (sq_length,D*Hout) #(时间步, 隐藏层大小)
        h_n : (D*num_layers,N,Hout) (隐藏层层数, 隐藏层大小)
        """
        if h_prev is None:
            h_prev = torch.zeros((self.num_layers,self.hidden_dim)).to(self.device)
        output, h_n = self.rnn(X,h_prev)
        out = self.fc(output[-1,:])
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
num_layers = 2    # 隐藏层层数
batch_size = 64   # 批次大小
learning_rate = 1e-3
num_epochs = 15000
window_size = 100
stride = 10
threshold = 1e-5
device = device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

dataset = slidingWindowDataset(train_data,window_size,stride)

#暂时选择不打乱数据
dataloader = torch.utils.data.DataLoader(dataset,batch_size,shuffle=False)

model = RNN(input_size,hidden_size,device,num_layers)
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
#损失函数暂时选择均方误差
criterion = torch.nn.MSELoss()

# %%
def train(model,dataloader,optimizer,criterion,num_epochs,device):
    model.to(device)
    counter = 0
    for i in range(10):
        with tqdm(total=int(num_epochs/10), desc='Iteration %d' % i) as pbar:
            loss_list = []
            for samples, signal in dataloader:
                samples = samples.to(device)
                signal = signal.to(device)
                optimizer.zero_grad()
                y_hat = model(samples)
                loss = criterion(y_hat,signal)
                loss_list.append(loss.item())
                loss.backward()
                optimizer.step()

                counter += 1
                if (counter+1) % 10 == 0:
                    pbar.set_postfix({'epochs': '%d' % (counter+10),
                                      'loss': '%.3f' % np.mean(loss_list[-10:])})
                pbar.update(1)

#测试函数 --- 不写了

#预测函数
def predict(model,test_data,criterion,threshold,window_size,device):
    """ 
    在测试集中测试模型的正确性
    test_data: [num_samples, num_features] 
    threshold: 异常判断的阈值
    """
    test_data = torch.Tensor(test_data)
    model.eval()
    model.to(device)
    test_data = test_data.to(device)
    windows = deque(maxlen=window_size)
    
    result = []
    total_loss = 0.0
    with torch.no_grad():
        for i , signal in enumerate(test_data):
            if i < window_size: 
                windows.append(signal)
                result.append(True)
                continue

            y_hat = model(torch.stack(list(windows)))
            loss = criterion(y_hat,signal)
            total_loss += loss.item()
            windows.append(signal)
            if loss.item() >= threshold:
                result.append(False)
            else:
                result.append(True)

    return total_loss,result

# %%
#训练模型
model.train()
train(model,dataset,optimizer,criterion,num_epochs,device=device)

# %%
model.eval()
total_loss,result = predict(model,testdata1,criterion,1e-1,window_size,device)
print(result.count(True))

# %%
print(result)


