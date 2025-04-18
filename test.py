import numpy as np

# 激活函数及其导数
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

# RNN Block 类
class RNNBlock:
    def __init__(self, input_dim, hidden_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 初始化权重和偏置
        self.W_h = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.W_x = np.random.randn(hidden_dim, input_dim) * 0.01
        self.b = np.zeros((hidden_dim, 1))
        
        # 存储中间状态
        self.h = None
        self.Z = []

    def forward(self, X):
        """
        前向传播
        :param X: 输入数据 (input_dim, T)
        :return: 隐藏状态序列 (hidden_dim, T)
        """
        T = X.shape[1]  # 时间步数
        self.h = np.zeros((self.hidden_dim, 1))  # 初始化隐藏状态
        self.Z = []
        H = []
        
        for t in range(T):
            x_t = X[:, t].reshape(-1, 1)
            z_t = np.dot(self.W_h, self.h) + np.dot(self.W_x, x_t) + self.b
            self.h = tanh(z_t)
            
            self.Z.append(z_t)
            H.append(self.h)
        
        return np.hstack(H)

    def backward(self, dH, X, learning_rate):
        """
        反向传播
        :param dH: 损失对隐藏状态的梯度 (hidden_dim, T)
        :param X: 输入数据 (input_dim, T)
        :param learning_rate: 学习率
        :return: 损失对输入的梯度 (input_dim, T)
        """
        T = X.shape[1]
        dW_h = np.zeros_like(self.W_h)
        dW_x = np.zeros_like(self.W_x)
        db = np.zeros_like(self.b)
        
        dh_next = np.zeros_like(self.h)
        
        dX = []
        
        for t in reversed(range(T)):
            x_t = X[:, t].reshape(-1, 1)
            z_t = self.Z[t]
            
            # 计算隐藏状态的梯度
            dh_t = dH[:, t].reshape(-1, 1) + dh_next
            dz_t = dh_t * tanh_derivative(z_t)
            
            # 计算权重和偏置的梯度
            dW_h += np.dot(dz_t, self.h.T)
            dW_x += np.dot(dz_t, x_t.T)
            db += dz_t
            
            # 计算输入的梯度
            dx_t = np.dot(self.W_x.T, dz_t)
            dX.insert(0, dx_t)
            
            # 更新隐藏状态的梯度
            dh_next = np.dot(self.W_h.T, dz_t)
        
        # 更新参数
        self.W_h -= learning_rate * dW_h
        self.W_x -= learning_rate * dW_x
        self.b -= learning_rate * db
        
        return np.hstack(dX)

# 双 RNN Block 网络类
class DoubleRNN:
    def __init__(self, input_dim, hidden_dim1, hidden_dim2):
        self.rnn1 = RNNBlock(input_dim, hidden_dim1)
        self.rnn2 = RNNBlock(hidden_dim1, hidden_dim2)

    def forward(self, X):
        """
        前向传播
        :param X: 输入数据 (input_dim, T)
        :return: 第二层隐藏状态序列 (hidden_dim2, T)
        """
        H1 = self.rnn1.forward(X)
        H2 = self.rnn2.forward(H1)
        return H2

    def backward(self, dH2, X, learning_rate):
        """
        反向传播
        :param dH2: 损失对第二层隐藏状态的梯度 (hidden_dim2, T)
        :param X: 输入数据 (input_dim, T)
        :param learning_rate: 学习率
        """
        # 第二层反向传播
        dH1 = self.rnn2.backward(dH2, self.rnn1.forward(X), learning_rate)
        
        # 第一层反向传播
        self.rnn1.backward(dH1, X, learning_rate)

# 测试代码
if __name__ == "__main__":
    # 数据准备
    np.random.seed(42)
    input_dim, hidden_dim1, hidden_dim2 = 3, 5, 4
    T = 6  # 时间步数
    X = np.random.randn(input_dim, T)  # 输入数据 (3, 6)
    Y_true = np.random.randn(hidden_dim2, T)  # 真实值 (4, 6)

    # 创建双 RNN 网络
    rnn = DoubleRNN(input_dim, hidden_dim1, hidden_dim2)

    # 训练
    for epoch in range(1000):
        # 前向传播
        Y_pred = rnn.forward(X)

        # 计算损失
        loss = np.mean((Y_pred - Y_true) ** 2)
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")

        # 反向传播
        dY_pred = 2 * (Y_pred - Y_true) / (T * hidden_dim2)
        rnn.backward(dY_pred, X, learning_rate=0.01)