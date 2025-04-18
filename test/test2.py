import pandas as pd 
import numpy as np 
from scipy.stats import chi2 
from pathlib import Path 
 
def gaussian_kernel_anomaly_detection():
    # 数据加载与参数初始化
    base_dir = Path(__file__).parent 
    mean = pd.read_csv(base_dir/'means_data.csv', header=None).squeeze().values 
    cov = pd.read_csv(base_dir/'cov_data.csv', header=None).values * 0.7  # 带宽参数应用 
    test_data = pd.read_csv(base_dir/'Test_data2.csv', header=None).values 
    
    # 矩阵稳定性处理 
    eigenvalues = np.linalg.eigvalsh(cov)
    assert np.all(eigenvalues > 1e-8), "协方差矩阵存在非正定问题"
    inv_cov = np.linalg.inv(cov)
    
    # 核心计算引擎 
    dim = 31 
    log_det = np.sum(np.log(eigenvalues))
    base_density = 1/( (2*np.pi)**(dim/2) * np.sqrt(np.exp(log_det)) )
    
    # 向量化距离计算 
    delta = test_data - mean.reshape(1, -1)
    mahalanobis = np.einsum('ij,jk,ik->i', delta, inv_cov, delta)
    
    # 双密度生成系统 
    raw_density = base_density * np.exp(-0.5 * mahalanobis)
    normalized_density = np.exp(-0.5 * mahalanobis)  # 理论归一化 
    
    # 智能判定逻辑 
    threshold = chi2.ppf(0.95, dim)
    is_anomaly = mahalanobis > threshold 
    
    # 报告生成与输出 
    report = pd.DataFrame(test_data, columns=[f'F_{i+1}' for i in range(dim)])
    report['Raw_Density'] = raw_density 
    report['Norm_Density'] = normalized_density 
    report['Is_Anomaly'] = is_anomaly 
    
    # 科学计数法格式化 
    report.to_csv(base_dir/'Report_data2.csv', index=False, float_format="%.4e")
    
    # 异常统计仪表盘 
    anomaly_ratio = np.mean(is_anomaly)
    print(f" 异常样本占比分析:\n  总样本数: {len(test_data)}\n  异常数量: {sum(is_anomaly)}"
          f"\n  异常比例: {anomaly_ratio:.2%} (±{1.96*np.sqrt(anomaly_ratio*(1-anomaly_ratio)/len(test_data)):.3%})")
 
if __name__ == "__main__":
    gaussian_kernel_anomaly_detection()