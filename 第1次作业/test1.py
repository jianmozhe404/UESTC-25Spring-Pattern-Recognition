import pandas as pd
import numpy as np
from scipy.stats import chi2
from pathlib import Path

def origin_normalized_detection():
    # 加载原始数据参数
    def load_origin_params():
        base_dir = Path(__file__).parent
        path = next((p for p in [base_dir, base_dir/'data'] if 
                    (p/'means_data.csv').exists()), None)
        if not path:
            raise FileNotFoundError("原始数据文件缺失，请检查means_data.csv和cov_data.csv位置")
            
        mean = pd.read_csv(path/'means_data.csv', header=None).squeeze().values
        cov = pd.read_csv(path/'cov_data.csv', header=None).values
        return mean.astype(np.float64), cov.astype(np.float64)

    # 计算原始数据最大概率
    def calc_origin_max_pdf(mean, cov):
        """理论最大值位于均值点：max_pdf = f(μ)"""
        _, logdet = np.linalg.slogdet(cov)
        return 1 / ( (2*np.pi)**15.5 * np.exp(logdet/2) )

    # 测试数据概率计算
    def test_pdf(X, mean, cov):
        inv_cov = np.linalg.pinv(cov)
        delta = X - mean
        md = np.einsum('ij,jk,ik->i', delta, inv_cov, delta)
        pdf = max_pdf_origin * np.exp(-0.5 * md)  # 利用预存max_pdf优化计算
        return pdf, md

    # 主流程
    mean, cov = load_origin_params()
    max_pdf_origin = calc_origin_max_pdf(mean, cov)  # 预计算原始最大概率
    
    # 加载测试数据
    test_path = next((p for p in [Path(__file__).parent, Path(__file__).parent/'data'] 
                     if (p/'Test_data2.csv').exists()), None)
    test_df = pd.read_csv(test_path/'Test_data2.csv', header=None)
    
    # 计算测试数据统计量
    pdf_values, md_scores = test_pdf(test_df.values, mean, cov)
    normalized_pdf = pdf_values / max_pdf_origin  # 基于原始数据的归一化
    
    # 构建结果集
    report = test_df.rename(columns=lambda x: f"X{x+1}")
    report['Raw_PDF'] = pdf_values
    report['Normalized_PDF'] = normalized_pdf
    report['Is_Anomaly'] = md_scores > chi2.ppf(0.99, 31)
    
    # 导出全量报告
    report.to_csv("Result_data2.csv", index=False, 
                 float_format="%.4e")  # 四位科学计数法
    return report

if __name__ == "__main__":
    try:
        df = origin_normalized_detection()
        print(f"原始基准归一化检测完成")
        print(f"▸ 输出文件：{Path('Result_data2.csv').resolve()}")
        print(f"▸ 归一化基准值：{df['Raw_PDF'].max()/df['Normalized_PDF'].max():.4e}")
        print(f"▸ 异常样本占比：{df['Is_Anomaly'].mean():.2%}")
    except Exception as e:
        print(f"[ERROR] 执行失败：{str(e)}")