import sys
import numpy as np
import pandas as pd
from scipy import linalg as la
from plot_utils import create_calibration_plot
from sklearn.model_selection import train_test_split

def ellipsoid_fit_svd(s):
    '''
    Estimate ellipsoid parameters from a set of points using SVD.

    Parameters
    ----------
    s : array_like
        The samples (M,N) where M=3 (x,y,z) and N=number of samples.

    Returns
    -------
    M, n, d : array_like, array_like, float
        The ellipsoid parameters M, n, d.
    '''
    # 确保输入数据是(3, N)的形状
    if s.shape[0] != 3:
        s = s.T

    # 构建数据矩阵D，包含所有二次项、线性项和常数项。
    # 方程为：ax^2 + by^2 + cz^2 + 2dxy + 2exz + 2fyz + 2gx + 2hy + 2iz + j = 0
    D = np.column_stack([s[0]**2, s[1]**2, s[2]**2,
                         2*s[0]*s[1], 2*s[0]*s[2], 2*s[1]*s[2],
                         2*s[0], 2*s[1], 2*s[2], np.ones_like(s[0])])

    # 执行SVD，找到D的零空间。
    # SVD会返回 U, Sigma, Vh，其中Vh的最后一行就是零空间向量。
    _, _, vh = np.linalg.svd(D, full_matrices=False)
    
    # 零空间向量v即为椭球方程的系数
    v = vh[-1, :]

    # 从系数v中提取M、n和d
    M = np.array([[v[0], v[3], v[4]],
                      [v[3], v[1], v[5]],
                      [v[4], v[5], v[2]]])
    n = np.array([[v[6]],
                      [v[7]],
                      [v[8]]])
    d = v[9]
    
    return M, n, d

def calibrate(s, F=500.):
    """
    使用svd拟合的参数进行校准。
    
    参数
    ----------
    s : array_like
        地磁数据
    F : float
        校准后的目标模值

    返回
    -------
    S, h : array_like, array_like
        软磁矩阵和硬磁向量
    """
    # 确保输入数据是(3, N)的形状，如果是(N, 3)则转置
    if s.shape[0] != 3:
        s = s.T
        
    M, n, d = ellipsoid_fit_svd(s)
    
    # 以下校准逻辑与原代码相同
    M_1 = la.inv(M)
    h = -np.dot(M_1, n).flatten()

    # 检查缩放因子
    scale_factor = np.dot(n.T, np.dot(M_1, n)) - d
    print(f"Scale factor before sqrt: {scale_factor}")
    
    # 使用 sqrtm() 处理矩阵的平方根
    S = np.real(F / np.sqrt(scale_factor) * la.sqrtm(M))
    
    return S, h

def main():
    """使用SVD算法椭球拟合示例"""
    if len(sys.argv) < 2:
        print("用法: python ellipsoid_fit_svd.py <地磁数据.csv>")
        sys.exit(1)

    df = pd.read_csv(sys.argv[1])[100:-100]
    data = df[['X', 'Y', 'Z']].values
    train, test = train_test_split(data, test_size=0.5, random_state=42)

    print(f"数据点数量: {len(train)}")

    # 使用校准API
    S, h = calibrate(train, F=500.0)
    print(S, h)
    print(data.shape)

    # 输出校准参数的C代码格式
    print("\n校准参数 (C代码格式):")
    print("float S[3][3] = {")
    for row in S:
        print("    { " + ", ".join(f"{v:.8f}" for v in row) + " },")
    print("};")
    print("float h[3] = { " + ", ".join(f"{v:.8f}" for v in h) + " };")

    # 校准数据
    calibrated = (test - h) @ S

    # 计算校准效果
    radius = la.norm(calibrated, axis=1)
    print(f"\n校准效果:")
    print(f"校准后半径: 均值={radius.mean():.3f}, 标准差={radius.std():.3f}")
    print(f"椭球中心: [{h[0]:.6f}, {h[1]:.6f}, {h[2]:.6f}]")

    # 可视化结果
    create_calibration_plot(data, calibrated, "SVD Algorithm", 'svd_calibration_result.png')

if __name__ == "__main__":
    main()