import sys
import argparse
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
    # 通过将常数项归一化为-1来固定尺度
    v = v / -v[-1]

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
    
    M_inv = la.inv(M)
    h = -np.dot(M_inv, n).flatten()

    # 构造一个平移矩阵，将椭球中心移到原点
    T = np.eye(4)
    T[3, 0:3] = h

    # 构造代数距离矩阵
    Q = np.eye(4)
    Q[0:3, 0:3] = M
    Q[0:3, 3] = n.flatten()
    Q[3, 0:3] = n.flatten()
    Q[3, 3] = d

    # 计算中心化后的代数距离矩阵
    Q_centered = T @ Q @ T.T

    # 提取中心化后的二次项矩阵和常数项
    M_centered = Q_centered[0:3, 0:3]
    d_centered = Q_centered[3, 3]

    # 计算缩放因子，确保其为正
    scale_factor = -d_centered
    if scale_factor <= 0:
        scale_factor = 1e-9 # 防止开方负数或零



    # 计算校准矩阵
    S = np.real(F * la.sqrtm(M_centered / scale_factor))
    
    return S, h

def main():
    """使用SVD算法椭球拟合示例 (稳定版)"""
    parser = argparse.ArgumentParser(description='磁力计椭球拟合校准程序 (SVD算法-稳定版)')
    parser.add_argument('-f', '--file', default='../data/1d000006ab_mag_20250218.csv', 
                       help='CSV文件路径 (默认: ../data/1d000006ab_mag_20250218.csv)')
    parser.add_argument('-n', '--train_samples', type=int, default=200,
                       help='训练样本数量 (默认: 200)')
    parser.add_argument('-t', '--target_field', type=float, default=500.0,
                       help='目标磁场强度 (默认: 500.0)')
    parser.add_argument('-s', '--skip_samples', type=int, default=100,
                       help='丢弃前后样本数量 (默认: 100)')
    
    args = parser.parse_args()
    
    print("=== Python磁力计校准程序 (SVD算法-稳定版) ===")
    print(f"CSV文件: {args.file}")
    print(f"训练样本数量: {args.train_samples}")
    print(f"目标磁场强度: {args.target_field}")
    print(f"丢弃前后样本数量: {args.skip_samples}")
    
    # 读取数据
    df = pd.read_csv(args.file)
    if args.skip_samples > 0:
        df = df[args.skip_samples:-args.skip_samples]
    data = df[['X', 'Y', 'Z']].values
    total_samples = len(data)
    
    print(f"从 {args.file} 读取了 {len(pd.read_csv(args.file))} 个样本")
    if args.skip_samples > 0:
        print(f"丢弃前后各 {args.skip_samples} 个样本，剩余 {total_samples} 个有效样本")
    
    # 验证训练样本数量
    if args.train_samples > total_samples:
        print(f"错误: 训练样本数量 ({args.train_samples}) 超过总样本数量 ({total_samples})")
        sys.exit(1)
    
    # 随机打乱数据
    np.random.seed(42)
    indices = np.random.permutation(total_samples)
    shuffled_data = data[indices]
    
    # 分离训练集和测试集
    train = shuffled_data[:args.train_samples]
    test = shuffled_data[args.train_samples:]
    
    print(f"训练集样本数量: {len(train)}")
    print(f"测试集样本数量: {len(test)}")
    
    print(f"\n=== 训练集数据统计 ===")
    print(f"样本数量: {len(train)}")
    print(f"均值: [{train.mean(axis=0)[0]:.1f}, {train.mean(axis=0)[1]:.1f}, {train.mean(axis=0)[2]:.1f}]")
    print(f"范围: X[{train[:,0].min():.1f}, {train[:,0].max():.1f}], Y[{train[:,1].min():.1f}, {train[:,1].max():.1f}], Z[{train[:,2].min():.1f}, {train[:,2].max():.1f}]")
    
    # 计算原始数据平均半径
    original_radius = la.norm(train, axis=1)
    print(f"原始数据平均半径: {original_radius.mean():.1f}")
    
    print(f"\n=== 开始椭球拟合校准 ===")
    
    # 使用校准API
    S, h = calibrate(train, F=args.target_field)
    
    print(f"\n=== 校准参数 ===")
    print(f"偏移向量 h: [{h[0]:.3f}, {h[1]:.3f}, {h[2]:.3f}]")
    print(f"缩放矩阵 S:")
    for i, row in enumerate(S):
        print(f"  [{row[0]:.6f}, {row[1]:.6f}, {row[2]:.6f}]")
    
    # 注释掉C代码格式输出
    # print("\n校准参数 (C代码格式):")
    # print("float S[3][3] = {")
    # for row in S:
    #     print("    { " + ", ".join(f"{v:.8f}" for v in row) + " },")
    # print("};")
    # print("float h[3] = { " + ", ".join(f"{v:.8f}" for v in h) + " };")
    
    # 计算训练集校准效果
    train_calibrated = (train - h) @ S
    train_radius = la.norm(train_calibrated, axis=1)
    
    print("\n=== 训练集校准效果 ===")
    print("\n=== 校准效果统计 ===")
    print(f"平均半径: {train_radius.mean():.3f}")
    print(f"标准差: {train_radius.std():.3f}")
    print(f"最小半径: {train_radius.min():.3f}")
    print(f"最大半径: {train_radius.max():.3f}")
    cv = (train_radius.std() / train_radius.mean()) * 100
    print(f"变异系数: {cv:.2f}%")
    print(f"期望半径: {args.target_field}")
    abs_error = abs(train_radius.mean() - args.target_field)
    rel_error = (abs_error / args.target_field) * 100
    print(f"绝对误差: {abs_error:.3f}")
    print(f"相对误差: {rel_error:.2f}%")
    
    # 计算测试集校准效果
    if len(test) > 0:
        test_calibrated = (test - h) @ S
        test_radius = la.norm(test_calibrated, axis=1)
        
        print("\n=== 测试集校准效果 ===")
        print("\n=== 校准效果统计 ===")
        print(f"平均半径: {test_radius.mean():.3f}")
        print(f"标准差: {test_radius.std():.3f}")
        print(f"最小半径: {test_radius.min():.3f}")
        print(f"最大半径: {test_radius.max():.3f}")
        cv_test = (test_radius.std() / test_radius.mean()) * 100
        print(f"变异系数: {cv_test:.2f}%")
        print(f"期望半径: {args.target_field}")
        abs_error_test = abs(test_radius.mean() - args.target_field)
        rel_error_test = (abs_error_test / args.target_field) * 100
        print(f"绝对误差: {abs_error_test:.3f}")
        print(f"相对误差: {rel_error_test:.2f}%")
        
        # 可视化结果
        create_calibration_plot(data, test_calibrated, "SVD Algorithm (Stable)", 'svd_stable_calibration_result.png')
    else:
        print("\n注意: 没有测试集数据")

if __name__ == "__main__":
    main()