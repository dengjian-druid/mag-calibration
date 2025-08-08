import sys
import argparse
import numpy as np
import pandas as pd
from scipy import linalg as la
from plot_utils import create_calibration_plot
from sklearn.model_selection import train_test_split

def ellipsoid_fit(s):
    ''' Estimate ellipsoid parameters from a set of points.
        Parameters
        ----------
         s : array_like
           The samples (M,N) where M=3 (x,y,z) and N=number of samples.

         Returns
        -------
        M, n, d : array_like, array_like, float
          The ellipsoid parameters M, n, d.

         References
         ----------
           [1] Qingde Li; Griffiths, J.G., "Least squares ellipsoid specific
           fitting," in Geometric Modeling and Processing, 2004.
           Proceedings, vol., no., pp.335-340, 2004
    '''

    # D (samples)
    D = np.array([s[0]**2., s[1]**2., s[2]**2.,
                  2.*s[1]*s[2], 2.*s[0]*s[2], 2.*s[0]*s[1],
                  2.*s[0], 2.*s[1], 2.*s[2], np.ones_like(s[0])])

    # S, S_11, S_12, S_21, S_22 (eq. 11)
    S = np.dot(D, D.T)
    S_11 = S[:6,:6]
    S_12 = S[:6,6:]
    S_22 = S[6:,6:]

    # C (Eq. 8, k=4)
    C = np.array([[-1,  1,  1,  0,  0,  0],
                  [ 1, -1,  1,  0,  0,  0],
                  [ 1,  1, -1,  0,  0,  0],
                  [ 0,  0,  0, -4,  0,  0],
                  [ 0,  0,  0,  0, -4,  0],
                  [ 0,  0,  0,  0,  0, -4]])

    # v_1 (eq. 15, solution)
    E = np.dot(la.inv(C),
                S_11 - np.dot(S_12, np.dot(la.inv(S_22), S_12.T)))

    E_w, E_v = la.eig(E)

    v_1 = E_v[:, np.argmax(E_w)]
    if v_1[0] < 0:
      v_1 = -v_1

    # v_2 (eq. 13, solution)
    v_2 = np.dot(np.dot(-la.inv(S_22), S_12.T), v_1)

    # quadric-form parameters
    M = np.array([[v_1[0], v_1[5], v_1[4]],
                  [v_1[5], v_1[1], v_1[3]],
                 [v_1[4], v_1[3], v_1[2]]])
    n = np.array([[v_2[0]],
                [v_2[1]],
                [v_2[2]]])
    d = v_2[3]

    return M, n, d

def ellipsoid_fit_v0(s):
    '''
    Estimate ellipsoid parameters from a set of points using Li's method.

    Parameters
    ----------
    s : array_like
      The samples (M,N) where M=3 (x,y,z) and N=number of samples.

    Returns
    -------
    M, n, d : array_like, array_like, float
      The ellipsoid parameters M, n, d.

    References
    ----------
    .. [1] Qingde Li; Griffiths, J.G., "Least squares ellipsoid specific
       fitting," in Geometric Modeling and Processing, 2004.
       Proceedings, vol., no., pp.335-340, 2004
    '''

    # D (samples)
    D = np.array([s[0]**2., s[1]**2., s[2]**2.,
                  2.*s[1]*s[2], 2.*s[0]*s[2], 2.*s[0]*s[1],
                  2.*s[0], 2.*s[1], 2.*s[2], np.ones_like(s[0])])

    # S, S_11, S_12, S_21, S_22 (eq. 11)
    S = np.dot(D, D.T)
    S_11 = S[:6,:6]
    S_12 = S[:6,6:]
    S_21 = S[6:,:6]
    S_22 = S[6:,6:]

    # C (Eq. 8, k=4)
    C = np.array([[-1,  1,  1,  0,  0,  0],
                  [ 1, -1,  1,  0,  0,  0],
                  [ 1,  1, -1,  0,  0,  0],
                  [ 0,  0,  0, -4,  0,  0],
                  [ 0,  0,  0,  0, -4,  0],
                  [ 0,  0,  0,  0,  0, -4]])

    # v_1 (eq. 15, solution)
    E = np.dot(la.inv(C),
               S_11 - np.dot(S_12, np.dot(la.inv(S_22), S_21)))

    E_w, E_v = la.eig(E)

    v_1 = E_v[:, np.argmax(E_w)]
    if v_1[0] < 0: v_1 = -v_1

    # v_2 (eq. 13, solution)
    v_2 = np.dot(np.dot(-la.inv(S_22), S_21), v_1)

    # quadric-form parameters
    M = np.array([[v_1[0], v_1[3], v_1[4]],
                  [v_1[3], v_1[1], v_1[5]],
                  [v_1[4], v_1[5], v_1[2]]])
    n = np.array([[v_2[0]],
                  [v_2[1]],
                  [v_2[2]]])
    d = v_2[3]

    return M, n, d

def calibrate(s, F=500.):
    # 确保输入数据是(3, N)的形状，如果是(N, 3)则转置
    if s.shape[0] != 3:
        s = s.T
    M, n, d = ellipsoid_fit(s)
    M_1 = la.inv(M)
    h = -np.dot(M_1, n).flatten()
    S = np.real(F / np.sqrt(np.dot(n.T, np.dot(M_1, n)) - d) * la.sqrtm(M))
    return S, h



def main():
    """Li算法椭球拟合示例"""
    parser = argparse.ArgumentParser(description='磁力计椭球拟合校准程序')
    parser.add_argument('-f', '--file', default='../data/1d000006ab_mag_20250218.csv', 
                       help='CSV文件路径 (默认: ../data/1d000006ab_mag_20250218.csv)')
    parser.add_argument('-n', '--train_samples', type=int, default=200,
                       help='训练样本数量 (默认: 200)')
    parser.add_argument('-t', '--target_field', type=float, default=500.0,
                       help='目标磁场强度 (默认: 500.0)')
    
    args = parser.parse_args()
    
    print("=== Python磁力计校准程序 ===")
    print(f"CSV文件: {args.file}")
    print(f"训练样本数量: {args.train_samples}")
    print(f"目标磁场强度: {args.target_field}")
    
    # 读取数据
    df = pd.read_csv(args.file)[100:-100]
    data = df[['X', 'Y', 'Z']].values
    total_samples = len(data)
    
    print(f"从 {args.file} 读取了 {total_samples} 个样本")
    
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
    original_radius = np.linalg.norm(train, axis=1).mean()
    print(f"原始数据平均半径: {original_radius:.1f}")

    print("\n=== 开始椭球拟合校准 ===")
    # 使用校准API
    S, h = calibrate(train, F=args.target_field)
    
    # print("\n=== 校准参数 ===")
    # print(f"偏移向量 h: [{h[0]:.3f}, {h[1]:.3f}, {h[2]:.3f}]")
    # print(f"缩放矩阵 S:")
    # for i, row in enumerate(S):
    #     print(f"  [{row[0]:.6f}, {row[1]:.6f}, {row[2]:.6f}]")
    
    # 输出校准参数的C代码格式
    print("\n=== C代码格式校准参数 ===")
    print("const float mag_offset[3] = {" + f"{h[0]:.3f}f, {h[1]:.3f}f, {h[2]:.3f}f" + "};")
    print("const float mag_scale[3][3] = {")
    for row in S:
        print("    { " + ", ".join(f"{v:.6f}f" for v in row) + " },")
    print("};")

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
        create_calibration_plot(data, test_calibrated, "Li Algorithm", 'li_calibration_result.png')
    else:
        print("\n注意: 没有测试集数据")
        # 可视化训练集结果
        create_calibration_plot(data, train_calibrated, "Li Algorithm", 'li_calibration_result.png')

if __name__ == "__main__":
    main()
