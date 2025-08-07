import sys
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
    if len(sys.argv) < 2:
        print("用法: python ellipsoid_fit_li.py <地磁数据.csv>")
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
    create_calibration_plot(data, calibrated, "Li Algorithm", 'li_calibration_result.png')

if __name__ == "__main__":
    main()
