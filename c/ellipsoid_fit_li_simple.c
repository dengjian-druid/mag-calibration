/**
 * @file ellipsoid_fit_li_simple.c
 * @brief Li算法的简化版本，不依赖CMSIS-DSP库
 * 
 * 这个版本使用标准C库实现，主要用于:
 * 1. 算法验证和测试
 * 2. 在没有CMSIS-DSP的环境中使用
 * 3. 理解算法原理
 * 
 * 注意: 这个版本的性能不如CMSIS-DSP优化版本
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

// 简化的数据结构
typedef struct {
    float center[3];
    float radii[3];
    float rotation[9];  // 3x3 rotation matrix
} ellipsoid_params_simple_t;

typedef struct {
    float h[3];         // offset vector
    float S[9];         // 3x3 transformation matrix
} calibration_params_simple_t;

// 矩阵操作函数
static void matrix_multiply_3x3(const float *A, const float *B, float *C) {
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            C[i*3 + j] = 0.0f;
            for (int k = 0; k < 3; k++) {
                C[i*3 + j] += A[i*3 + k] * B[k*3 + j];
            }
        }
    }
}

static void matrix_transpose_3x3(const float *A, float *AT) {
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            AT[j*3 + i] = A[i*3 + j];
        }
    }
}

static void matrix_vector_multiply_3x3(const float *A, const float *v, float *result) {
    for (int i = 0; i < 3; i++) {
        result[i] = 0.0f;
        for (int j = 0; j < 3; j++) {
            result[i] += A[i*3 + j] * v[j];
        }
    }
}

// 简化的特征值分解 (仅用于3x3对称矩阵)
static int eigen_decomposition_3x3_symmetric(const float *A, float *eigenvalues, float *eigenvectors) {
    // 这是一个简化的实现，使用雅可比方法
    // 在实际应用中，建议使用更稳定的算法
    
    // 复制输入矩阵到工作矩阵
    float work[9];
    memcpy(work, A, 9 * sizeof(float));
    
    // 初始化特征向量为单位矩阵
    memset(eigenvectors, 0, 9 * sizeof(float));
    eigenvectors[0] = eigenvectors[4] = eigenvectors[8] = 1.0f;
    
    const int max_iterations = 50;
    const float tolerance = 1e-6f;
    
    for (int iter = 0; iter < max_iterations; iter++) {
        // 找到最大的非对角元素
        float max_off_diag = 0.0f;
        int p = 0, q = 1;
        
        for (int i = 0; i < 3; i++) {
            for (int j = i + 1; j < 3; j++) {
                float val = fabsf(work[i*3 + j]);
                if (val > max_off_diag) {
                    max_off_diag = val;
                    p = i;
                    q = j;
                }
            }
        }
        
        if (max_off_diag < tolerance) {
            break;  // 收敛
        }
        
        // 计算旋转角度
        float theta;
        if (fabsf(work[p*3 + p] - work[q*3 + q]) < 1e-10f) {
            theta = M_PI / 4.0f;
        } else {
            theta = 0.5f * atanf(2.0f * work[p*3 + q] / (work[p*3 + p] - work[q*3 + q]));
        }
        
        float c = cosf(theta);
        float s = sinf(theta);
        
        // 应用Givens旋转
        float temp_pp = work[p*3 + p];
        float temp_qq = work[q*3 + q];
        float temp_pq = work[p*3 + q];
        
        work[p*3 + p] = c*c*temp_pp + s*s*temp_qq - 2*c*s*temp_pq;
        work[q*3 + q] = s*s*temp_pp + c*c*temp_qq + 2*c*s*temp_pq;
        work[p*3 + q] = work[q*3 + p] = 0.0f;
        
        // 更新其他元素
        for (int i = 0; i < 3; i++) {
            if (i != p && i != q) {
                float temp_ip = work[i*3 + p];
                float temp_iq = work[i*3 + q];
                work[i*3 + p] = work[p*3 + i] = c*temp_ip - s*temp_iq;
                work[i*3 + q] = work[q*3 + i] = s*temp_ip + c*temp_iq;
            }
        }
        
        // 更新特征向量
        for (int i = 0; i < 3; i++) {
            float temp_ip = eigenvectors[i*3 + p];
            float temp_iq = eigenvectors[i*3 + q];
            eigenvectors[i*3 + p] = c*temp_ip - s*temp_iq;
            eigenvectors[i*3 + q] = s*temp_ip + c*temp_iq;
        }
    }
    
    // 提取特征值
    eigenvalues[0] = work[0];
    eigenvalues[1] = work[4];
    eigenvalues[2] = work[8];
    
    return 0;  // 成功
}

// 简化的矩阵平方根
static int matrix_sqrt_3x3(const float *A, float *sqrt_A) {
    float eigenvalues[3];
    float eigenvectors[9];
    
    // 特征值分解
    if (eigen_decomposition_3x3_symmetric(A, eigenvalues, eigenvectors) != 0) {
        return -1;
    }
    
    // 检查特征值是否为正
    for (int i = 0; i < 3; i++) {
        if (eigenvalues[i] <= 0.0f) {
            eigenvalues[i] = 1e-9f;  // 避免负数或零
        }
    }
    
    // 计算特征值的平方根
    float sqrt_eigenvalues[3];
    for (int i = 0; i < 3; i++) {
        sqrt_eigenvalues[i] = sqrtf(eigenvalues[i]);
    }
    
    // 重构矩阵: sqrt_A = V * diag(sqrt(λ)) * V^T
    float V_T[9];
    matrix_transpose_3x3(eigenvectors, V_T);
    
    float temp[9] = {0};
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            temp[i*3 + j] = eigenvectors[i*3 + j] * sqrt_eigenvalues[j];
        }
    }
    
    matrix_multiply_3x3(temp, V_T, sqrt_A);
    
    return 0;
}

// Li算法的简化实现
int ellipsoid_fit_li_simple(const float *samples, int num_samples, 
                           ellipsoid_params_simple_t *ellipsoid) {
    if (num_samples < 9) {
        printf("错误: 样本数量不足 (需要至少9个)\n");
        return -1;
    }
    
    // 构建设计矩阵 D (n x 9)
    float *D = malloc(num_samples * 9 * sizeof(float));
    if (!D) {
        printf("错误: 内存分配失败\n");
        return -1;
    }
    
    for (int i = 0; i < num_samples; i++) {
        float x = samples[i*3 + 0];
        float y = samples[i*3 + 1];
        float z = samples[i*3 + 2];
        
        D[i*9 + 0] = x*x;
        D[i*9 + 1] = y*y;
        D[i*9 + 2] = z*z;
        D[i*9 + 3] = 2*x*y;
        D[i*9 + 4] = 2*x*z;
        D[i*9 + 5] = 2*y*z;
        D[i*9 + 6] = 2*x;
        D[i*9 + 7] = 2*y;
        D[i*9 + 8] = 2*z;
    }
    
    // 构建约束矩阵 C (9 x 9)
    float C[81] = {0};
    C[0*9 + 1] = C[1*9 + 0] = -1;  // C[0,1] = C[1,0] = -1
    C[0*9 + 2] = C[2*9 + 0] = -1;  // C[0,2] = C[2,0] = -1
    C[1*9 + 2] = C[2*9 + 1] = -1;  // C[1,2] = C[2,1] = -1
    C[3*9 + 3] = -4;               // C[3,3] = -4
    C[4*9 + 4] = -4;               // C[4,4] = -4
    C[5*9 + 5] = -4;               // C[5,5] = -4
    
    // 计算 S = D^T * D (9 x 9)
    float S[81] = {0};
    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 9; j++) {
            for (int k = 0; k < num_samples; k++) {
                S[i*9 + j] += D[k*9 + i] * D[k*9 + j];
            }
        }
    }
    
    // 这里应该解广义特征值问题 S*v = λ*C*v
    // 为简化，我们使用一个近似方法
    // 在实际应用中，建议使用更精确的算法
    
    // 简化处理：使用基于数据的启发式方法估计椭球参数
    // 计算数据的中心和范围
    float mean[3] = {0, 0, 0};
    for (int i = 0; i < num_samples; i++) {
        mean[0] += samples[i*3 + 0];
        mean[1] += samples[i*3 + 1];
        mean[2] += samples[i*3 + 2];
    }
    mean[0] /= num_samples;
    mean[1] /= num_samples;
    mean[2] /= num_samples;
    
    // 计算方差
    float var[3] = {0, 0, 0};
    for (int i = 0; i < num_samples; i++) {
        float dx = samples[i*3 + 0] - mean[0];
        float dy = samples[i*3 + 1] - mean[1];
        float dz = samples[i*3 + 2] - mean[2];
        var[0] += dx * dx;
        var[1] += dy * dy;
        var[2] += dz * dz;
    }
    var[0] /= num_samples;
    var[1] /= num_samples;
    var[2] /= num_samples;
    
    // 构建近似的椭球系数
    float v[9];
    v[0] = 1.0f / (var[0] + 1e-6f);  // x^2 coefficient
    v[1] = 1.0f / (var[1] + 1e-6f);  // y^2 coefficient
    v[2] = 1.0f / (var[2] + 1e-6f);  // z^2 coefficient
    v[3] = 0.0f;  // xy coefficient
    v[4] = 0.0f;  // xz coefficient
    v[5] = 0.0f;  // yz coefficient
    v[6] = -2.0f * mean[0] * v[0];  // x coefficient
    v[7] = -2.0f * mean[1] * v[1];  // y coefficient
    v[8] = -2.0f * mean[2] * v[2];  // z coefficient
    
    // 添加常数项以形成完整的椭球方程
    float constant = mean[0]*mean[0]*v[0] + mean[1]*mean[1]*v[1] + mean[2]*mean[2]*v[2] - 1.0f;
    
    // 将常数项设为v[8]的一部分（这里简化处理）
    v[8] = v[8] + constant;
    
    // 从系数向量v提取椭球参数
    float A_matrix[9] = {
        v[0], v[3], v[4],
        v[3], v[1], v[5],
        v[4], v[5], v[2]
    };
    
    float b_vector[3] = {v[6], v[7], v[8]};
    float c = v[8];  // 常数项
    
    // 计算椭球中心: center = -0.5 * A^(-1) * b
    // 简化处理：假设A为对角矩阵
    if (fabsf(A_matrix[0]) > 1e-10f) {
        ellipsoid->center[0] = -b_vector[0] / (2.0f * A_matrix[0]);
    } else {
        ellipsoid->center[0] = 0.0f;
    }
    
    if (fabsf(A_matrix[4]) > 1e-10f) {
        ellipsoid->center[1] = -b_vector[1] / (2.0f * A_matrix[4]);
    } else {
        ellipsoid->center[1] = 0.0f;
    }
    
    if (fabsf(A_matrix[8]) > 1e-10f) {
        ellipsoid->center[2] = -b_vector[2] / (2.0f * A_matrix[8]);
    } else {
        ellipsoid->center[2] = 0.0f;
    }
    
    // 计算椭球半径
    // 对于椭球方程 (x-h)^T * A * (x-h) = 1
    // 半径为 1/sqrt(eigenvalue)
    float center_term = A_matrix[0] * ellipsoid->center[0] * ellipsoid->center[0] +
                       A_matrix[4] * ellipsoid->center[1] * ellipsoid->center[1] +
                       A_matrix[8] * ellipsoid->center[2] * ellipsoid->center[2] +
                       b_vector[0] * ellipsoid->center[0] +
                       b_vector[1] * ellipsoid->center[1] +
                       b_vector[2] * ellipsoid->center[2] + c;
    
    if (fabsf(A_matrix[0]) > 1e-10f && center_term > 0) {
        ellipsoid->radii[0] = sqrtf(center_term / A_matrix[0]);
    } else {
        ellipsoid->radii[0] = 100.0f;  // 默认值
    }
    
    if (fabsf(A_matrix[4]) > 1e-10f && center_term > 0) {
        ellipsoid->radii[1] = sqrtf(center_term / A_matrix[4]);
    } else {
        ellipsoid->radii[1] = 100.0f;  // 默认值
    }
    
    if (fabsf(A_matrix[8]) > 1e-10f && center_term > 0) {
        ellipsoid->radii[2] = sqrtf(center_term / A_matrix[8]);
    } else {
        ellipsoid->radii[2] = 100.0f;  // 默认值
    }
    
    // 简化的旋转矩阵（单位矩阵）
    memset(ellipsoid->rotation, 0, 9 * sizeof(float));
    ellipsoid->rotation[0] = ellipsoid->rotation[4] = ellipsoid->rotation[8] = 1.0f;
    
    free(D);
    return 0;
}

// 将椭球参数转换为校准参数
int ellipsoid_to_calibration_simple(const ellipsoid_params_simple_t *ellipsoid,
                                   float expected_field_strength,
                                   calibration_params_simple_t *calib) {
    // 偏移向量就是椭球中心
    calib->h[0] = ellipsoid->center[0];
    calib->h[1] = ellipsoid->center[1];
    calib->h[2] = ellipsoid->center[2];
    
    // 构建缩放矩阵
    float scale_matrix[9] = {0};
    
    // 计算平均半径作为参考
    float avg_radius = (ellipsoid->radii[0] + ellipsoid->radii[1] + ellipsoid->radii[2]) / 3.0f;
    
    if (avg_radius > 1e-6f) {
        // 缩放因子应该将椭球半径归一化到期望的磁场强度
        float scale_factor = expected_field_strength / avg_radius;
        scale_matrix[0] = scale_factor * avg_radius / ellipsoid->radii[0];
        scale_matrix[4] = scale_factor * avg_radius / ellipsoid->radii[1];
        scale_matrix[8] = scale_factor * avg_radius / ellipsoid->radii[2];
    } else {
        // 默认单位矩阵
        scale_matrix[0] = scale_matrix[4] = scale_matrix[8] = 1.0f;
    }
    
    // S = R * Scale * R^T (简化版本，假设R为单位矩阵)
    memcpy(calib->S, scale_matrix, 9 * sizeof(float));
    
    return 0;
}

// 应用校准
void apply_calibration_simple(const float raw[3], 
                             const calibration_params_simple_t *calib,
                             float calibrated[3]) {
    // 减去偏移
    float temp[3];
    temp[0] = raw[0] - calib->h[0];
    temp[1] = raw[1] - calib->h[1];
    temp[2] = raw[2] - calib->h[2];
    
    // 应用变换矩阵
    matrix_vector_multiply_3x3(calib->S, temp, calibrated);
}

// 完整的校准流程
int magnetometer_calibrate_simple(const float *samples, int num_samples,
                                 float expected_field_strength,
                                 calibration_params_simple_t *calib) {
    ellipsoid_params_simple_t ellipsoid;
    
    // 椭球拟合
    if (ellipsoid_fit_li_simple(samples, num_samples, &ellipsoid) != 0) {
        printf("椭球拟合失败\n");
        return -1;
    }
    
    // 转换为校准参数
    if (ellipsoid_to_calibration_simple(&ellipsoid, expected_field_strength, calib) != 0) {
        printf("校准参数计算失败\n");
        return -1;
    }
    
    return 0;
}

// 测试函数
int main(void) {
    printf("Li算法简化版本测试\n");
    printf("==================\n");
    
    // 生成测试数据
    const int num_samples = 100;
    float *samples = malloc(num_samples * 3 * sizeof(float));
    
    if (!samples) {
        printf("内存分配失败\n");
        return -1;
    }
    
    // 生成椭球形数据
    for (int i = 0; i < num_samples; i++) {
        float angle1 = 2.0f * M_PI * i / num_samples;
        float angle2 = M_PI * (i % 20) / 20.0f;
        
        // 生成椭球形数据（更简单的椭球）
        float a = 150.0f, b = 200.0f, c = 180.0f;
        float offset_x = 100.0f, offset_y = -50.0f, offset_z = 20.0f;
        
        // 使用参数方程生成椭球面上的点
        float u = 2.0f * M_PI * i / num_samples;
        float v_param = M_PI * (i % 10) / 10.0f - M_PI/2.0f;
        
        samples[i*3 + 0] = a * cosf(v_param) * cosf(u) + offset_x;
        samples[i*3 + 1] = b * cosf(v_param) * sinf(u) + offset_y;
        samples[i*3 + 2] = c * sinf(v_param) + offset_z;
    }
    
    printf("生成了 %d 个测试样本\n", num_samples);
    
    // 执行校准
    calibration_params_simple_t calib;
    int result = magnetometer_calibrate_simple(samples, num_samples, 500.0f, &calib);
    
    if (result == 0) {
        printf("\n校准成功！\n");
        
        printf("\n偏移向量 h:\n");
        printf("  h = [%.3f, %.3f, %.3f]\n", calib.h[0], calib.h[1], calib.h[2]);
        
        printf("\n变换矩阵 S:\n");
        for (int i = 0; i < 3; i++) {
            printf("  [");
            for (int j = 0; j < 3; j++) {
                printf("%8.3f", calib.S[i*3 + j]);
                if (j < 2) printf(", ");
            }
            printf("]\n");
        }
        
        // 测试校准效果
        printf("\n校准效果测试:\n");
        float sum_radius = 0.0f;
        for (int i = 0; i < 10; i++) {
            float raw[3] = {samples[i*3], samples[i*3+1], samples[i*3+2]};
            float calibrated[3];
            apply_calibration_simple(raw, &calib, calibrated);
            
            float radius = sqrtf(calibrated[0]*calibrated[0] + 
                                calibrated[1]*calibrated[1] + 
                                calibrated[2]*calibrated[2]);
            sum_radius += radius;
            
            printf("  样本%d: 原始[%6.1f,%6.1f,%6.1f] -> 校准[%6.1f,%6.1f,%6.1f] |%.1f|\n",
                   i, raw[0], raw[1], raw[2], 
                   calibrated[0], calibrated[1], calibrated[2], radius);
        }
        
        printf("\n平均半径: %.3f\n", sum_radius / 10.0f);
        
    } else {
        printf("校准失败！\n");
    }
    
    free(samples);
    return 0;
}