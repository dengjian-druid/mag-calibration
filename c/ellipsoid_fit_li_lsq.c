/**
 * @file ellipsoid_fit_li_lsq.c
 * @brief 基于最小二乘法的Li算法实现
 * 
 * 使用标准的最小二乘法求解椭球方程，避免复杂的特征值分解
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define MAX_SAMPLES 1000
#define TOLERANCE 1e-8
#define MIN_VALUE 1e-12

// 椭球参数结构
typedef struct {
    float center[3];     // 椭球中心
    float radii[3];      // 椭球半径
} ellipsoid_params_t;

// 校准参数结构
typedef struct {
    float h[3];          // 偏移向量
    float S[9];          // 变换矩阵 (3x3)
} calibration_params_t;

// 安全函数
float safe_sqrt(float x) {
    return sqrtf(fmaxf(x, MIN_VALUE));
}

float safe_divide(float a, float b) {
    if (fabsf(b) < MIN_VALUE) {
        return 0.0f;
    }
    return a / b;
}

// 高斯消元法求解线性方程组 Ax = b
int solve_linear_system(float *A, float *b, float *x, int n) {
    // 前向消元
    for (int i = 0; i < n; i++) {
        // 寻找主元
        int max_row = i;
        for (int k = i + 1; k < n; k++) {
            if (fabsf(A[k*n + i]) > fabsf(A[max_row*n + i])) {
                max_row = k;
            }
        }
        
        // 交换行
        if (max_row != i) {
            for (int k = 0; k < n; k++) {
                float temp = A[i*n + k];
                A[i*n + k] = A[max_row*n + k];
                A[max_row*n + k] = temp;
            }
            float temp = b[i];
            b[i] = b[max_row];
            b[max_row] = temp;
        }
        
        // 检查主元是否为零
        if (fabsf(A[i*n + i]) < MIN_VALUE) {
            return -1; // 矩阵奇异
        }
        
        // 消元
        for (int k = i + 1; k < n; k++) {
            float factor = A[k*n + i] / A[i*n + i];
            for (int j = i; j < n; j++) {
                A[k*n + j] -= factor * A[i*n + j];
            }
            b[k] -= factor * b[i];
        }
    }
    
    // 回代
    for (int i = n - 1; i >= 0; i--) {
        x[i] = b[i];
        for (int j = i + 1; j < n; j++) {
            x[i] -= A[i*n + j] * x[j];
        }
        x[i] /= A[i*n + i];
    }
    
    return 0;
}

// 使用最小二乘法拟合椭球
int ellipsoid_fit_lsq(const float *samples, int num_samples, ellipsoid_params_t *params) {
    if (num_samples < 6) {
        printf("错误：样本数量不足，至少需要6个样本\n");
        return -1;
    }
    
    printf("使用%d个样本进行椭球拟合\n", num_samples);
    
    // 椭球的一般方程: Ax^2 + By^2 + Cz^2 + Dx + Ey + Fz = 1
    // 我们要求解参数 [A, B, C, D, E, F]
    
    // 构建设计矩阵 H 和观测向量 f
    float *H = malloc(num_samples * 6 * sizeof(float));
    float *f = malloc(num_samples * sizeof(float));
    
    if (!H || !f) {
        free(H);
        free(f);
        return -1;
    }
    
    for (int i = 0; i < num_samples; i++) {
        float x = samples[i*3 + 0];
        float y = samples[i*3 + 1];
        float z = samples[i*3 + 2];
        
        H[i*6 + 0] = x * x;
        H[i*6 + 1] = y * y;
        H[i*6 + 2] = z * z;
        H[i*6 + 3] = x;
        H[i*6 + 4] = y;
        H[i*6 + 5] = z;
        
        f[i] = 1.0f;
    }
    
    // 计算法方程 H^T * H * theta = H^T * f
    float HTH[36]; // 6x6 矩阵
    float HTf[6];  // 6x1 向量
    
    memset(HTH, 0, 36 * sizeof(float));
    memset(HTf, 0, 6 * sizeof(float));
    
    // 计算 H^T * H
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            for (int k = 0; k < num_samples; k++) {
                HTH[i*6 + j] += H[k*6 + i] * H[k*6 + j];
            }
        }
    }
    
    // 计算 H^T * f
    for (int i = 0; i < 6; i++) {
        for (int k = 0; k < num_samples; k++) {
            HTf[i] += H[k*6 + i] * f[k];
        }
    }
    
    // 求解线性方程组
    float theta[6];
    if (solve_linear_system(HTH, HTf, theta, 6) != 0) {
        printf("求解线性方程组失败\n");
        free(H);
        free(f);
        return -1;
    }
    
    printf("椭球方程系数: A=%.6f, B=%.6f, C=%.6f, D=%.6f, E=%.6f, F=%.6f\n",
           theta[0], theta[1], theta[2], theta[3], theta[4], theta[5]);
    
    // 从椭球方程系数提取中心和半径
    // 椭球方程: Ax^2 + By^2 + Cz^2 + Dx + Ey + Fz = 1
    // 标准形式: A(x-h)^2 + B(y-k)^2 + C(z-l)^2 = 1 + Ah^2 + Bk^2 + Cl^2
    
    float A = theta[0], B = theta[1], C = theta[2];
    float D = theta[3], E = theta[4], F = theta[5];
    
    // 检查系数的有效性
    if (A <= MIN_VALUE || B <= MIN_VALUE || C <= MIN_VALUE) {
        printf("无效的椭球系数\n");
        free(H);
        free(f);
        return -1;
    }
    
    // 计算椭球中心
    params->center[0] = -D / (2.0f * A);
    params->center[1] = -E / (2.0f * B);
    params->center[2] = -F / (2.0f * C);
    
    // 计算椭球半径
    float center_term = A * params->center[0] * params->center[0] + 
                       B * params->center[1] * params->center[1] + 
                       C * params->center[2] * params->center[2];
    
    float scale_factor = 1.0f + center_term;
    
    if (scale_factor <= MIN_VALUE) {
        printf("无效的椭球尺度因子\n");
        free(H);
        free(f);
        return -1;
    }
    
    params->radii[0] = safe_sqrt(scale_factor / A);
    params->radii[1] = safe_sqrt(scale_factor / B);
    params->radii[2] = safe_sqrt(scale_factor / C);
    
    printf("椭球中心: [%.3f, %.3f, %.3f]\n", 
           params->center[0], params->center[1], params->center[2]);
    printf("椭球半径: [%.3f, %.3f, %.3f]\n", 
           params->radii[0], params->radii[1], params->radii[2]);
    
    free(H);
    free(f);
    return 0;
}

// 将椭球参数转换为校准参数
int ellipsoid_to_calibration_lsq(const ellipsoid_params_t *ellipsoid, 
                                float target_field_strength,
                                calibration_params_t *calibration) {
    // 偏移向量就是椭球中心
    calibration->h[0] = ellipsoid->center[0];
    calibration->h[1] = ellipsoid->center[1];
    calibration->h[2] = ellipsoid->center[2];
    
    // 计算缩放因子
    float scale_x = safe_divide(target_field_strength, ellipsoid->radii[0]);
    float scale_y = safe_divide(target_field_strength, ellipsoid->radii[1]);
    float scale_z = safe_divide(target_field_strength, ellipsoid->radii[2]);
    
    printf("计算的缩放因子: [%.3f, %.3f, %.3f]\n", scale_x, scale_y, scale_z);
    
    // 构建对角变换矩阵
    memset(calibration->S, 0, 9 * sizeof(float));
    calibration->S[0] = scale_x;
    calibration->S[4] = scale_y;
    calibration->S[8] = scale_z;
    
    return 0;
}

// 应用校准
void apply_calibration_lsq(const float *raw_data, const calibration_params_t *calibration, 
                          float *calibrated_data) {
    // 减去偏移
    float temp[3];
    temp[0] = raw_data[0] - calibration->h[0];
    temp[1] = raw_data[1] - calibration->h[1];
    temp[2] = raw_data[2] - calibration->h[2];
    
    // 应用对角变换矩阵
    calibrated_data[0] = calibration->S[0] * temp[0];
    calibrated_data[1] = calibration->S[4] * temp[1];
    calibrated_data[2] = calibration->S[8] * temp[2];
}

// 完整的校准流程
int magnetometer_calibrate_lsq(const float *samples, int num_samples, 
                              float target_field_strength,
                              calibration_params_t *calibration) {
    ellipsoid_params_t ellipsoid;
    
    if (ellipsoid_fit_lsq(samples, num_samples, &ellipsoid) != 0) {
        return -1;
    }
    
    return ellipsoid_to_calibration_lsq(&ellipsoid, target_field_strength, calibration);
}

// 生成测试数据
void generate_test_data_lsq(float *samples, int num_samples) {
    for (int i = 0; i < num_samples; i++) {
        // 椭球参数
        float a = 150.0f, b = 200.0f, c = 180.0f;
        float offset_x = 100.0f, offset_y = -50.0f, offset_z = 20.0f;
        
        // 使用参数方程生成椭球面上的点
        float u = 2.0f * M_PI * i / num_samples;
        float v_param = M_PI * (i % 10) / 10.0f - M_PI/2.0f;
        
        samples[i*3 + 0] = a * cosf(v_param) * cosf(u) + offset_x;
        samples[i*3 + 1] = b * cosf(v_param) * sinf(u) + offset_y;
        samples[i*3 + 2] = c * sinf(v_param) + offset_z;
    }
}

// 计算校准统计信息
void calculate_calibration_stats_lsq(const float *samples, int num_samples, 
                                    const calibration_params_t *calibration,
                                    float target_field) {
    float sum_radius = 0.0f;
    float sum_radius_sq = 0.0f;
    float min_radius = 1e6f;
    float max_radius = 0.0f;
    
    printf("\n前10个样本的校准效果:\n");
    for (int i = 0; i < fminf(10, num_samples); i++) {
        float raw[3] = {samples[i*3], samples[i*3+1], samples[i*3+2]};
        float calibrated[3];
        
        apply_calibration_lsq(raw, calibration, calibrated);
        
        float radius = sqrtf(calibrated[0]*calibrated[0] + 
                            calibrated[1]*calibrated[1] + 
                            calibrated[2]*calibrated[2]);
        
        printf("  样本%d: 原始[%6.1f,%6.1f,%6.1f] -> 校准[%7.1f,%7.1f,%7.1f] |%.1f|\n",
               i, raw[0], raw[1], raw[2], 
               calibrated[0], calibrated[1], calibrated[2], radius);
    }
    
    // 计算所有样本的统计
    for (int i = 0; i < num_samples; i++) {
        float raw[3] = {samples[i*3], samples[i*3+1], samples[i*3+2]};
        float calibrated[3];
        
        apply_calibration_lsq(raw, calibration, calibrated);
        
        float radius = sqrtf(calibrated[0]*calibrated[0] + 
                            calibrated[1]*calibrated[1] + 
                            calibrated[2]*calibrated[2]);
        
        sum_radius += radius;
        sum_radius_sq += radius * radius;
        
        if (radius < min_radius) min_radius = radius;
        if (radius > max_radius) max_radius = radius;
    }
    
    float mean_radius = sum_radius / num_samples;
    float variance = (sum_radius_sq / num_samples) - (mean_radius * mean_radius);
    float std_radius = safe_sqrt(variance);
    
    printf("\n=== 校准效果统计 ===\n");
    printf("平均半径: %.3f\n", mean_radius);
    printf("标准差: %.3f\n", std_radius);
    printf("最小半径: %.3f\n", min_radius);
    printf("最大半径: %.3f\n", max_radius);
    printf("变异系数: %.2f%%\n", (std_radius / mean_radius) * 100.0f);
    printf("期望半径: %.1f\n", target_field);
    printf("绝对误差: %.3f\n", fabsf(mean_radius - target_field));
    printf("相对误差: %.2f%%\n", fabsf(mean_radius - target_field) / target_field * 100.0f);
}

// 测试函数
void test_lsq_algorithm() {
    printf("=== 最小二乘法Li算法测试 ===\n");
    
    const int num_samples = 100;
    float *samples = malloc(num_samples * 3 * sizeof(float));
    
    // 生成测试数据
    generate_test_data_lsq(samples, num_samples);
    printf("生成了%d个测试样本\n", num_samples);
    
    // 执行校准
    calibration_params_t calibration;
    float target_field = 500.0f;
    
    printf("\n=== 椭球拟合过程 ===\n");
    if (magnetometer_calibrate_lsq(samples, num_samples, target_field, &calibration) == 0) {
        printf("\n=== 校准结果 ===\n");
        printf("偏移向量 h: [%.3f, %.3f, %.3f]\n", 
               calibration.h[0], calibration.h[1], calibration.h[2]);
        printf("变换矩阵 S (对角元素): [%.3f, %.3f, %.3f]\n", 
               calibration.S[0], calibration.S[4], calibration.S[8]);
        
        // 与期望值比较
        printf("\n=== 与期望值比较 ===\n");
        printf("期望偏移: [100.0, -50.0, 20.0]\n");
        printf("实际偏移: [%.1f, %.1f, %.1f]\n", 
               calibration.h[0], calibration.h[1], calibration.h[2]);
        printf("偏移误差: [%.1f, %.1f, %.1f]\n", 
               fabsf(calibration.h[0] - 100.0f), 
               fabsf(calibration.h[1] + 50.0f), 
               fabsf(calibration.h[2] - 20.0f));
        
        float expected_scale[3] = {500.0f/150.0f, 500.0f/200.0f, 500.0f/180.0f};
        printf("期望缩放: [%.3f, %.3f, %.3f]\n", 
               expected_scale[0], expected_scale[1], expected_scale[2]);
        printf("实际缩放: [%.3f, %.3f, %.3f]\n", 
               calibration.S[0], calibration.S[4], calibration.S[8]);
        
        // 计算校准统计
        calculate_calibration_stats_lsq(samples, num_samples, &calibration, target_field);
        
    } else {
        printf("校准失败！\n");
    }
    
    free(samples);
}

int main(void) {
    test_lsq_algorithm();
    return 0;
}