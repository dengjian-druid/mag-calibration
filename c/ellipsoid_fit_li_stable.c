/**
 * @file ellipsoid_fit_li_stable.c
 * @brief 稳定版本的Li算法实现
 * 
 * 采用更稳定的数值方法，避免NaN和数值不稳定问题
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define MAX_SAMPLES 1000
#define TOLERANCE 1e-6
#define MIN_VALUE 1e-10

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

// 安全的平方根函数
float safe_sqrt(float x) {
    return sqrtf(fmaxf(x, MIN_VALUE));
}

// 安全的除法函数
float safe_divide(float a, float b) {
    if (fabsf(b) < MIN_VALUE) {
        return (a >= 0) ? 1e6f : -1e6f;
    }
    return a / b;
}

// 使用最小二乘法的简化椭球拟合
int ellipsoid_fit_stable(const float *samples, int num_samples, ellipsoid_params_t *params) {
    if (num_samples < 9) {
        printf("错误：样本数量不足，至少需要9个样本\n");
        return -1;
    }
    
    // 计算数据的均值（作为椭球中心的初始估计）
    float mean[3] = {0, 0, 0};
    for (int i = 0; i < num_samples; i++) {
        mean[0] += samples[i*3 + 0];
        mean[1] += samples[i*3 + 1];
        mean[2] += samples[i*3 + 2];
    }
    mean[0] /= num_samples;
    mean[1] /= num_samples;
    mean[2] /= num_samples;
    
    printf("数据均值: [%.3f, %.3f, %.3f]\n", mean[0], mean[1], mean[2]);
    
    // 计算协方差矩阵的对角元素（简化方法）
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
    
    printf("数据方差: [%.3f, %.3f, %.3f]\n", var[0], var[1], var[2]);
    
    // 椭球中心就是数据均值
    params->center[0] = mean[0];
    params->center[1] = mean[1];
    params->center[2] = mean[2];
    
    // 椭球半径基于标准差
    params->radii[0] = safe_sqrt(var[0]);
    params->radii[1] = safe_sqrt(var[1]);
    params->radii[2] = safe_sqrt(var[2]);
    
    printf("计算的椭球中心: [%.3f, %.3f, %.3f]\n", 
           params->center[0], params->center[1], params->center[2]);
    printf("计算的椭球半径: [%.3f, %.3f, %.3f]\n", 
           params->radii[0], params->radii[1], params->radii[2]);
    
    return 0;
}

// 改进的椭球拟合（使用迭代方法）
int ellipsoid_fit_iterative(const float *samples, int num_samples, ellipsoid_params_t *params) {
    if (num_samples < 9) {
        return -1;
    }
    
    // 初始估计：使用数据的均值和标准差
    ellipsoid_fit_stable(samples, num_samples, params);
    
    // 迭代改进椭球参数
    for (int iter = 0; iter < 10; iter++) {
        float new_center[3] = {0, 0, 0};
        float new_radii[3] = {0, 0, 0};
        float weight_sum = 0;
        
        // 计算加权中心
        for (int i = 0; i < num_samples; i++) {
            float x = samples[i*3 + 0];
            float y = samples[i*3 + 1];
            float z = samples[i*3 + 2];
            
            // 计算到当前椭球中心的距离
            float dx = x - params->center[0];
            float dy = y - params->center[1];
            float dz = z - params->center[2];
            
            float dist = sqrtf(dx*dx + dy*dy + dz*dz);
            float weight = 1.0f / (1.0f + dist * 0.01f); // 距离权重
            
            new_center[0] += weight * x;
            new_center[1] += weight * y;
            new_center[2] += weight * z;
            weight_sum += weight;
        }
        
        if (weight_sum > MIN_VALUE) {
            new_center[0] /= weight_sum;
            new_center[1] /= weight_sum;
            new_center[2] /= weight_sum;
        }
        
        // 计算新的半径
        float sum_sq[3] = {0, 0, 0};
        for (int i = 0; i < num_samples; i++) {
            float dx = samples[i*3 + 0] - new_center[0];
            float dy = samples[i*3 + 1] - new_center[1];
            float dz = samples[i*3 + 2] - new_center[2];
            
            sum_sq[0] += dx * dx;
            sum_sq[1] += dy * dy;
            sum_sq[2] += dz * dz;
        }
        
        new_radii[0] = safe_sqrt(sum_sq[0] / num_samples);
        new_radii[1] = safe_sqrt(sum_sq[1] / num_samples);
        new_radii[2] = safe_sqrt(sum_sq[2] / num_samples);
        
        // 检查收敛
        float center_change = fabsf(new_center[0] - params->center[0]) + 
                             fabsf(new_center[1] - params->center[1]) + 
                             fabsf(new_center[2] - params->center[2]);
        
        // 更新参数
        memcpy(params->center, new_center, 3 * sizeof(float));
        memcpy(params->radii, new_radii, 3 * sizeof(float));
        
        if (center_change < TOLERANCE) {
            printf("迭代收敛于第%d次\n", iter + 1);
            break;
        }
    }
    
    return 0;
}

// 将椭球参数转换为校准参数
int ellipsoid_to_calibration_stable(const ellipsoid_params_t *ellipsoid, 
                                   float target_field_strength,
                                   calibration_params_t *calibration) {
    // 偏移向量就是椭球中心
    calibration->h[0] = ellipsoid->center[0];
    calibration->h[1] = ellipsoid->center[1];
    calibration->h[2] = ellipsoid->center[2];
    
    // 计算缩放因子，确保数值稳定
    float scale_x = safe_divide(target_field_strength, ellipsoid->radii[0]);
    float scale_y = safe_divide(target_field_strength, ellipsoid->radii[1]);
    float scale_z = safe_divide(target_field_strength, ellipsoid->radii[2]);
    
    // 限制缩放因子的范围，避免极端值
    scale_x = fmaxf(0.1f, fminf(scale_x, 10.0f));
    scale_y = fmaxf(0.1f, fminf(scale_y, 10.0f));
    scale_z = fmaxf(0.1f, fminf(scale_z, 10.0f));
    
    printf("计算的缩放因子: [%.3f, %.3f, %.3f]\n", scale_x, scale_y, scale_z);
    
    // 构建对角变换矩阵
    memset(calibration->S, 0, 9 * sizeof(float));
    calibration->S[0] = scale_x;
    calibration->S[4] = scale_y;
    calibration->S[8] = scale_z;
    
    return 0;
}

// 应用校准
void apply_calibration_stable(const float *raw_data, const calibration_params_t *calibration, 
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
int magnetometer_calibrate_stable(const float *samples, int num_samples, 
                                float target_field_strength,
                                calibration_params_t *calibration) {
    ellipsoid_params_t ellipsoid;
    
    if (ellipsoid_fit_iterative(samples, num_samples, &ellipsoid) != 0) {
        return -1;
    }
    
    return ellipsoid_to_calibration_stable(&ellipsoid, target_field_strength, calibration);
}

// 生成测试数据
void generate_test_data_stable(float *samples, int num_samples) {
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
void calculate_calibration_stats(const float *samples, int num_samples, 
                               const calibration_params_t *calibration,
                               float target_field) {
    float sum_radius = 0.0f;
    float sum_radius_sq = 0.0f;
    float min_radius = 1e6f;
    float max_radius = 0.0f;
    
    for (int i = 0; i < num_samples; i++) {
        float raw[3] = {samples[i*3], samples[i*3+1], samples[i*3+2]};
        float calibrated[3];
        
        apply_calibration_stable(raw, calibration, calibrated);
        
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
void test_stable_algorithm() {
    printf("=== 稳定版Li算法测试 ===\n");
    
    const int num_samples = 100;
    float *samples = malloc(num_samples * 3 * sizeof(float));
    
    // 生成测试数据
    generate_test_data_stable(samples, num_samples);
    printf("生成了%d个测试样本\n", num_samples);
    
    // 执行校准
    calibration_params_t calibration;
    float target_field = 500.0f;
    
    printf("\n=== 椭球拟合过程 ===\n");
    if (magnetometer_calibrate_stable(samples, num_samples, target_field, &calibration) == 0) {
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
        calculate_calibration_stats(samples, num_samples, &calibration, target_field);
        
    } else {
        printf("校准失败！\n");
    }
    
    free(samples);
}

int main(void) {
    test_stable_algorithm();
    return 0;
}