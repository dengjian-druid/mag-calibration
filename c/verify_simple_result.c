/**
 * @file verify_simple_result.c
 * @brief 验证简化版本Li算法的结果正确性
 * 
 * 通过分析测试数据的生成参数和校准结果，
 * 计算理论期望值并与实际输出进行对比
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

// 复制简化版本的数据结构
typedef struct {
    float center[3];
    float radii[3];
    float rotation[9];
} ellipsoid_params_simple_t;

typedef struct {
    float h[3];
    float S[9];
} calibration_params_simple_t;

// 生成与简化版本相同的测试数据
void generate_test_data(float *samples, int num_samples) {
    for (int i = 0; i < num_samples; i++) {
        // 椭球参数（与简化版本相同）
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

// 计算数据的统计信息
void calculate_data_statistics(const float *samples, int num_samples) {
    printf("=== 测试数据统计分析 ===\n");
    
    // 计算均值
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
    
    // 计算方差和标准差
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
    printf("数据标准差: [%.3f, %.3f, %.3f]\n", sqrtf(var[0]), sqrtf(var[1]), sqrtf(var[2]));
    
    // 计算数据范围
    float min_val[3] = {1e6f, 1e6f, 1e6f};
    float max_val[3] = {-1e6f, -1e6f, -1e6f};
    
    for (int i = 0; i < num_samples; i++) {
        for (int j = 0; j < 3; j++) {
            float val = samples[i*3 + j];
            if (val < min_val[j]) min_val[j] = val;
            if (val > max_val[j]) max_val[j] = val;
        }
    }
    
    printf("数据范围:\n");
    printf("  X: [%.3f, %.3f], 范围: %.3f\n", min_val[0], max_val[0], max_val[0] - min_val[0]);
    printf("  Y: [%.3f, %.3f], 范围: %.3f\n", min_val[1], max_val[1], max_val[1] - min_val[1]);
    printf("  Z: [%.3f, %.3f], 范围: %.3f\n", min_val[2], max_val[2], max_val[2] - min_val[2]);
}

// 计算理论椭球参数
void calculate_theoretical_parameters() {
    printf("\n=== 理论椭球参数 ===\n");
    
    // 椭球生成参数
    float a = 150.0f, b = 200.0f, c = 180.0f;
    float offset_x = 100.0f, offset_y = -50.0f, offset_z = 20.0f;
    
    printf("椭球半径: a=%.1f, b=%.1f, c=%.1f\n", a, b, c);
    printf("椭球中心: [%.1f, %.1f, %.1f]\n", offset_x, offset_y, offset_z);
    
    // 计算平均半径
    float avg_radius = (a + b + c) / 3.0f;
    printf("平均半径: %.3f\n", avg_radius);
    
    // 期望的校准参数
    printf("\n期望的校准参数:\n");
    printf("偏移向量 h: [%.1f, %.1f, %.1f]\n", offset_x, offset_y, offset_z);
    
    // 期望的缩放因子（将椭球归一化到期望磁场强度）
    float expected_field = 500.0f;
    printf("期望磁场强度: %.1f\n", expected_field);
    printf("期望缩放因子: X=%.3f, Y=%.3f, Z=%.3f\n", 
           expected_field/a, expected_field/b, expected_field/c);
}

// 分析简化版本的结果
void analyze_simple_result() {
    printf("\n=== 简化版本结果分析 ===\n");
    
    // 简化版本的实际输出（从测试结果复制）
    float h_actual[3] = {100.000f, -50.000f, -8158.491f};
    float S_actual[9] = {
        5.000f, 0.000f, 0.000f,
        0.000f, 5.000f, 0.000f,
        0.000f, 0.000f, 5.000f
    };
    
    printf("实际偏移向量: [%.3f, %.3f, %.3f]\n", h_actual[0], h_actual[1], h_actual[2]);
    printf("实际变换矩阵对角元素: [%.3f, %.3f, %.3f]\n", S_actual[0], S_actual[4], S_actual[8]);
    
    // 分析偏移向量的准确性
    float h_expected[3] = {100.0f, -50.0f, 20.0f};
    printf("\n偏移向量误差分析:\n");
    printf("  X轴: 期望=%.1f, 实际=%.3f, 误差=%.3f\n", 
           h_expected[0], h_actual[0], fabsf(h_actual[0] - h_expected[0]));
    printf("  Y轴: 期望=%.1f, 实际=%.3f, 误差=%.3f\n", 
           h_expected[1], h_actual[1], fabsf(h_actual[1] - h_expected[1]));
    printf("  Z轴: 期望=%.1f, 实际=%.3f, 误差=%.3f (严重偏差!)\n", 
           h_expected[2], h_actual[2], fabsf(h_actual[2] - h_expected[2]));
    
    // 分析缩放因子
    float expected_scale[3] = {500.0f/150.0f, 500.0f/200.0f, 500.0f/180.0f};
    printf("\n缩放因子误差分析:\n");
    printf("  X轴: 期望=%.3f, 实际=%.3f, 误差=%.3f\n", 
           expected_scale[0], S_actual[0], fabsf(S_actual[0] - expected_scale[0]));
    printf("  Y轴: 期望=%.3f, 实际=%.3f, 误差=%.3f\n", 
           expected_scale[1], S_actual[4], fabsf(S_actual[4] - expected_scale[1]));
    printf("  Z轴: 期望=%.3f, 实际=%.3f, 误差=%.3f\n", 
           expected_scale[2], S_actual[8], fabsf(S_actual[8] - expected_scale[2]));
}

// 验证校准效果
void verify_calibration_effect() {
    printf("\n=== 校准效果验证 ===\n");
    
    // 生成测试数据
    const int num_samples = 100;
    float *samples = malloc(num_samples * 3 * sizeof(float));
    generate_test_data(samples, num_samples);
    
    // 简化版本的校准参数
    float h[3] = {100.000f, -50.000f, -8158.491f};
    float S[9] = {
        5.000f, 0.000f, 0.000f,
        0.000f, 5.000f, 0.000f,
        0.000f, 0.000f, 5.000f
    };
    
    // 应用校准并计算统计
    float sum_radius = 0.0f;
    float sum_radius_sq = 0.0f;
    float min_radius = 1e6f;
    float max_radius = 0.0f;
    
    printf("前10个样本的校准效果:\n");
    for (int i = 0; i < 10; i++) {
        float raw[3] = {samples[i*3], samples[i*3+1], samples[i*3+2]};
        
        // 应用校准: calibrated = S * (raw - h)
        float temp[3] = {
            raw[0] - h[0],
            raw[1] - h[1],
            raw[2] - h[2]
        };
        
        float calibrated[3] = {
            S[0] * temp[0],
            S[4] * temp[1],
            S[8] * temp[2]
        };
        
        float radius = sqrtf(calibrated[0]*calibrated[0] + 
                            calibrated[1]*calibrated[1] + 
                            calibrated[2]*calibrated[2]);
        
        sum_radius += radius;
        sum_radius_sq += radius * radius;
        
        if (radius < min_radius) min_radius = radius;
        if (radius > max_radius) max_radius = radius;
        
        printf("  样本%d: 原始[%6.1f,%6.1f,%6.1f] -> 校准[%7.1f,%7.1f,%7.1f] |%.1f|\n",
               i, raw[0], raw[1], raw[2], 
               calibrated[0], calibrated[1], calibrated[2], radius);
    }
    
    // 计算所有样本的统计
    for (int i = 10; i < num_samples; i++) {
        float raw[3] = {samples[i*3], samples[i*3+1], samples[i*3+2]};
        
        float temp[3] = {
            raw[0] - h[0],
            raw[1] - h[1],
            raw[2] - h[2]
        };
        
        float calibrated[3] = {
            S[0] * temp[0],
            S[4] * temp[1],
            S[8] * temp[2]
        };
        
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
    float std_radius = sqrtf(variance);
    
    printf("\n校准后半径统计 (所有%d个样本):\n", num_samples);
    printf("  均值: %.3f\n", mean_radius);
    printf("  标准差: %.3f\n", std_radius);
    printf("  最小值: %.3f\n", min_radius);
    printf("  最大值: %.3f\n", max_radius);
    printf("  变异系数: %.3f%%\n", (std_radius / mean_radius) * 100.0f);
    printf("  期望值: 500.0\n");
    printf("  误差: %.3f (%.1f%%)\n", fabsf(mean_radius - 500.0f), 
           fabsf(mean_radius - 500.0f) / 500.0f * 100.0f);
    
    free(samples);
}

int main(void) {
    printf("简化版本Li算法结果验证\n");
    printf("========================\n");
    
    // 生成测试数据并分析
    const int num_samples = 100;
    float *samples = malloc(num_samples * 3 * sizeof(float));
    generate_test_data(samples, num_samples);
    
    calculate_data_statistics(samples, num_samples);
    calculate_theoretical_parameters();
    analyze_simple_result();
    verify_calibration_effect();
    
    printf("\n=== 结论 ===\n");
    printf("1. X和Y轴的偏移检测准确\n");
    printf("2. Z轴偏移存在严重误差 (-8158 vs 20)\n");
    printf("3. 缩放因子偏差较大 (5.0 vs 期望的2.78-3.33)\n");
    printf("4. 校准后半径远大于期望值 (40000+ vs 500)\n");
    printf("5. 简化版本的特征值分解和椭球参数提取需要改进\n");
    
    free(samples);
    return 0;
}