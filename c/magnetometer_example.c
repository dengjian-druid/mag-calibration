/**
 * @file magnetometer_example.c
 * @brief nRF52840磁力计校准示例程序
 * 
 * 本示例展示如何在nRF52840上使用Li算法进行磁力计校准
 * 需要链接CMSIS-DSP库
 */

#include "ellipsoid_fit_li.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// 模拟磁力计数据读取函数
static void read_magnetometer(float32_t *x, float32_t *y, float32_t *z) {
    // 在实际应用中，这里应该是从I2C/SPI读取磁力计数据
    // 这里使用模拟数据作为示例
    static uint32_t counter = 0;
    float32_t angle = counter * 0.1f;
    
    // 模拟椭球形磁场数据（带偏移和缩放）
    *x = 150.0f * cosf(angle) + 100.0f;  // 偏移100
    *y = 200.0f * sinf(angle) - 50.0f;   // 偏移-50
    *z = 180.0f * sinf(2.0f * angle) + 20.0f; // 偏移20
    
    counter++;
}

// 打印校准参数
static void print_calibration_params(const calibration_params_t *calib) {
    printf("\n=== 校准参数 ===\n");
    printf("偏移向量 h:\n");
    printf("  h[0] = %.6f\n", calib->h[0]);
    printf("  h[1] = %.6f\n", calib->h[1]);
    printf("  h[2] = %.6f\n", calib->h[2]);
    
    printf("\n变换矩阵 S:\n");
    for (int i = 0; i < 3; i++) {
        printf("  [");
        for (int j = 0; j < 3; j++) {
            printf("%8.6f", calib->S[i * 3 + j]);
            if (j < 2) printf(", ");
        }
        printf("]\n");
    }
}

// 打印C代码格式的校准参数
static void print_c_code_format(const calibration_params_t *calib) {
    printf("\n=== C代码格式 ===\n");
    printf("// 磁力计校准参数\n");
    printf("static const float32_t mag_offset[3] = {\n");
    printf("    %.8ff, %.8ff, %.8ff\n", calib->h[0], calib->h[1], calib->h[2]);
    printf("};\n\n");
    
    printf("static const float32_t mag_transform[9] = {\n");
    for (int i = 0; i < 3; i++) {
        printf("    ");
        for (int j = 0; j < 3; j++) {
            printf("%.8ff", calib->S[i * 3 + j]);
            if (i < 2 || j < 2) printf(", ");
        }
        printf("\n");
    }
    printf("};\n");
}

// 计算校准效果统计
static void calculate_calibration_stats(const float32_t *samples, uint32_t num_samples,
                                       const calibration_params_t *calib) {
    float32_t sum_radius = 0.0f;
    float32_t sum_radius_sq = 0.0f;
    float32_t min_radius = 1e6f;
    float32_t max_radius = 0.0f;
    
    for (uint32_t i = 0; i < num_samples; i++) {
        float32_t raw[3] = {
            samples[i * 3 + 0],
            samples[i * 3 + 1], 
            samples[i * 3 + 2]
        };
        
        float32_t calibrated[3];
        apply_calibration(raw, calib, calibrated);
        
        // 计算半径
        float32_t radius = sqrtf(calibrated[0]*calibrated[0] + 
                                calibrated[1]*calibrated[1] + 
                                calibrated[2]*calibrated[2]);
        
        sum_radius += radius;
        sum_radius_sq += radius * radius;
        
        if (radius < min_radius) min_radius = radius;
        if (radius > max_radius) max_radius = radius;
    }
    
    float32_t mean_radius = sum_radius / num_samples;
    float32_t variance = (sum_radius_sq / num_samples) - (mean_radius * mean_radius);
    float32_t std_radius = sqrtf(variance);
    
    printf("\n=== 校准效果统计 ===\n");
    printf("样本数量: %lu\n", (unsigned long)num_samples);
    printf("校准后半径统计:\n");
    printf("  均值: %.3f\n", mean_radius);
    printf("  标准差: %.3f\n", std_radius);
    printf("  最小值: %.3f\n", min_radius);
    printf("  最大值: %.3f\n", max_radius);
    printf("  变异系数: %.3f%%\n", (std_radius / mean_radius) * 100.0f);
}

int main(void) {
    printf("nRF52840 磁力计校准示例\n");
    printf("使用Li算法和CMSIS-DSP库\n");
    printf("========================\n");
    
    // 分配工作缓冲区（在实际应用中可能放在静态内存中）
    static ellipsoid_work_buffer_t work_buffer;
    static calibration_params_t calib_params;
    
    // 初始化工作缓冲区
    init_work_buffer(&work_buffer);
    
    // 收集磁力计数据
    const uint32_t num_samples = 500;
    static float32_t samples[500 * 3];  // 静态分配以节省栈空间
    
    printf("正在收集 %lu 个磁力计样本...\n", (unsigned long)num_samples);
    
    for (uint32_t i = 0; i < num_samples; i++) {
        float32_t x, y, z;
        read_magnetometer(&x, &y, &z);
        
        samples[i * 3 + 0] = x;
        samples[i * 3 + 1] = y;
        samples[i * 3 + 2] = z;
        
        // 每100个样本打印一次进度
        if ((i + 1) % 100 == 0) {
            printf("已收集 %lu/%lu 样本\n", (unsigned long)(i + 1), (unsigned long)num_samples);
        }
    }
    
    printf("数据收集完成！\n");
    
    // 执行校准
    printf("\n开始执行Li算法校准...\n");
    
    arm_status status = magnetometer_calibrate_li(samples, num_samples, 500.0f, 
                                                 &calib_params, &work_buffer);
    
    if (status == ARM_MATH_SUCCESS) {
        printf("校准成功！\n");
        
        // 打印校准参数
        print_calibration_params(&calib_params);
        print_c_code_format(&calib_params);
        
        // 计算校准效果
        calculate_calibration_stats(samples, num_samples, &calib_params);
        
        // 演示实时校准应用
        printf("\n=== 实时校准演示 ===\n");
        for (int i = 0; i < 5; i++) {
            float32_t raw[3], calibrated[3];
            read_magnetometer(&raw[0], &raw[1], &raw[2]);
            apply_calibration(raw, &calib_params, calibrated);
            
            float32_t raw_magnitude = sqrtf(raw[0]*raw[0] + raw[1]*raw[1] + raw[2]*raw[2]);
            float32_t cal_magnitude = sqrtf(calibrated[0]*calibrated[0] + 
                                           calibrated[1]*calibrated[1] + 
                                           calibrated[2]*calibrated[2]);
            
            printf("原始: [%7.2f, %7.2f, %7.2f] |%.2f| -> ", 
                   raw[0], raw[1], raw[2], raw_magnitude);
            printf("校准: [%7.2f, %7.2f, %7.2f] |%.2f|\n", 
                   calibrated[0], calibrated[1], calibrated[2], cal_magnitude);
        }
        
    } else {
        printf("校准失败！错误代码: %d\n", status);
        return -1;
    }
    
    printf("\n校准完成！\n");
    return 0;
}

// 内存使用情况报告
void print_memory_usage(void) {
    printf("\n=== 内存使用情况 ===\n");
    printf("ellipsoid_work_buffer_t: %zu bytes\n", sizeof(ellipsoid_work_buffer_t));
    printf("calibration_params_t: %zu bytes\n", sizeof(calibration_params_t));
    printf("ellipsoid_params_t: %zu bytes\n", sizeof(ellipsoid_params_t));
    printf("样本数据 (500点): %zu bytes\n", 500 * 3 * sizeof(float32_t));
    printf("总计约: %zu bytes\n", 
           sizeof(ellipsoid_work_buffer_t) + sizeof(calibration_params_t) + 
           sizeof(ellipsoid_params_t) + 500 * 3 * sizeof(float32_t));
}