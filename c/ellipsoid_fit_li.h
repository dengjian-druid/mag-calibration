#ifndef ELLIPSOID_FIT_LI_H
#define ELLIPSOID_FIT_LI_H

#include "arm_math.h"
#include <stdint.h>
#include <stdbool.h>

// 配置参数
#define MAX_SAMPLES 1000        // 最大样本数量
#define MATRIX_SIZE_6x6 36      // 6x6矩阵元素数量
#define MATRIX_SIZE_4x4 16      // 4x4矩阵元素数量
#define MATRIX_SIZE_3x3 9       // 3x3矩阵元素数量
#define VECTOR_SIZE_6 6         // 6维向量
#define VECTOR_SIZE_4 4         // 4维向量
#define VECTOR_SIZE_3 3         // 3维向量

// 椭球参数结构体
typedef struct {
    float32_t M[MATRIX_SIZE_3x3];   // 3x3 椭球矩阵 M
    float32_t n[VECTOR_SIZE_3];     // 3x1 向量 n
    float32_t d;                    // 标量 d
} ellipsoid_params_t;

// 校准参数结构体
typedef struct {
    float32_t S[MATRIX_SIZE_3x3];   // 3x3 校准矩阵 S
    float32_t h[VECTOR_SIZE_3];     // 3x1 偏移向量 h
} calibration_params_t;

// 工作缓冲区结构体 - 用于减少栈内存使用
typedef struct {
    // 中间计算矩阵
    float32_t D[10 * MAX_SAMPLES];  // 10xN 设计矩阵
    float32_t S[10 * 10];           // 10x10 矩阵 S = D*D^T
    float32_t S_11[MATRIX_SIZE_6x6]; // 6x6 子矩阵
    float32_t S_12[6 * 4];          // 6x4 子矩阵
    float32_t S_22[MATRIX_SIZE_4x4]; // 4x4 子矩阵
    float32_t C[MATRIX_SIZE_6x6];   // 6x6 约束矩阵
    float32_t E[MATRIX_SIZE_6x6];   // 6x6 特征值问题矩阵
    
    // 临时矩阵和向量
    float32_t temp_6x6[MATRIX_SIZE_6x6];
    float32_t temp_6x4[6 * 4];
    float32_t temp_4x4[MATRIX_SIZE_4x4];
    float32_t temp_4x6[4 * 6];
    float32_t temp_6x1[VECTOR_SIZE_6];
    float32_t temp_4x1[VECTOR_SIZE_4];
    float32_t temp_3x3[MATRIX_SIZE_3x3];
    float32_t temp_3x1[VECTOR_SIZE_3];
    
    // 特征值和特征向量
    float32_t eigenvalues[VECTOR_SIZE_6];
    float32_t eigenvectors[MATRIX_SIZE_6x6];
    
    // CMSIS-DSP 矩阵实例
    arm_matrix_instance_f32 mat_D;
    arm_matrix_instance_f32 mat_S;
    arm_matrix_instance_f32 mat_S_11;
    arm_matrix_instance_f32 mat_S_12;
    arm_matrix_instance_f32 mat_S_22;
    arm_matrix_instance_f32 mat_C;
    arm_matrix_instance_f32 mat_E;
    arm_matrix_instance_f32 mat_temp_6x6;
    arm_matrix_instance_f32 mat_temp_6x4;
    arm_matrix_instance_f32 mat_temp_4x4;
    arm_matrix_instance_f32 mat_temp_4x6;
    arm_matrix_instance_f32 mat_temp_3x3;
} ellipsoid_work_buffer_t;

// 函数声明

/**
 * @brief 使用Li算法进行椭球拟合
 * @param samples 输入样本数据 (3xN格式: x, y, z)
 * @param num_samples 样本数量
 * @param params 输出椭球参数
 * @param work_buffer 工作缓冲区
 * @return ARM_MATH_SUCCESS 成功, 其他值表示错误
 */
arm_status ellipsoid_fit_li(const float32_t *samples, 
                           uint32_t num_samples,
                           ellipsoid_params_t *params,
                           ellipsoid_work_buffer_t *work_buffer);

/**
 * @brief 从椭球参数计算校准参数
 * @param params 椭球参数
 * @param F 目标磁场强度 (默认500.0)
 * @param calib 输出校准参数
 * @param work_buffer 工作缓冲区
 * @return ARM_MATH_SUCCESS 成功, 其他值表示错误
 */
arm_status ellipsoid_to_calibration(const ellipsoid_params_t *params,
                                   float32_t F,
                                   calibration_params_t *calib,
                                   ellipsoid_work_buffer_t *work_buffer);

/**
 * @brief 完整的磁力计校准流程
 * @param samples 输入样本数据 (3xN格式: x, y, z)
 * @param num_samples 样本数量
 * @param F 目标磁场强度
 * @param calib 输出校准参数
 * @param work_buffer 工作缓冲区
 * @return ARM_MATH_SUCCESS 成功, 其他值表示错误
 */
arm_status magnetometer_calibrate_li(const float32_t *samples,
                                    uint32_t num_samples,
                                    float32_t F,
                                    calibration_params_t *calib,
                                    ellipsoid_work_buffer_t *work_buffer);

/**
 * @brief 应用校准参数到原始磁力计数据
 * @param raw_data 原始数据 [x, y, z]
 * @param calib 校准参数
 * @param calibrated_data 校准后数据 [x, y, z]
 */
void apply_calibration(const float32_t raw_data[3],
                      const calibration_params_t *calib,
                      float32_t calibrated_data[3]);

/**
 * @brief 初始化工作缓冲区
 * @param work_buffer 工作缓冲区指针
 */
void init_work_buffer(ellipsoid_work_buffer_t *work_buffer);

#endif // ELLIPSOID_FIT_LI_H