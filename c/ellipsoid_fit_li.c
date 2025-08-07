#include "ellipsoid_fit_li.h"
#include <math.h>
#include <string.h>

// 内部辅助函数声明
static arm_status build_design_matrix(const float32_t *samples, uint32_t num_samples, 
                                     ellipsoid_work_buffer_t *work_buffer);
static arm_status solve_eigenvalue_problem(ellipsoid_work_buffer_t *work_buffer, 
                                          float32_t *v1, float32_t *v2);
static arm_status find_max_eigenvalue_index(const float32_t *eigenvalues, uint32_t size, uint32_t *max_idx);
static arm_status matrix_sqrt(const float32_t *input, float32_t *output, ellipsoid_work_buffer_t *work_buffer);

void init_work_buffer(ellipsoid_work_buffer_t *work_buffer) {
    // 清零所有缓冲区
    memset(work_buffer, 0, sizeof(ellipsoid_work_buffer_t));
    
    // 初始化约束矩阵 C (Eq. 8, k=4)
    float32_t C_data[36] = {
        -1,  1,  1,  0,  0,  0,
         1, -1,  1,  0,  0,  0,
         1,  1, -1,  0,  0,  0,
         0,  0,  0, -4,  0,  0,
         0,  0,  0,  0, -4,  0,
         0,  0,  0,  0,  0, -4
    };
    memcpy(work_buffer->C, C_data, sizeof(C_data));
    
    // 初始化矩阵实例
    arm_mat_init_f32(&work_buffer->mat_C, 6, 6, work_buffer->C);
    arm_mat_init_f32(&work_buffer->mat_S_11, 6, 6, work_buffer->S_11);
    arm_mat_init_f32(&work_buffer->mat_S_12, 6, 4, work_buffer->S_12);
    arm_mat_init_f32(&work_buffer->mat_S_22, 4, 4, work_buffer->S_22);
    arm_mat_init_f32(&work_buffer->mat_E, 6, 6, work_buffer->E);
    arm_mat_init_f32(&work_buffer->mat_temp_6x6, 6, 6, work_buffer->temp_6x6);
    arm_mat_init_f32(&work_buffer->mat_temp_6x4, 6, 4, work_buffer->temp_6x4);
    arm_mat_init_f32(&work_buffer->mat_temp_4x4, 4, 4, work_buffer->temp_4x4);
    arm_mat_init_f32(&work_buffer->mat_temp_4x6, 4, 6, work_buffer->temp_4x6);
    arm_mat_init_f32(&work_buffer->mat_temp_3x3, 3, 3, work_buffer->temp_3x3);
}

static arm_status build_design_matrix(const float32_t *samples, uint32_t num_samples, 
                                     ellipsoid_work_buffer_t *work_buffer) {
    if (num_samples > MAX_SAMPLES) {
        return ARM_MATH_SIZE_MISMATCH;
    }
    
    // 构建设计矩阵 D (10 x N)
    // D = [x^2, y^2, z^2, 2*y*z, 2*x*z, 2*x*y, 2*x, 2*y, 2*z, 1]
    for (uint32_t i = 0; i < num_samples; i++) {
        float32_t x = samples[i * 3 + 0];
        float32_t y = samples[i * 3 + 1];
        float32_t z = samples[i * 3 + 2];
        
        work_buffer->D[0 * num_samples + i] = x * x;           // x^2
        work_buffer->D[1 * num_samples + i] = y * y;           // y^2
        work_buffer->D[2 * num_samples + i] = z * z;           // z^2
        work_buffer->D[3 * num_samples + i] = 2.0f * y * z;    // 2*y*z
        work_buffer->D[4 * num_samples + i] = 2.0f * x * z;    // 2*x*z
        work_buffer->D[5 * num_samples + i] = 2.0f * x * y;    // 2*x*y
        work_buffer->D[6 * num_samples + i] = 2.0f * x;        // 2*x
        work_buffer->D[7 * num_samples + i] = 2.0f * y;        // 2*y
        work_buffer->D[8 * num_samples + i] = 2.0f * z;        // 2*z
        work_buffer->D[9 * num_samples + i] = 1.0f;            // 1
    }
    
    // 初始化设计矩阵实例
    arm_mat_init_f32(&work_buffer->mat_D, 10, num_samples, work_buffer->D);
    
    return ARM_MATH_SUCCESS;
}

static arm_status solve_eigenvalue_problem(ellipsoid_work_buffer_t *work_buffer, 
                                          float32_t *v1, float32_t *v2) {
    arm_status status;
    
    // 计算 S = D * D^T (10x10)
    arm_matrix_instance_f32 mat_D_T;
    arm_mat_init_f32(&mat_D_T, work_buffer->mat_D.numCols, work_buffer->mat_D.numRows, work_buffer->D);
    arm_mat_init_f32(&work_buffer->mat_S, 10, 10, work_buffer->S);
    
    status = arm_mat_mult_f32(&work_buffer->mat_D, &mat_D_T, &work_buffer->mat_S);
    if (status != ARM_MATH_SUCCESS) return status;
    
    // 提取子矩阵 S_11 (6x6), S_12 (6x4), S_22 (4x4)
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            work_buffer->S_11[i * 6 + j] = work_buffer->S[i * 10 + j];
        }
        for (int j = 0; j < 4; j++) {
            work_buffer->S_12[i * 4 + j] = work_buffer->S[i * 10 + (j + 6)];
        }
    }
    
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            work_buffer->S_22[i * 4 + j] = work_buffer->S[(i + 6) * 10 + (j + 6)];
        }
    }
    
    // 计算 S_22^(-1)
    status = arm_mat_inverse_f32(&work_buffer->mat_S_22, &work_buffer->mat_temp_4x4);
    if (status != ARM_MATH_SUCCESS) return status;
    
    // 计算 S_12 * S_22^(-1)
    status = arm_mat_mult_f32(&work_buffer->mat_S_12, &work_buffer->mat_temp_4x4, &work_buffer->mat_temp_6x4);
    if (status != ARM_MATH_SUCCESS) return status;
    
    // 计算 S_12 * S_22^(-1) * S_12^T
    arm_matrix_instance_f32 mat_S_12_T;
    arm_mat_init_f32(&mat_S_12_T, 4, 6, work_buffer->S_12);
    status = arm_mat_mult_f32(&work_buffer->mat_temp_6x4, &mat_S_12_T, &work_buffer->mat_temp_6x6);
    if (status != ARM_MATH_SUCCESS) return status;
    
    // 计算 S_11 - S_12 * S_22^(-1) * S_12^T
    status = arm_mat_sub_f32(&work_buffer->mat_S_11, &work_buffer->mat_temp_6x6, &work_buffer->mat_temp_6x6);
    if (status != ARM_MATH_SUCCESS) return status;
    
    // 计算 C^(-1)
    arm_matrix_instance_f32 mat_C_inv;
    arm_mat_init_f32(&mat_C_inv, 6, 6, work_buffer->temp_6x6);
    status = arm_mat_inverse_f32(&work_buffer->mat_C, &mat_C_inv);
    if (status != ARM_MATH_SUCCESS) return status;
    
    // 计算 E = C^(-1) * (S_11 - S_12 * S_22^(-1) * S_12^T)
    status = arm_mat_mult_f32(&mat_C_inv, &work_buffer->mat_temp_6x6, &work_buffer->mat_E);
    if (status != ARM_MATH_SUCCESS) return status;
    
    // 注意: CMSIS-DSP 没有直接的特征值分解函数
    // 这里需要使用简化的幂迭代法或者其他数值方法
    // 为了简化，我们使用一个近似方法
    
    // 简化的特征值求解 - 使用幂迭代法找最大特征值对应的特征向量
    float32_t *E_data = work_buffer->E;
    float32_t *v = work_buffer->temp_6x1;
    
    // 初始化随机向量
    v[0] = 1.0f; v[1] = 0.0f; v[2] = 0.0f; v[3] = 0.0f; v[4] = 0.0f; v[5] = 0.0f;
    
    // 幂迭代法
    for (int iter = 0; iter < 50; iter++) {
        // v_new = E * v
        for (int i = 0; i < 6; i++) {
            work_buffer->temp_6x1[i] = 0.0f;
            for (int j = 0; j < 6; j++) {
                work_buffer->temp_6x1[i] += E_data[i * 6 + j] * v[j];
            }
        }
        
        // 归一化
        float32_t norm = 0.0f;
        arm_dot_prod_f32(work_buffer->temp_6x1, work_buffer->temp_6x1, 6, &norm);
        norm = sqrtf(norm);
        
        if (norm > 1e-10f) {
            arm_scale_f32(work_buffer->temp_6x1, 1.0f / norm, v, 6);
        }
    }
    
    // 确保 v1[0] > 0
    if (v[0] < 0.0f) {
        arm_scale_f32(v, -1.0f, v, 6);
    }
    
    // 复制 v1
    memcpy(v1, v, 6 * sizeof(float32_t));
    
    // 计算 v2 = -S_22^(-1) * S_12^T * v1
    arm_matrix_instance_f32 mat_v1, mat_v2;
    arm_mat_init_f32(&mat_v1, 6, 1, v1);
    arm_mat_init_f32(&mat_v2, 4, 1, v2);
    
    // S_12^T * v1
    status = arm_mat_mult_f32(&mat_S_12_T, &mat_v1, &work_buffer->mat_temp_4x1);
    if (status != ARM_MATH_SUCCESS) return status;
    
    // -S_22^(-1) * (S_12^T * v1)
    arm_matrix_instance_f32 mat_temp_4x1;
    arm_mat_init_f32(&mat_temp_4x1, 4, 1, work_buffer->temp_4x1);
    status = arm_mat_mult_f32(&work_buffer->mat_temp_4x4, &mat_temp_4x1, &mat_v2);
    if (status != ARM_MATH_SUCCESS) return status;
    
    arm_scale_f32(v2, -1.0f, v2, 4);
    
    return ARM_MATH_SUCCESS;
}

arm_status ellipsoid_fit_li(const float32_t *samples, 
                           uint32_t num_samples,
                           ellipsoid_params_t *params,
                           ellipsoid_work_buffer_t *work_buffer) {
    arm_status status;
    float32_t v1[6], v2[4];
    
    if (!samples || !params || !work_buffer || num_samples == 0) {
        return ARM_MATH_ARGUMENT_ERROR;
    }
    
    // 构建设计矩阵
    status = build_design_matrix(samples, num_samples, work_buffer);
    if (status != ARM_MATH_SUCCESS) return status;
    
    // 求解特征值问题
    status = solve_eigenvalue_problem(work_buffer, v1, v2);
    if (status != ARM_MATH_SUCCESS) return status;
    
    // 构建椭球参数矩阵 M (3x3)
    params->M[0] = v1[0]; params->M[1] = v1[5]; params->M[2] = v1[4];  // [0,0] [0,1] [0,2]
    params->M[3] = v1[5]; params->M[4] = v1[1]; params->M[5] = v1[3];  // [1,0] [1,1] [1,2]
    params->M[6] = v1[4]; params->M[7] = v1[3]; params->M[8] = v1[2];  // [2,0] [2,1] [2,2]
    
    // 构建向量 n (3x1)
    params->n[0] = v2[0];
    params->n[1] = v2[1];
    params->n[2] = v2[2];
    
    // 标量 d
    params->d = v2[3];
    
    return ARM_MATH_SUCCESS;
}

static arm_status matrix_sqrt(const float32_t *input, float32_t *output, ellipsoid_work_buffer_t *work_buffer) {
    // 简化的矩阵平方根计算
    // 对于3x3对称正定矩阵，使用Cholesky分解的近似
    
    // 这里使用一个简化的方法：假设矩阵接近对角矩阵
    // 实际应用中可能需要更精确的算法
    
    for (int i = 0; i < 9; i++) {
        output[i] = input[i];
    }
    
    // 对角元素取平方根
    output[0] = sqrtf(fabsf(output[0]));
    output[4] = sqrtf(fabsf(output[4]));
    output[8] = sqrtf(fabsf(output[8]));
    
    // 非对角元素缩放
    output[1] *= 0.5f; output[2] *= 0.5f;
    output[3] *= 0.5f; output[5] *= 0.5f;
    output[6] *= 0.5f; output[7] *= 0.5f;
    
    return ARM_MATH_SUCCESS;
}

arm_status ellipsoid_to_calibration(const ellipsoid_params_t *params,
                                   float32_t F,
                                   calibration_params_t *calib,
                                   ellipsoid_work_buffer_t *work_buffer) {
    arm_status status;
    
    if (!params || !calib || !work_buffer) {
        return ARM_MATH_ARGUMENT_ERROR;
    }
    
    // 计算 M^(-1)
    arm_matrix_instance_f32 mat_M, mat_M_inv;
    arm_mat_init_f32(&mat_M, 3, 3, (float32_t*)params->M);
    arm_mat_init_f32(&mat_M_inv, 3, 3, work_buffer->temp_3x3);
    
    status = arm_mat_inverse_f32(&mat_M, &mat_M_inv);
    if (status != ARM_MATH_SUCCESS) return status;
    
    // 计算 h = -M^(-1) * n
    arm_matrix_instance_f32 mat_n, mat_h;
    arm_mat_init_f32(&mat_n, 3, 1, (float32_t*)params->n);
    arm_mat_init_f32(&mat_h, 3, 1, calib->h);
    
    status = arm_mat_mult_f32(&mat_M_inv, &mat_n, &mat_h);
    if (status != ARM_MATH_SUCCESS) return status;
    
    arm_scale_f32(calib->h, -1.0f, calib->h, 3);
    
    // 计算缩放因子: sqrt(n^T * M^(-1) * n - d)
    float32_t scale_factor;
    arm_dot_prod_f32(params->n, calib->h, 3, &scale_factor);
    scale_factor = scale_factor - params->d;
    
    if (scale_factor <= 0.0f) {
        return ARM_MATH_ARGUMENT_ERROR;  // 无效的椭球参数
    }
    
    scale_factor = sqrtf(scale_factor);
    
    // 计算 S = (F / scale_factor) * sqrt(M)
    status = matrix_sqrt(params->M, work_buffer->temp_3x3, work_buffer);
    if (status != ARM_MATH_SUCCESS) return status;
    
    float32_t coeff = F / scale_factor;
    arm_scale_f32(work_buffer->temp_3x3, coeff, calib->S, 9);
    
    return ARM_MATH_SUCCESS;
}

arm_status magnetometer_calibrate_li(const float32_t *samples,
                                    uint32_t num_samples,
                                    float32_t F,
                                    calibration_params_t *calib,
                                    ellipsoid_work_buffer_t *work_buffer) {
    arm_status status;
    ellipsoid_params_t params;
    
    // 椭球拟合
    status = ellipsoid_fit_li(samples, num_samples, &params, work_buffer);
    if (status != ARM_MATH_SUCCESS) return status;
    
    // 转换为校准参数
    status = ellipsoid_to_calibration(&params, F, calib, work_buffer);
    if (status != ARM_MATH_SUCCESS) return status;
    
    return ARM_MATH_SUCCESS;
}

void apply_calibration(const float32_t raw_data[3],
                      const calibration_params_t *calib,
                      float32_t calibrated_data[3]) {
    // 应用偏移: data_offset = raw_data - h
    float32_t offset_data[3];
    arm_sub_f32(raw_data, calib->h, offset_data, 3);
    
    // 应用变换矩阵: calibrated_data = S * offset_data
    arm_matrix_instance_f32 mat_S, mat_offset, mat_calib;
    arm_mat_init_f32(&mat_S, 3, 3, (float32_t*)calib->S);
    arm_mat_init_f32(&mat_offset, 3, 1, offset_data);
    arm_mat_init_f32(&mat_calib, 3, 1, calibrated_data);
    
    arm_mat_mult_f32(&mat_S, &mat_offset, &mat_calib);
}