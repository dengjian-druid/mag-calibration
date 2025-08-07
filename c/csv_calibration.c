/**
 * @file csv_calibration.c
 * @brief 从CSV文件中随机采样数据并使用Li算法计算校准参数
 * 
 * 支持从多个CSV文件中读取磁力计数据，随机采样后进行椭球拟合校准
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#define MAX_SAMPLES 10000
#define MAX_LINE_LENGTH 256
#define MIN_SAMPLES_FOR_CALIBRATION 50
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

// 磁力计数据结构
typedef struct {
    float x, y, z;
    int timestamp;
} mag_sample_t;

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
    
    // 椭球的一般方程: Ax^2 + By^2 + Cz^2 + Dx + Ey + Fz = 1
    // 我们要求解参数 [A, B, C, D, E, F]
    // 为了提高数值稳定性，先对数据进行归一化
    
    // 计算数据的均值和范围用于归一化
    float mean[3] = {0, 0, 0};
    float range[3] = {0, 0, 0};
    float min_vals[3] = {samples[0], samples[1], samples[2]};
    float max_vals[3] = {samples[0], samples[1], samples[2]};
    
    for (int i = 0; i < num_samples; i++) {
        for (int j = 0; j < 3; j++) {
            float val = samples[i*3 + j];
            mean[j] += val;
            if (val < min_vals[j]) min_vals[j] = val;
            if (val > max_vals[j]) max_vals[j] = val;
        }
    }
    
    for (int j = 0; j < 3; j++) {
        mean[j] /= num_samples;
        range[j] = max_vals[j] - min_vals[j];
        if (range[j] < MIN_VALUE) range[j] = 1.0f;
    }
    
    // 构建设计矩阵 H 和观测向量 f
    float *H = malloc(num_samples * 6 * sizeof(float));
    float *f = malloc(num_samples * sizeof(float));
    
    if (!H || !f) {
        free(H);
        free(f);
        return -1;
    }
    
    for (int i = 0; i < num_samples; i++) {
        // 使用归一化的坐标
        float x = (samples[i*3 + 0] - mean[0]) / range[0];
        float y = (samples[i*3 + 1] - mean[1]) / range[1];
        float z = (samples[i*3 + 2] - mean[2]) / range[2];
        
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
    
    // 从椭球方程系数提取中心和半径
    float A = theta[0], B = theta[1], C = theta[2];
    float D = theta[3], E = theta[4], F = theta[5];
    
    // 检查系数的有效性并输出调试信息
    printf("椭球系数: A=%.6f, B=%.6f, C=%.6f\n", A, B, C);
    printf("线性系数: D=%.6f, E=%.6f, F=%.6f\n", D, E, F);
    
    if (A <= MIN_VALUE || B <= MIN_VALUE || C <= MIN_VALUE) {
        printf("无效的椭球系数: A=%.6f, B=%.6f, C=%.6f\n", A, B, C);
        printf("尝试使用绝对值...\n");
        A = fabsf(A) + MIN_VALUE;
        B = fabsf(B) + MIN_VALUE;
        C = fabsf(C) + MIN_VALUE;
        printf("修正后系数: A=%.6f, B=%.6f, C=%.6f\n", A, B, C);
    }
    
    // 计算归一化坐标系中的椭球中心
    float norm_center[3];
    norm_center[0] = -D / (2.0f * A);
    norm_center[1] = -E / (2.0f * B);
    norm_center[2] = -F / (2.0f * C);
    
    // 转换回原始坐标系
    params->center[0] = norm_center[0] * range[0] + mean[0];
    params->center[1] = norm_center[1] * range[1] + mean[1];
    params->center[2] = norm_center[2] * range[2] + mean[2];
    
    // 计算椭球半径
    float center_term = A * norm_center[0] * norm_center[0] + 
                       B * norm_center[1] * norm_center[1] + 
                       C * norm_center[2] * norm_center[2];
    
    float scale_factor = 1.0f + center_term;
    
    if (scale_factor <= MIN_VALUE) {
        printf("无效的椭球尺度因子: %.6f\n", scale_factor);
        scale_factor = 1.0f;  // 使用默认值
    }
    
    // 计算归一化坐标系中的半径，然后转换回原始坐标系
    float norm_radii[3];
    norm_radii[0] = safe_sqrt(scale_factor / A);
    norm_radii[1] = safe_sqrt(scale_factor / B);
    norm_radii[2] = safe_sqrt(scale_factor / C);
    
    params->radii[0] = norm_radii[0] * range[0];
    params->radii[1] = norm_radii[1] * range[1];
    params->radii[2] = norm_radii[2] * range[2];
    
    free(H);
    free(f);
    return 0;
}

// 将椭球参数转换为校准参数
int ellipsoid_to_calibration(const ellipsoid_params_t *ellipsoid, 
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
    
    // 构建对角变换矩阵
    memset(calibration->S, 0, 9 * sizeof(float));
    calibration->S[0] = scale_x;
    calibration->S[4] = scale_y;
    calibration->S[8] = scale_z;
    
    return 0;
}

// 应用校准
void apply_calibration(const float *raw_data, const calibration_params_t *calibration, 
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

// 从CSV文件读取磁力计数据
int read_csv_data(const char *filename, mag_sample_t *samples, int max_samples) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("无法打开文件: %s\n", filename);
        return -1;
    }
    
    char line[MAX_LINE_LENGTH];
    int count = 0;
    
    // 跳过标题行
    if (fgets(line, sizeof(line), file) == NULL) {
        fclose(file);
        return -1;
    }
    
    // 读取数据行
    while (fgets(line, sizeof(line), file) && count < max_samples) {
        int timestamp;
        float x, y, z;
        
        // 解析CSV行: timestamp,x,y,z,
        if (sscanf(line, "%d,%f,%f,%f,", &timestamp, &x, &y, &z) == 4) {
            samples[count].timestamp = timestamp;
            samples[count].x = x;
            samples[count].y = y;
            samples[count].z = z;
            count++;
        }
    }
    
    fclose(file);
    printf("从 %s 读取了 %d 个样本\n", filename, count);
    return count;
}

// 随机采样函数
void random_sample(const mag_sample_t *input_samples, int input_count, 
                  float *output_samples, int sample_count) {
    if (sample_count > input_count) {
        sample_count = input_count;
    }
    
    // 创建索引数组
    int *indices = malloc(input_count * sizeof(int));
    for (int i = 0; i < input_count; i++) {
        indices[i] = i;
    }
    
    // Fisher-Yates洗牌算法
    for (int i = input_count - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int temp = indices[i];
        indices[i] = indices[j];
        indices[j] = temp;
    }
    
    // 提取前sample_count个样本
    for (int i = 0; i < sample_count; i++) {
        int idx = indices[i];
        output_samples[i*3 + 0] = input_samples[idx].x;
        output_samples[i*3 + 1] = input_samples[idx].y;
        output_samples[i*3 + 2] = input_samples[idx].z;
    }
    
    free(indices);
    printf("从 %d 个样本中随机选择了 %d 个样本\n", input_count, sample_count);
}

// 计算数据统计信息
void calculate_data_statistics(const float *samples, int num_samples) {
    float mean[3] = {0, 0, 0};
    float min_val[3] = {1e6f, 1e6f, 1e6f};
    float max_val[3] = {-1e6f, -1e6f, -1e6f};
    
    // 计算均值和范围
    for (int i = 0; i < num_samples; i++) {
        for (int j = 0; j < 3; j++) {
            float val = samples[i*3 + j];
            mean[j] += val;
            if (val < min_val[j]) min_val[j] = val;
            if (val > max_val[j]) max_val[j] = val;
        }
    }
    
    mean[0] /= num_samples;
    mean[1] /= num_samples;
    mean[2] /= num_samples;
    
    printf("\n=== 数据统计信息 ===\n");
    printf("样本数量: %d\n", num_samples);
    printf("均值: [%.1f, %.1f, %.1f]\n", mean[0], mean[1], mean[2]);
    printf("范围: X[%.1f, %.1f], Y[%.1f, %.1f], Z[%.1f, %.1f]\n",
           min_val[0], max_val[0], min_val[1], max_val[1], min_val[2], max_val[2]);
    
    // 计算原始数据的平均半径
    float sum_radius = 0.0f;
    for (int i = 0; i < num_samples; i++) {
        float x = samples[i*3 + 0];
        float y = samples[i*3 + 1];
        float z = samples[i*3 + 2];
        sum_radius += sqrtf(x*x + y*y + z*z);
    }
    printf("原始数据平均半径: %.1f\n", sum_radius / num_samples);
}

// 计算校准效果统计
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
        
        apply_calibration(raw, calibration, calibrated);
        
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

// 完整的校准流程
int magnetometer_calibrate_csv(const float *samples, int num_samples, 
                              float target_field_strength,
                              calibration_params_t *calibration) {
    ellipsoid_params_t ellipsoid;
    
    if (ellipsoid_fit_lsq(samples, num_samples, &ellipsoid) != 0) {
        return -1;
    }
    
    printf("\n=== 椭球拟合结果 ===\n");
    printf("椭球中心: [%.3f, %.3f, %.3f]\n", 
           ellipsoid.center[0], ellipsoid.center[1], ellipsoid.center[2]);
    printf("椭球半径: [%.3f, %.3f, %.3f]\n", 
           ellipsoid.radii[0], ellipsoid.radii[1], ellipsoid.radii[2]);
    
    return ellipsoid_to_calibration(&ellipsoid, target_field_strength, calibration);
}

// 主函数
int main(int argc, char *argv[]) {
    // 初始化随机数种子
    srand(time(NULL));
    
    // 默认参数
    const char *csv_file = "../data/1300001855-circle-round-01.csv";
    int sample_count = 300;
    float target_field = 500.0f;
    
    // 解析命令行参数
    if (argc > 1) {
        csv_file = argv[1];
    }
    if (argc > 2) {
        sample_count = atoi(argv[2]);
    }
    if (argc > 3) {
        target_field = atof(argv[3]);
    }
    
    printf("=== CSV磁力计校准程序 ===\n");
    printf("CSV文件: %s\n", csv_file);
    printf("采样数量: %d\n", sample_count);
    printf("目标磁场强度: %.1f\n", target_field);
    
    // 读取CSV数据
    mag_sample_t *all_samples = malloc(MAX_SAMPLES * sizeof(mag_sample_t));
    if (!all_samples) {
        printf("内存分配失败\n");
        return -1;
    }
    
    int total_samples = read_csv_data(csv_file, all_samples, MAX_SAMPLES);
    if (total_samples < MIN_SAMPLES_FOR_CALIBRATION) {
        printf("样本数量不足，至少需要 %d 个样本\n", MIN_SAMPLES_FOR_CALIBRATION);
        free(all_samples);
        return -1;
    }
    
    // 限制采样数量
    if (sample_count > total_samples) {
        sample_count = total_samples;
    }
    if (sample_count < MIN_SAMPLES_FOR_CALIBRATION) {
        sample_count = MIN_SAMPLES_FOR_CALIBRATION;
    }
    
    // 随机采样
    float *selected_samples = malloc(sample_count * 3 * sizeof(float));
    if (!selected_samples) {
        printf("内存分配失败\n");
        free(all_samples);
        return -1;
    }
    
    random_sample(all_samples, total_samples, selected_samples, sample_count);
    
    // 计算数据统计
    calculate_data_statistics(selected_samples, sample_count);
    
    // 执行校准
    calibration_params_t calibration;
    printf("\n=== 开始椭球拟合校准 ===\n");
    
    if (magnetometer_calibrate_csv(selected_samples, sample_count, target_field, &calibration) == 0) {
        printf("\n=== 校准参数 ===\n");
        printf("偏移向量 h: [%.3f, %.3f, %.3f]\n", 
               calibration.h[0], calibration.h[1], calibration.h[2]);
        printf("缩放因子: [%.3f, %.3f, %.3f]\n", 
               calibration.S[0], calibration.S[4], calibration.S[8]);
        
        // 计算校准效果
        calculate_calibration_stats(selected_samples, sample_count, &calibration, target_field);
        
        // 输出C代码格式的校准参数
        printf("\n=== C代码格式校准参数 ===\n");
        printf("// 磁力计校准参数\n");
        printf("const float mag_offset[3] = {%.3ff, %.3ff, %.3ff};\n",
               calibration.h[0], calibration.h[1], calibration.h[2]);
        printf("const float mag_scale[3] = {%.3ff, %.3ff, %.3ff};\n",
               calibration.S[0], calibration.S[4], calibration.S[8]);
        printf("\n// 校准函数\n");
        printf("void calibrate_magnetometer(float raw[3], float calibrated[3]) {\n");
        printf("    calibrated[0] = (raw[0] - %.3ff) * %.3ff;\n", calibration.h[0], calibration.S[0]);
        printf("    calibrated[1] = (raw[1] - %.3ff) * %.3ff;\n", calibration.h[1], calibration.S[4]);
        printf("    calibrated[2] = (raw[2] - %.3ff) * %.3ff;\n", calibration.h[2], calibration.S[8]);
        printf("}\n");
        
    } else {
        printf("校准失败！\n");
    }
    
    free(all_samples);
    free(selected_samples);
    return 0;
}

// 使用说明：
// ./csv_calibration [csv_file] [sample_count] [target_field]
// 例如：
// ./csv_calibration ../data/1d000006ab_mag_20250218.csv 200 500
// ./csv_calibration ../data/1300001855-circle-round-01.csv 300 400