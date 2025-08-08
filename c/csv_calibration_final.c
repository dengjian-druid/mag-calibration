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

// 磁力计样本结构
typedef struct {
    float x, y, z;
    int timestamp;
} mag_sample_t;

// 安全开方函数
float safe_sqrt(float x) {
    return sqrtf(fmaxf(x, MIN_VALUE));
}

// 安全除法函数
float safe_divide(float a, float b) {
    if (fabsf(b) < MIN_VALUE) {
        return (a >= 0) ? 1.0f : -1.0f;
    }
    return a / b;
}

// 高斯消元法求解线性方程组
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
            for (int j = 0; j < n; j++) {
                float temp = A[i*n + j];
                A[i*n + j] = A[max_row*n + j];
                A[max_row*n + j] = temp;
            }
            float temp = b[i];
            b[i] = b[max_row];
            b[max_row] = temp;
        }
        
        // 检查主元是否为零
        if (fabsf(A[i*n + i]) < TOLERANCE) {
            printf("矩阵奇异，无法求解\n");
            return -1;
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

// 使用最小二乘法拟合椭球（带数据归一化）
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
    
    // 缩放因子
    for (int i = 0; i < 3; i++) {
        calibration->S[i*3 + i] = safe_divide(target_field_strength, ellipsoid->radii[i]);
        // 非对角元素设为0（假设椭球轴对齐）
        for (int j = 0; j < 3; j++) {
            if (i != j) {
                calibration->S[i*3 + j] = 0.0f;
            }
        }
    }
    
    return 0;
}

// 应用校准
void apply_calibration(const float *raw_data, const calibration_params_t *calibration, 
                      float *calibrated_data) {
    // 先减去偏移
    float temp[3];
    temp[0] = raw_data[0] - calibration->h[0];
    temp[1] = raw_data[1] - calibration->h[1];
    temp[2] = raw_data[2] - calibration->h[2];
    
    // 再应用变换矩阵（这里简化为对角矩阵）
    calibrated_data[0] = temp[0] * calibration->S[0];
    calibrated_data[1] = temp[1] * calibration->S[4];
    calibrated_data[2] = temp[2] * calibration->S[8];
}

// 从CSV文件读取数据
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
    
    while (fgets(line, sizeof(line), file) && count < max_samples) {
        char *token = strtok(line, ",");
        if (!token) continue;
        
        samples[count].timestamp = atoi(token);
        
        token = strtok(NULL, ",");
        if (!token) continue;
        samples[count].x = atof(token);
        
        token = strtok(NULL, ",");
        if (!token) continue;
        samples[count].y = atof(token);
        
        token = strtok(NULL, ",");
        if (!token) continue;
        samples[count].z = atof(token);
        
        count++;
    }
    
    fclose(file);
    return count;
}

// 随机采样
void random_sample(const mag_sample_t *input_samples, int input_count, 
                  float *output_samples, int sample_count) {
    srand(time(NULL));
    
    for (int i = 0; i < sample_count; i++) {
        int idx = rand() % input_count;
        output_samples[i*3 + 0] = input_samples[idx].x;
        output_samples[i*3 + 1] = input_samples[idx].y;
        output_samples[i*3 + 2] = input_samples[idx].z;
    }
}

// 计算数据统计信息
void calculate_data_statistics(const float *samples, int num_samples) {
    float mean[3] = {0, 0, 0};
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
    }
    
    float avg_radius = 0;
    for (int i = 0; i < num_samples; i++) {
        float radius = sqrtf(samples[i*3]*samples[i*3] + 
                           samples[i*3+1]*samples[i*3+1] + 
                           samples[i*3+2]*samples[i*3+2]);
        avg_radius += radius;
    }
    avg_radius /= num_samples;
    
    printf("\n=== 数据统计信息 ===\n");
    printf("样本数量: %d\n", num_samples);
    printf("均值: [%.1f, %.1f, %.1f]\n", mean[0], mean[1], mean[2]);
    printf("范围: X[%.1f, %.1f], Y[%.1f, %.1f], Z[%.1f, %.1f]\n", 
           min_vals[0], max_vals[0], min_vals[1], max_vals[1], min_vals[2], max_vals[2]);
    printf("原始数据平均半径: %.1f\n", avg_radius);
}

// 计算校准效果统计
void calculate_calibration_stats(const float *samples, int num_samples, 
                               const calibration_params_t *calibration,
                               float target_field) {
    float *radii = malloc(num_samples * sizeof(float));
    float sum = 0, sum_sq = 0;
    float min_radius = INFINITY, max_radius = 0;
    
    for (int i = 0; i < num_samples; i++) {
        float calibrated[3];
        apply_calibration(&samples[i*3], calibration, calibrated);
        
        float radius = sqrtf(calibrated[0]*calibrated[0] + 
                           calibrated[1]*calibrated[1] + 
                           calibrated[2]*calibrated[2]);
        radii[i] = radius;
        sum += radius;
        sum_sq += radius * radius;
        
        if (radius < min_radius) min_radius = radius;
        if (radius > max_radius) max_radius = radius;
    }
    
    float mean = sum / num_samples;
    float variance = (sum_sq / num_samples) - (mean * mean);
    float std_dev = sqrtf(variance);
    float cv = (std_dev / mean) * 100.0f;
    
    printf("\n=== 校准效果统计 ===\n");
    printf("平均半径: %.3f\n", mean);
    printf("标准差: %.3f\n", std_dev);
    printf("最小半径: %.3f\n", min_radius);
    printf("最大半径: %.3f\n", max_radius);
    printf("变异系数: %.2f%%\n", cv);
    printf("期望半径: %.1f\n", target_field);
    printf("绝对误差: %.3f\n", fabsf(mean - target_field));
    printf("相对误差: %.2f%%\n", fabsf(mean - target_field) / target_field * 100.0f);
    
    free(radii);
}

// 主校准函数
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

void print_usage(const char *program_name) {
    printf("用法: %s [选项]\n", program_name);
    printf("选项:\n");
    printf("  -f <文件>     CSV文件路径 (默认: ../data/1d000006ab_mag_20250218.csv)\n");
    printf("  -n <数量>     训练样本数量 (默认: 200)\n");
    printf("  -t <强度>     目标磁场强度 (默认: 500.0)\n");
    printf("  -s <数量>     丢弃前后样本数量 (默认: 100)\n");
    printf("  -h            显示此帮助信息\n");
    printf("\n说明:\n");
    printf("  程序将从CSV文件中丢弃前后指定数量的样本，然后从剩余样本中\n");
    printf("  随机选择指定数量的样本作为训练集，剩余的所有样本作为测试集。\n");
    printf("\n示例:\n");
    printf("  %s -f ../data/test.csv -n 300 -t 450.0 -s 50\n", program_name);
}

int main(int argc, char *argv[]) {
    // 默认参数
    const char *csv_file = "../data/1d000006ab_mag_20250218.csv";
    int train_count = 200;  // 训练样本数量
    float target_field = 500.0f;
    int skip_samples = 100; // 丢弃前后样本数量
    
    // 解析命令行参数
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-f") == 0 && i + 1 < argc) {
            csv_file = argv[++i];
        } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            train_count = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            target_field = atof(argv[++i]);
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            skip_samples = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            printf("未知参数: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }
    
    printf("=== CSV磁力计校准程序 ===\n");
    printf("CSV文件: %s\n", csv_file);
    printf("训练样本数量: %d\n", train_count);
    printf("目标磁场强度: %.1f\n", target_field);
    printf("丢弃前后样本数量: %d\n", skip_samples);
    
    // 验证参数
    if (train_count < MIN_SAMPLES_FOR_CALIBRATION) {
        printf("错误：训练样本数量太少，至少需要 %d 个样本\n", MIN_SAMPLES_FOR_CALIBRATION);
        return 1;
    }
    
    if (train_count > MAX_SAMPLES) {
        printf("错误：训练样本数量太多，最多支持 %d 个样本\n", MAX_SAMPLES);
        return 1;
    }
    
    // 读取CSV数据
    mag_sample_t *all_samples = malloc(MAX_SAMPLES * sizeof(mag_sample_t));
    if (!all_samples) {
        printf("内存分配失败\n");
        return 1;
    }
    
    int total_samples = read_csv_data(csv_file, all_samples, MAX_SAMPLES);
    if (total_samples < 0) {
        free(all_samples);
        return 1;
    }
    
    printf("从 %s 读取了 %d 个样本\n", csv_file, total_samples);
    
    // 验证丢弃样本数量的合理性
    if (skip_samples < 0) {
        printf("错误：丢弃样本数量不能为负数\n");
        free(all_samples);
        return 1;
    }
    
    if (skip_samples * 2 >= total_samples) {
        printf("错误：丢弃样本数量过多，前后各丢弃 %d 个样本会超过总样本数 %d\n", 
               skip_samples, total_samples);
        free(all_samples);
        return 1;
    }
    
    // 丢弃前后指定数量的样本
    int effective_samples = total_samples - 2 * skip_samples;
    mag_sample_t *filtered_samples = malloc(effective_samples * sizeof(mag_sample_t));
    if (!filtered_samples) {
        printf("内存分配失败\n");
        free(all_samples);
        return 1;
    }
    
    // 复制中间的有效样本
    for (int i = 0; i < effective_samples; i++) {
        filtered_samples[i] = all_samples[skip_samples + i];
    }
    
    printf("丢弃前后各 %d 个样本，剩余有效样本: %d\n", skip_samples, effective_samples);
    
    if (effective_samples < train_count) {
        printf("错误：有效样本数量 (%d) 少于请求的训练样本数量 (%d)\n", 
               effective_samples, train_count);
        free(all_samples);
        free(filtered_samples);
        return 1;
    }
    
    // 计算测试集数量
    int test_count = effective_samples - train_count;
    
    float *train_samples = malloc(train_count * 3 * sizeof(float));
    float *test_samples = malloc(test_count * 3 * sizeof(float));
    
    if (!train_samples || !test_samples) {
        printf("内存分配失败\n");
        free(all_samples);
        free(filtered_samples);
        free(train_samples);
        free(test_samples);
        return 1;
    }
    
    // 随机打乱有效样本
    srand(time(NULL));
    for (int i = effective_samples - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        mag_sample_t temp = filtered_samples[i];
        filtered_samples[i] = filtered_samples[j];
        filtered_samples[j] = temp;
    }
    
    // 前train_count个作为训练集
    for (int i = 0; i < train_count; i++) {
        train_samples[i*3 + 0] = filtered_samples[i].x;
        train_samples[i*3 + 1] = filtered_samples[i].y;
        train_samples[i*3 + 2] = filtered_samples[i].z;
    }
    
    // 剩余的作为测试集
    for (int i = 0; i < test_count; i++) {
        int idx = train_count + i;
        test_samples[i*3 + 0] = filtered_samples[idx].x;
        test_samples[i*3 + 1] = filtered_samples[idx].y;
        test_samples[i*3 + 2] = filtered_samples[idx].z;
    }
    
    free(all_samples);
    free(filtered_samples);
    
    printf("训练集样本数量: %d\n", train_count);
    printf("测试集样本数量: %d\n", test_count);
    
    // 计算训练集数据统计
    printf("\n=== 训练集数据统计 ===\n");
    calculate_data_statistics(train_samples, train_count);
    
    // 执行校准（使用训练集）
    printf("\n=== 开始椭球拟合校准 ===\n");
    calibration_params_t calibration;
    
    if (magnetometer_calibrate_csv(train_samples, train_count, target_field, &calibration) != 0) {
        printf("校准失败！\n");
        free(train_samples);
        free(test_samples);
        return 1;
    }
    
    printf("\n=== 校准参数 ===\n");
    printf("偏移向量 h: [%.3f, %.3f, %.3f]\n", 
           calibration.h[0], calibration.h[1], calibration.h[2]);
    printf("缩放因子: [%.3f, %.3f, %.3f]\n", 
           calibration.S[0], calibration.S[4], calibration.S[8]);
    
    // 计算训练集校准效果
    printf("\n=== 训练集校准效果 ===\n");
    calculate_calibration_stats(train_samples, train_count, &calibration, target_field);
    
    // 计算测试集校准效果
    printf("\n=== 测试集校准效果 ===\n");
    calculate_calibration_stats(test_samples, test_count, &calibration, target_field);
    
    // 输出C代码格式的校准参数
    printf("\n=== C代码格式校准参数 ===\n");
    printf("const float mag_offset[3] = {%.3ff, %.3ff, %.3ff};\n", 
           calibration.h[0], calibration.h[1], calibration.h[2]);
    printf("const float mag_scale[3] = {%.3ff, %.3ff, %.3ff};\n", 
           calibration.S[0], calibration.S[4], calibration.S[8]);
    printf("\nvoid calibrate_magnetometer(float raw[3], float calibrated[3]) {\n");
    printf("    calibrated[0] = (raw[0] - %.3ff) * %.3ff;\n", calibration.h[0], calibration.S[0]);
    printf("    calibrated[1] = (raw[1] - %.3ff) * %.3ff;\n", calibration.h[1], calibration.S[4]);
    printf("    calibrated[2] = (raw[2] - %.3ff) * %.3ff;\n", calibration.h[2], calibration.S[8]);
    printf("}\n");
    
    free(train_samples);
    free(test_samples);
    return 0;
}