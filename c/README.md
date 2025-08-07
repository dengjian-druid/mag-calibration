# nRF52840 磁力计校准库 (Li算法)

这是一个专为nRF52840微控制器设计的磁力计校准库，使用Li算法和CMSIS-DSP库实现高效的椭球拟合和校准参数计算。

## 特性

- **高效算法**: 基于Li等人的椭球拟合算法，计算复杂度低
- **CMSIS-DSP优化**: 充分利用Cortex-M4的DSP指令和FPU
- **内存优化**: 使用工作缓冲区减少栈内存使用
- **实时友好**: 支持在线校准，适合嵌入式实时系统
- **易于集成**: 简洁的API设计，易于集成到现有项目

## 文件结构

```
c/
├── ellipsoid_fit_li.h      # 头文件，包含API声明和数据结构
├── ellipsoid_fit_li.c      # 实现文件，包含Li算法的C实现
├── magnetometer_example.c  # 示例程序，展示如何使用库
├── Makefile               # 编译配置文件
└── README.md              # 本文档
```

## 依赖要求

### 硬件要求
- nRF52840微控制器 (Cortex-M4F)
- 至少32KB RAM (推荐64KB)
- 磁力计传感器 (如HMC5883L, LSM303, 等)

### 软件要求
- ARM GCC工具链 (`arm-none-eabi-gcc`)
- CMSIS-DSP库 (v1.7.0或更高版本)
- nRF5 SDK (v17.0或更高版本)

## 安装和配置

### 1. 安装ARM工具链

```bash
# macOS (使用Homebrew)
brew install --cask gcc-arm-embedded

# Ubuntu/Debian
sudo apt-get install gcc-arm-none-eabi

# 或从ARM官网下载: https://developer.arm.com/tools-and-software/open-source-software/developer-tools/gnu-toolchain/gnu-rm
```

### 2. 下载CMSIS-DSP库

```bash
# 克隆CMSIS仓库
git clone https://github.com/ARM-software/CMSIS_5.git /opt/CMSIS_5

# 或下载预编译库
wget https://github.com/ARM-software/CMSIS_5/releases/download/5.8.0/ARM.CMSIS.5.8.0.pack
```

### 3. 下载nRF5 SDK

```bash
# 从Nordic官网下载
wget https://www.nordicsemi.com/-/media/Software-and-other-downloads/SDKs/nRF5/Binaries/nRF5SDK1702d674dde.zip
unzip nRF5SDK1702d674dde.zip -d /opt/nRF5_SDK
```

### 4. 配置环境变量

```bash
export CMSIS_PATH=/opt/CMSIS_5
export NRF_SDK_PATH=/opt/nRF5_SDK
```

## 编译

### 检查依赖

```bash
make check-deps
```

### 本地测试编译

```bash
# 使用本地GCC进行功能测试
make test-local
make run-test
```

### ARM目标编译

```bash
# 编译完整固件
make all

# 仅编译库文件
make lib
```

### 清理

```bash
make clean
```

## API使用说明

### 基本数据结构

```c
// 椭球参数
typedef struct {
    float32_t center[3];    // 椭球中心
    float32_t radii[3];     // 椭球半径
    float32_t rotation[9];  // 旋转矩阵 (3x3)
} ellipsoid_params_t;

// 校准参数
typedef struct {
    float32_t h[3];         // 偏移向量
    float32_t S[9];         // 变换矩阵 (3x3)
} calibration_params_t;
```

### 核心API

#### 1. 完整校准流程

```c
// 一步完成校准
arm_status magnetometer_calibrate_li(
    const float32_t *samples,           // 输入: 磁力计样本 [x1,y1,z1,x2,y2,z2,...]
    uint32_t num_samples,               // 输入: 样本数量
    float32_t expected_field_strength,  // 输入: 期望磁场强度
    calibration_params_t *calib,        // 输出: 校准参数
    ellipsoid_work_buffer_t *work_buf   // 工作缓冲区
);
```

#### 2. 应用校准

```c
// 对单个测量值应用校准
void apply_calibration(
    const float32_t raw[3],             // 输入: 原始磁力计数据
    const calibration_params_t *calib,  // 输入: 校准参数
    float32_t calibrated[3]             // 输出: 校准后数据
);
```

### 使用示例

```c
#include "ellipsoid_fit_li.h"

int main(void) {
    // 1. 分配工作缓冲区和校准参数
    static ellipsoid_work_buffer_t work_buffer;
    static calibration_params_t calib_params;
    
    // 2. 初始化工作缓冲区
    init_work_buffer(&work_buffer);
    
    // 3. 收集磁力计数据 (建议500-1000个样本)
    float32_t samples[1500]; // 500个样本 × 3轴
    uint32_t num_samples = 500;
    
    // ... 收集数据的代码 ...
    
    // 4. 执行校准
    arm_status status = magnetometer_calibrate_li(
        samples, num_samples, 500.0f,  // 假设地磁场强度为500
        &calib_params, &work_buffer
    );
    
    if (status == ARM_MATH_SUCCESS) {
        // 5. 校准成功，保存参数到Flash或使用
        
        // 6. 应用校准到新的测量值
        float32_t raw[3] = {100.0f, 200.0f, 150.0f};
        float32_t calibrated[3];
        apply_calibration(raw, &calib_params, calibrated);
        
        printf("校准后: [%.2f, %.2f, %.2f]\n", 
               calibrated[0], calibrated[1], calibrated[2]);
    }
    
    return 0;
}
```

## 内存使用

| 组件 | 大小 (字节) | 说明 |
|------|-------------|------|
| `ellipsoid_work_buffer_t` | ~2400 | 工作缓冲区，可重复使用 |
| `calibration_params_t` | 48 | 校准参数，需要持久保存 |
| 样本数据 (500点) | 6000 | 临时存储，校准完成后可释放 |
| **总计** | **~8.5KB** | 峰值内存使用 |

## 性能特性

- **计算时间**: 约50-100ms (500个样本，64MHz)
- **内存占用**: 峰值8.5KB
- **精度**: 校准后半径标准差通常<5%
- **收敛性**: 对初值不敏感，鲁棒性好

## 集成到现有项目

### 1. 复制文件

```bash
cp ellipsoid_fit_li.h ellipsoid_fit_li.c /path/to/your/project/
```

### 2. 修改Makefile

```makefile
# 添加源文件
SRCS += ellipsoid_fit_li.c

# 添加CMSIS-DSP库
LDFLAGS += -larm_cortexM4lf_math

# 添加编译标志
CFLAGS += -DARM_MATH_CM4 -DARM_MATH_MATRIX_CHECK
```

### 3. 在代码中使用

```c
#include "ellipsoid_fit_li.h"

// 在适当的地方调用校准函数
// 建议在系统初始化时或用户触发时执行
```

## 校准数据收集建议

### 数据收集策略

1. **样本数量**: 建议500-1000个样本
2. **采样频率**: 10-50Hz
3. **运动模式**: 
   - 绕X、Y、Z轴各旋转一圈
   - 或进行"8字形"运动
   - 确保覆盖所有方向

### 数据质量检查

```c
// 检查数据范围
float32_t min_val[3], max_val[3];
for (int axis = 0; axis < 3; axis++) {
    // 确保每个轴的数据范围足够大
    float32_t range = max_val[axis] - min_val[axis];
    if (range < 100.0f) {
        // 警告: 数据范围太小
    }
}
```

## 故障排除

### 常见问题

1. **编译错误**: 
   - 检查CMSIS-DSP库路径
   - 确认ARM工具链安装正确

2. **链接错误**:
   - 检查链接脚本路径
   - 确认CMSIS-DSP库版本兼容

3. **运行时错误**:
   - 检查栈大小是否足够
   - 确认FPU已启用

4. **校准效果差**:
   - 增加样本数量
   - 改善数据收集策略
   - 检查传感器硬件

### 调试技巧

```c
// 启用调试输出
#define DEBUG_CALIBRATION 1

// 检查中间结果
if (status != ARM_MATH_SUCCESS) {
    printf("校准失败，错误代码: %d\n", status);
}
```

## 许可证

本代码基于MIT许可证发布，可自由用于商业和非商业项目。

## 参考文献

1. Li, Q. and Griffiths, J.G., 2004. Least squares ellipsoid specific fitting. *Geometric modeling and processing*, pp.335-340.

2. ARM Limited. CMSIS-DSP Software Library. https://arm-software.github.io/CMSIS_5/DSP/html/index.html

3. Nordic Semiconductor. nRF52840 Product Specification. https://www.nordicsemi.com/Products/Low-power-short-range-wireless/nRF52840

## 联系和支持

如有问题或建议，请通过以下方式联系：
- 创建GitHub Issue
- 发送邮件至开发者

---

**注意**: 本库专为nRF52840设计，但也可以适配其他Cortex-M4F微控制器。使用前请确保硬件支持浮点运算。