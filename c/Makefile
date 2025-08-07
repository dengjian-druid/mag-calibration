# Makefile for nRF52840 Magnetometer Calibration
# 使用Li算法和CMSIS-DSP库

# 编译器设置
CC = arm-none-eabi-gcc
AR = arm-none-eabi-ar
OBJCOPY = arm-none-eabi-objcopy
SIZE = arm-none-eabi-size

# 目标MCU设置
MCU = -mcpu=cortex-m4 -mthumb -mfpu=fpv4-sp-d16 -mfloat-abi=hard

# 项目名称
PROJECT = magnetometer_calibration

# 源文件
SRCS = ellipsoid_fit_li.c magnetometer_example.c

# 头文件路径
INCLUDES = -I. \
           -I$(CMSIS_PATH)/CMSIS/Core/Include \
           -I$(CMSIS_PATH)/CMSIS/DSP/Include \
           -I$(NRF_SDK_PATH)/components/device \
           -I$(NRF_SDK_PATH)/components/toolchain/cmsis/include

# CMSIS-DSP库路径 (需要根据实际安装路径调整)
CMSIS_PATH ?= /opt/CMSIS_5
CMSIS_DSP_LIB = $(CMSIS_PATH)/CMSIS/DSP/Lib/GCC/libarm_cortexM4lf_math.a

# nRF SDK路径 (需要根据实际安装路径调整)
NRF_SDK_PATH ?= /opt/nRF5_SDK

# 编译标志
CFLAGS = $(MCU) \
         -DNRF52840_XXAA \
         -DARM_MATH_CM4 \
         -DARM_MATH_MATRIX_CHECK \
         -DARM_MATH_ROUNDING \
         -D__FPU_PRESENT=1 \
         -Wall \
         -Wextra \
         -Wno-unused-parameter \
         -O2 \
         -g3 \
         -ffunction-sections \
         -fdata-sections \
         -fno-strict-aliasing \
         -fno-builtin \
         --short-enums \
         $(INCLUDES)

# 链接标志
LDFLAGS = $(MCU) \
          -Wl,--gc-sections \
          -Wl,--print-memory-usage \
          --specs=nano.specs \
          -lc \
          -lnosys \
          -lm

# 链接脚本 (需要根据实际nRF52840链接脚本路径调整)
LINKER_SCRIPT ?= nrf52840_xxaa.ld

# 目标文件
OBJS = $(SRCS:.c=.o)

# 默认目标
all: $(PROJECT).elf $(PROJECT).hex $(PROJECT).bin size

# 编译规则
%.o: %.c
	@echo "Compiling $<"
	$(CC) $(CFLAGS) -c $< -o $@

# 链接规则
$(PROJECT).elf: $(OBJS)
	@echo "Linking $@"
	$(CC) $(LDFLAGS) -T$(LINKER_SCRIPT) $(OBJS) $(CMSIS_DSP_LIB) -o $@

# 生成HEX文件
$(PROJECT).hex: $(PROJECT).elf
	@echo "Creating $@"
	$(OBJCOPY) -O ihex $< $@

# 生成BIN文件
$(PROJECT).bin: $(PROJECT).elf
	@echo "Creating $@"
	$(OBJCOPY) -O binary $< $@

# 显示大小信息
size: $(PROJECT).elf
	@echo "Size information:"
	$(SIZE) $<

# 清理
clean:
	rm -f *.o *.elf *.hex *.bin *.map

# 仅编译库文件 (用于测试)
lib: ellipsoid_fit_li.o
	@echo "Library object created: ellipsoid_fit_li.o"

# 使用本地GCC进行测试编译 (不依赖ARM工具链)
test-local:
	@echo "Testing compilation with local GCC..."
	gcc -DTEST_LOCAL -DARM_MATH_CM4 -I. -Wall -Wextra -O2 -lm \
	    ellipsoid_fit_li.c magnetometer_example.c -o $(PROJECT)_test
	@echo "Local test binary created: $(PROJECT)_test"

# 运行本地测试
run-test: test-local
	@echo "Running local test..."
	./$(PROJECT)_test

# 检查依赖
check-deps:
	@echo "Checking dependencies..."
	@echo "ARM GCC: $(shell which $(CC) 2>/dev/null || echo 'NOT FOUND')"
	@echo "CMSIS Path: $(CMSIS_PATH)"
	@echo "CMSIS-DSP Lib: $(shell ls $(CMSIS_DSP_LIB) 2>/dev/null || echo 'NOT FOUND')"
	@echo "nRF SDK Path: $(NRF_SDK_PATH)"

# 帮助信息
help:
	@echo "Available targets:"
	@echo "  all        - Build complete firmware (requires ARM toolchain)"
	@echo "  lib        - Build only the library object file"
	@echo "  test-local - Build with local GCC for testing"
	@echo "  run-test   - Build and run local test"
	@echo "  check-deps - Check if required dependencies are available"
	@echo "  clean      - Remove all build files"
	@echo "  help       - Show this help message"
	@echo ""
	@echo "Environment variables:"
	@echo "  CMSIS_PATH    - Path to CMSIS installation"
	@echo "  NRF_SDK_PATH  - Path to nRF5 SDK installation"
	@echo "  LINKER_SCRIPT - Path to nRF52840 linker script"

.PHONY: all clean lib test-local run-test check-deps help size