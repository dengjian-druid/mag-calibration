import numpy as np
import matplotlib.pyplot as plt

def plot_views(data, title_prefix, fig, pos_offset=0, draw_unit_circle=False):
    """绘制三个视图的散点图
    
    参数:
        data: 数据点 (N x 3)
        title_prefix: 图表标题前缀
        fig: matplotlib figure对象
        pos_offset: 子图位置偏移
        draw_unit_circle: 是否绘制单位圆
    """
    views = [('XY', 0, 1), ('XZ', 0, 2), ('YZ', 1, 2)]
    
    for i, (view_name, x_idx, y_idx) in enumerate(views):
        ax = fig.add_subplot(2, 3, i + 1 + pos_offset)
        ax.scatter(data[:, x_idx], data[:, y_idx], alpha=0.6, s=1)
        ax.set_xlabel(view_name[0])
        ax.set_ylabel(view_name[1])
        ax.set_title(f"{title_prefix} - {view_name} View")
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        if draw_unit_circle:
            circle = plt.Circle((0, 0), 500, fill=False, color='red', linestyle='--', alpha=0.7)
            ax.add_patch(circle)

def create_calibration_plot(original_data, calibrated_data, algorithm_name, save_filename):
    """创建校准前后对比图
    
    参数:
        original_data: 原始数据 (N x 3)
        calibrated_data: 校准后数据 (N x 3)
        algorithm_name: 算法名称
        save_filename: 保存文件名
    """
    fig = plt.figure(figsize=(15, 10))
    
    # 绘制原始数据
    plot_views(original_data, "Raw Data", fig, pos_offset=0, draw_unit_circle=False)
    
    # 绘制校准后数据
    plot_views(calibrated_data, "Calibrated Data", fig, pos_offset=3, draw_unit_circle=True)
    
    plt.suptitle(f'{algorithm_name} Magnetometer Calibration Result', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Visualization saved as: {save_filename}")