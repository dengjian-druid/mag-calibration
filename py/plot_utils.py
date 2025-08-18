import numpy as np
import matplotlib.pyplot as plt

def plot_views(data, title_prefix, fig, pos_offset=0, draw_unit_circle=False):
    """Plot scatter plots for three views
    
    Parameters:
        data: Data points (N x 3)
        title_prefix: Chart title prefix
        fig: matplotlib figure object
        pos_offset: Subplot position offset
        draw_unit_circle: Whether to draw unit circle
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
    """Create before and after calibration comparison plot
    
    Parameters:
        original_data: Original data (N x 3)
        calibrated_data: Calibrated data (N x 3)
        algorithm_name: Algorithm name
        save_filename: Save filename
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

def create_re_visualization(calibrated_data, target_field, algorithm_name, save_filename):
    """Create Residual Error (RE) visualization charts
    
    Parameters:
        calibrated_data: Calibrated data (N x 3)
        target_field: Target magnetic field strength
        algorithm_name: Algorithm name
        save_filename: Save filename
    """
    # Calculate radius and residual errors
    radius = np.linalg.norm(calibrated_data, axis=1)
    residual_errors = np.abs(radius - target_field)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Residual error histogram
    ax1.hist(residual_errors, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Residual Error (RE)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Residual Error Distribution Histogram')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(residual_errors.mean(), color='red', linestyle='--', 
                label=f'Mean RE: {residual_errors.mean():.3f}')
    ax1.legend()
    
    # 2. Radius vs residual error scatter plot
    ax2.scatter(radius, residual_errors, alpha=0.6, s=10, color='orange')
    ax2.set_xlabel('Calibrated Radius')
    ax2.set_ylabel('Residual Error (RE)')
    ax2.set_title('Radius vs Residual Error')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(residual_errors.mean(), color='red', linestyle='--', 
                label=f'Mean RE: {residual_errors.mean():.3f}')
    ax2.axvline(target_field, color='green', linestyle='--', 
                label=f'Target Radius: {target_field}')
    ax2.legend()
    
    # 3. Cumulative Distribution Function (CDF)
    sorted_re = np.sort(residual_errors)
    cdf = np.arange(1, len(sorted_re) + 1) / len(sorted_re)
    ax3.plot(sorted_re, cdf, linewidth=2, color='purple')
    ax3.set_xlabel('Residual Error (RE)')
    ax3.set_ylabel('Cumulative Probability')
    ax3.set_title('Residual Error Cumulative Distribution Function')
    ax3.grid(True, alpha=0.3)
    # Add 50% and 95% percentile lines
    p50 = np.percentile(residual_errors, 50)
    p95 = np.percentile(residual_errors, 95)
    ax3.axvline(p50, color='orange', linestyle='--', label=f'50th Percentile: {p50:.3f}')
    ax3.axvline(p95, color='red', linestyle='--', label=f'95th Percentile: {p95:.3f}')
    ax3.legend()
    
    # 4. Box plot
    ax4.boxplot(residual_errors, vert=True, patch_artist=True, 
                boxprops=dict(facecolor='lightgreen', alpha=0.7))
    ax4.set_ylabel('Residual Error (RE)')
    ax4.set_title('Residual Error Box Plot')
    ax4.grid(True, alpha=0.3)
    ax4.set_xticklabels(['RE Distribution'])
    
    # Add statistics text
    stats_text = f'Statistics:\n'
    stats_text += f'Sample Count: {len(residual_errors)}\n'
    stats_text += f'Mean RE: {residual_errors.mean():.3f}\n'
    stats_text += f'Std Dev: {residual_errors.std():.3f}\n'
    stats_text += f'Min Value: {residual_errors.min():.3f}\n'
    stats_text += f'Max Value: {residual_errors.max():.3f}\n'
    stats_text += f'Median: {np.median(residual_errors):.3f}'
    
    ax4.text(1.1, 0.5, stats_text, transform=ax4.transAxes, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             verticalalignment='center')
    
    plt.suptitle(f'{algorithm_name} - Residual Error (RE) Analysis', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"RE visualization saved as: {save_filename}")
    
    return {
        'mean_re': residual_errors.mean(),
        'std_re': residual_errors.std(),
        'min_re': residual_errors.min(),
        'max_re': residual_errors.max(),
        'median_re': np.median(residual_errors),
        'p95_re': np.percentile(residual_errors, 95)
    }

def calculate_attitude_coverage(data, grid_size=12):
    """计算姿态覆盖度量指标
    
    Parameters:
        data: 原始磁力计数据 (N x 3)
        grid_size: 球面网格大小 (默认12x12)
    
    Returns:
        dict: 包含各种覆盖指标的字典
    """
    # 数据归一化到单位球面
    norms = np.linalg.norm(data, axis=1)
    normalized_data = data / norms[:, np.newaxis]
    
    # 1. 球面覆盖度计算
    # 转换为球面坐标 (theta: 0到pi, phi: 0到2pi)
    theta = np.arccos(np.clip(normalized_data[:, 2], -1, 1))  # 极角
    phi = np.arctan2(normalized_data[:, 1], normalized_data[:, 0]) + np.pi  # 方位角，转换到[0, 2pi]
    
    # 创建球面网格
    theta_bins = np.linspace(0, np.pi, grid_size + 1)
    phi_bins = np.linspace(0, 2 * np.pi, grid_size + 1)
    
    # 统计每个网格的数据点数量
    hist, _, _ = np.histogram2d(theta, phi, bins=[theta_bins, phi_bins])
    covered_grids = np.sum(hist > 0)
    total_grids = grid_size * grid_size
    spherical_coverage = covered_grids / total_grids
    
    # 2. 关键姿态覆盖检测 (六面体)
    key_directions = np.array([
        [1, 0, 0], [-1, 0, 0],   # ±X
        [0, 1, 0], [0, -1, 0],   # ±Y
        [0, 0, 1], [0, 0, -1]    # ±Z
    ])
    
    coverage_threshold = np.cos(np.pi / 6)  # 30度阈值
    covered_faces = 0
    face_coverage_details = []
    
    for i, direction in enumerate(key_directions):
        # 计算与关键方向的夹角余弦值
        dot_products = np.dot(normalized_data, direction)
        close_points = np.sum(dot_products > coverage_threshold)
        is_covered = close_points >= 10  # 至少10个点
        
        if is_covered:
            covered_faces += 1
        
        face_name = ['X+', 'X-', 'Y+', 'Y-', 'Z+', 'Z-'][i]
        face_coverage_details.append({
            'face': face_name,
            'covered': is_covered,
            'point_count': close_points
        })
    
    key_attitude_coverage = covered_faces / 6
    
    # 3. 最大空隙角度计算
    # 使用简化方法：在球面上均匀采样点，找到距离最近数据点最远的采样点
    n_sample = 1000
    # 生成均匀分布的球面采样点
    u = np.random.uniform(0, 1, n_sample)
    v = np.random.uniform(0, 1, n_sample)
    sample_theta = np.arccos(2 * u - 1)
    sample_phi = 2 * np.pi * v
    
    sample_points = np.array([
        np.sin(sample_theta) * np.cos(sample_phi),
        np.sin(sample_theta) * np.sin(sample_phi),
        np.cos(sample_theta)
    ]).T
    
    max_gap_angle = 0
    for sample_point in sample_points:
        # 计算到所有数据点的角距离
        dot_products = np.dot(normalized_data, sample_point)
        dot_products = np.clip(dot_products, -1, 1)
        min_angle = np.min(np.arccos(dot_products))
        max_gap_angle = max(max_gap_angle, min_angle)
    
    max_gap_angle_deg = np.degrees(max_gap_angle)
    
    return {
        'spherical_coverage': spherical_coverage,
        'key_attitude_coverage': key_attitude_coverage,
        'covered_faces': covered_faces,
        'face_coverage_details': face_coverage_details,
        'max_gap_angle_deg': max_gap_angle_deg,
        'total_points': len(data),
        'grid_size': grid_size
    }

def print_attitude_coverage_report(coverage_metrics, dataset_name="Dataset"):
    """打印姿态覆盖报告
    
    Parameters:
        coverage_metrics: calculate_attitude_coverage返回的指标字典
        dataset_name: 数据集名称
    """
    print(f"\n=== {dataset_name} 姿态覆盖分析 ===")
    print(f"数据点总数: {coverage_metrics['total_points']}")
    print(f"球面覆盖率: {coverage_metrics['spherical_coverage']:.1%} ({coverage_metrics['grid_size']}x{coverage_metrics['grid_size']} 网格)")
    print(f"关键姿态覆盖: {coverage_metrics['covered_faces']}/6 面 ({coverage_metrics['key_attitude_coverage']:.1%})")
    print(f"最大空隙角度: {coverage_metrics['max_gap_angle_deg']:.1f}°")
    
    print("\n六面体覆盖详情:")
    for face_info in coverage_metrics['face_coverage_details']:
        status = "✓" if face_info['covered'] else "✗"
        print(f"  {face_info['face']}: {status} ({face_info['point_count']} 点)")
    
    # 校准完成判断
    calibration_ready = (
        coverage_metrics['spherical_coverage'] >= 0.75 and
        coverage_metrics['covered_faces'] >= 5 and
        coverage_metrics['max_gap_angle_deg'] <= 75 and
        coverage_metrics['total_points'] >= 200
    )
    
    print(f"\n校准完成判断: {'✓ 可以开始校准' if calibration_ready else '✗ 需要更多数据'}")
    if not calibration_ready:
        print("建议:")
        if coverage_metrics['spherical_coverage'] < 0.75:
            print(f"  - 球面覆盖率不足 ({coverage_metrics['spherical_coverage']:.1%} < 75%)")
        if coverage_metrics['covered_faces'] < 5:
            print(f"  - 关键姿态覆盖不足 ({coverage_metrics['covered_faces']}/6 < 5/6)")
        if coverage_metrics['max_gap_angle_deg'] > 75:
            print(f"  - 存在较大空隙区域 ({coverage_metrics['max_gap_angle_deg']:.1f}° > 75°)")
        if coverage_metrics['total_points'] < 200:
            print(f"  - 数据点数量不足 ({coverage_metrics['total_points']} < 200)")
    
    return calibration_ready

def calculate_three_axis_coverage(data, axis_threshold_deg=30, min_points_per_direction=10):
    """计算三主轴选择覆盖指标
    
    Parameters:
        data: numpy数组，形状为(n, 3)的磁力计数据
        axis_threshold_deg: 轴向阈值角度（度），默认30度
        min_points_per_direction: 每个方向的最小点数要求
        
    Returns:
        dict: 包含三主轴覆盖指标的字典
    """
    # 数据归一化到单位球面
    norms = np.linalg.norm(data, axis=1)
    normalized_data = data / norms[:, np.newaxis]
    
    # 定义三主轴的正负方向
    axis_directions = {
        'X+': np.array([1, 0, 0]),
        'X-': np.array([-1, 0, 0]),
        'Y+': np.array([0, 1, 0]),
        'Y-': np.array([0, -1, 0]),
        'Z+': np.array([0, 0, 1]),
        'Z-': np.array([0, 0, -1])
    }
    
    axis_threshold_cos = np.cos(np.radians(axis_threshold_deg))
    
    # 计算每个轴向的覆盖情况
    axis_coverage = {}
    total_axis_points = 0
    
    for axis_name, direction in axis_directions.items():
        # 计算与轴向的夹角余弦值
        dot_products = np.dot(normalized_data, direction)
        # 找到在阈值范围内的点
        axis_points = np.sum(dot_products > axis_threshold_cos)
        is_covered = axis_points >= min_points_per_direction
        
        # 计算轴向数据的角度分布
        if axis_points > 0:
            axis_angles = np.arccos(np.clip(dot_products[dot_products > axis_threshold_cos], -1, 1))
            avg_angle = np.degrees(np.mean(axis_angles))
            std_angle = np.degrees(np.std(axis_angles))
        else:
            avg_angle = 0
            std_angle = 0
        
        axis_coverage[axis_name] = {
            'point_count': axis_points,
            'covered': is_covered,
            'avg_angle_deg': avg_angle,
            'std_angle_deg': std_angle
        }
        
        if is_covered:
            total_axis_points += axis_points
    
    # 计算轴向覆盖统计
    covered_directions = sum(1 for info in axis_coverage.values() if info['covered'])
    axis_coverage_ratio = covered_directions / 6
    
    # 计算每个轴（X、Y、Z）的双向覆盖情况
    axes_bilateral_coverage = {}
    for axis in ['X', 'Y', 'Z']:
        pos_covered = axis_coverage[f'{axis}+']['covered']
        neg_covered = axis_coverage[f'{axis}-']['covered']
        bilateral_covered = pos_covered and neg_covered
        
        axes_bilateral_coverage[axis] = {
            'positive_covered': pos_covered,
            'negative_covered': neg_covered,
            'bilateral_covered': bilateral_covered,
            'positive_points': axis_coverage[f'{axis}+']['point_count'],
            'negative_points': axis_coverage[f'{axis}-']['point_count']
        }
    
    bilateral_axes_count = sum(1 for info in axes_bilateral_coverage.values() if info['bilateral_covered'])
    bilateral_coverage_ratio = bilateral_axes_count / 3
    
    # 计算轴向数据分布均匀性
    axis_point_counts = [info['point_count'] for info in axis_coverage.values() if info['covered']]
    if len(axis_point_counts) > 1:
        axis_uniformity = 1 - (np.std(axis_point_counts) / np.mean(axis_point_counts))
        axis_uniformity = max(0, axis_uniformity)  # 确保非负
    else:
        axis_uniformity = 0
    
    # 计算主轴选择质量评分
    # 基于双向覆盖、数据分布均匀性和总点数
    quality_score = (
        bilateral_coverage_ratio * 0.5 +  # 双向覆盖权重50%
        axis_uniformity * 0.3 +           # 均匀性权重30%
        min(total_axis_points / (6 * min_points_per_direction), 1) * 0.2  # 数据充足性权重20%
    )
    
    return {
        'axis_coverage': axis_coverage,
        'axes_bilateral_coverage': axes_bilateral_coverage,
        'covered_directions': covered_directions,
        'axis_coverage_ratio': axis_coverage_ratio,
        'bilateral_axes_count': bilateral_axes_count,
        'bilateral_coverage_ratio': bilateral_coverage_ratio,
        'axis_uniformity': axis_uniformity,
        'total_axis_points': total_axis_points,
        'quality_score': quality_score,
        'axis_threshold_deg': axis_threshold_deg,
        'min_points_per_direction': min_points_per_direction
    }

def print_three_axis_coverage_report(axis_metrics, dataset_name="Dataset"):
    """打印三主轴覆盖报告
    
    Parameters:
        axis_metrics: calculate_three_axis_coverage返回的指标字典
        dataset_name: 数据集名称
    """
    print(f"\n=== {dataset_name} 三主轴覆盖分析 ===")
    print(f"轴向阈值: {axis_metrics['axis_threshold_deg']}°")
    print(f"最小点数要求: {axis_metrics['min_points_per_direction']} 点/方向")
    print(f"轴向覆盖: {axis_metrics['covered_directions']}/6 方向 ({axis_metrics['axis_coverage_ratio']:.1%})")
    print(f"双向覆盖: {axis_metrics['bilateral_axes_count']}/3 轴 ({axis_metrics['bilateral_coverage_ratio']:.1%})")
    print(f"数据均匀性: {axis_metrics['axis_uniformity']:.3f}")
    print(f"质量评分: {axis_metrics['quality_score']:.3f}")
    
    print("\n各轴向覆盖详情:")
    for axis_name, info in axis_metrics['axis_coverage'].items():
        status = "✓" if info['covered'] else "✗"
        print(f"  {axis_name}: {status} ({info['point_count']} 点, 平均角度: {info['avg_angle_deg']:.1f}°)")
    
    print("\n双向覆盖详情:")
    for axis, info in axis_metrics['axes_bilateral_coverage'].items():
        pos_status = "✓" if info['positive_covered'] else "✗"
        neg_status = "✓" if info['negative_covered'] else "✗"
        bilateral_status = "✓" if info['bilateral_covered'] else "✗"
        print(f"  {axis}轴: {bilateral_status} (正向{pos_status}:{info['positive_points']}点, 负向{neg_status}:{info['negative_points']}点)")
    
    # 三主轴校准建议
    axis_calibration_ready = (
        axis_metrics['bilateral_coverage_ratio'] >= 0.67 and  # 至少2/3轴双向覆盖
        axis_metrics['quality_score'] >= 0.6 and              # 质量评分≥0.6
        axis_metrics['total_axis_points'] >= 60               # 轴向总点数≥60
    )
    
    print(f"\n三主轴校准判断: {'✓ 轴向覆盖充分' if axis_calibration_ready else '✗ 轴向覆盖不足'}")
    if not axis_calibration_ready:
        print("建议:")
        if axis_metrics['bilateral_coverage_ratio'] < 0.67:
            print(f"  - 需要更多轴向双向数据 ({axis_metrics['bilateral_axes_count']}/3 < 2/3)")
        if axis_metrics['quality_score'] < 0.6:
            print(f"  - 提高数据质量和均匀性 (评分: {axis_metrics['quality_score']:.3f} < 0.6)")
        if axis_metrics['total_axis_points'] < 60:
            print(f"  - 增加轴向数据点数 ({axis_metrics['total_axis_points']} < 60)")
    
    return axis_calibration_ready

def visualize_data_normalization(data, title_prefix="Data", save_path=None):
    """可视化数据归一化前后的对比
    
    Parameters:
        data: numpy数组，形状为(n, 3)的磁力计数据
        title_prefix: 图表标题前缀
        save_path: 保存路径，如果为None则不保存
    """
    import matplotlib.pyplot as plt
    
    # 计算归一化数据
    norms = np.linalg.norm(data, axis=1)
    normalized_data = data / norms[:, np.newaxis]
    
    # 创建图表
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'{title_prefix} - Before/After Normalization Comparison', fontsize=16)
    
    # 原始数据的三个投影平面
    views = [('X', 'Y', 0, 1), ('X', 'Z', 0, 2), ('Y', 'Z', 1, 2)]
    
    for i, (xlabel, ylabel, x_idx, y_idx) in enumerate(views):
        # 原始数据
        ax1 = axes[0, i]
        ax1.scatter(data[:, x_idx], data[:, y_idx], s=2, alpha=0.6, c='blue')
        ax1.set_title(f'Original Data - {xlabel}{ylabel} Plane')
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel)
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')
        
        # 归一化数据
        ax2 = axes[1, i]
        ax2.scatter(normalized_data[:, x_idx], normalized_data[:, y_idx], s=2, alpha=0.6, c='red')
        ax2.set_title(f'Normalized Data - {xlabel}{ylabel} Plane')
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel(ylabel)
        ax2.grid(True, alpha=0.3)
        ax2.axis('equal')
        
        # 在归一化数据上绘制单位圆
        theta = np.linspace(0, 2*np.pi, 100)
        ax2.plot(np.cos(theta), np.sin(theta), 'k--', linewidth=1, alpha=0.7, label='Unit Circle')
        ax2.legend()
        ax2.set_xlim(-1.2, 1.2)
        ax2.set_ylim(-1.2, 1.2)
    
    # 添加统计信息
    stats_text = f"""Data Statistics:
Original Data Range: X[{data[:,0].min():.1f}, {data[:,0].max():.1f}], Y[{data[:,1].min():.1f}, {data[:,1].max():.1f}], Z[{data[:,2].min():.1f}, {data[:,2].max():.1f}]
Original Data Average Radius: {norms.mean():.1f}
Normalized Radius: 1.0 (All points on unit sphere)
Total Data Points: {len(data)}"""
    
    fig.text(0.02, 0.02, stats_text, fontsize=10, verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Normalization comparison plot saved as: {save_path}")
    
    plt.show()
    
    return normalized_data

def visualize_normalization_comparison(original_data, unit_vector_data, min_max_data, offset_norm_data):
    """
    Visualize comparison between three normalization methods
    
    Args:
        original_data: Original magnetometer data
        unit_vector_data: Unit vector normalized data
        min_max_data: Min-Max normalized data
        offset_norm_data: Offset-Norm normalized data
    """
    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    fig.suptitle('Three Normalization Methods Comparison', fontsize=16, fontweight='bold')
    
    # Define projection pairs and labels
    projections = [(0, 1, 'XY'), (0, 2, 'XZ'), (1, 2, 'YZ')]
    
    # Row 1: Unit Vector Normalization
    for i, (x_idx, y_idx, label) in enumerate(projections):
        ax = axes[0, i]
        
        # Plot original data
        ax.scatter(original_data[:, x_idx], original_data[:, y_idx], 
                  alpha=0.6, s=20, c='lightblue', label='Original', edgecolors='blue', linewidth=0.5)
        
        # Plot unit vector normalized data
        ax.scatter(unit_vector_data[:, x_idx], unit_vector_data[:, y_idx], 
                  alpha=0.8, s=15, c='red', label='Unit Vector', marker='x')
        
        # Add unit circle for reference
        circle = plt.Circle((0, 0), 1, fill=False, color='red', linestyle='--', alpha=0.7)
        ax.add_patch(circle)
        
        ax.set_xlabel(f'{label[0]} axis')
        ax.set_ylabel(f'{label[1]} axis')
        ax.set_title(f'Unit Vector Normalization - {label} Plane')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_aspect('equal')
        
        # Set axis limits
        max_range = max(np.max(np.abs(original_data[:, [x_idx, y_idx]])), 1.2)
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
    
    # Row 2: Min-Max Normalization
    for i, (x_idx, y_idx, label) in enumerate(projections):
        ax = axes[1, i]
        
        # Plot original data (normalized to [0,1] for comparison)
        orig_min = np.min(original_data[:, [x_idx, y_idx]], axis=0)
        orig_max = np.max(original_data[:, [x_idx, y_idx]], axis=0)
        orig_range = orig_max - orig_min
        orig_range[orig_range == 0] = 1
        orig_normalized = (original_data[:, [x_idx, y_idx]] - orig_min) / orig_range
        
        ax.scatter(orig_normalized[:, 0], orig_normalized[:, 1], 
                  alpha=0.6, s=20, c='lightgreen', label='Original (scaled)', edgecolors='green', linewidth=0.5)
        
        # Plot Min-Max normalized data
        ax.scatter(min_max_data[:, x_idx], min_max_data[:, y_idx], 
                  alpha=0.8, s=15, c='orange', label='Min-Max', marker='s')
        
        # Add unit square for reference
        square = plt.Rectangle((0, 0), 1, 1, fill=False, color='orange', linestyle='--', alpha=0.7)
        ax.add_patch(square)
        
        ax.set_xlabel(f'{label[0]} axis')
        ax.set_ylabel(f'{label[1]} axis')
        ax.set_title(f'Min-Max Normalization - {label} Plane')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_aspect('equal')
        
        # Set axis limits for [0,1] range
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
    
    # Row 3: Offset-Norm Normalization
    for i, (x_idx, y_idx, label) in enumerate(projections):
        ax = axes[2, i]
        
        # Plot original data (centered for comparison)
        orig_centered = original_data - np.mean(original_data, axis=0)
        ax.scatter(orig_centered[:, x_idx], orig_centered[:, y_idx], 
                  alpha=0.6, s=20, c='lightcyan', label='Original (centered)', edgecolors='cyan', linewidth=0.5)
        
        # Plot Offset-Norm normalized data
        ax.scatter(offset_norm_data[:, x_idx], offset_norm_data[:, y_idx], 
                  alpha=0.8, s=15, c='purple', label='Offset-Norm', marker='^')
        
        # Add unit circle for reference
        circle = plt.Circle((0, 0), 1, fill=False, color='purple', linestyle='--', alpha=0.7)
        ax.add_patch(circle)
        
        ax.set_xlabel(f'{label[0]} axis')
        ax.set_ylabel(f'{label[1]} axis')
        ax.set_title(f'Offset-Norm Normalization - {label} Plane')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_aspect('equal')
        
        # Set axis limits
        max_range = max(np.max(np.abs(orig_centered[:, [x_idx, y_idx]])), 1.2)
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
    
    # Add statistics text
    stats_text = f"""Statistics Summary:
    
Unit Vector Normalization:
    • All points on unit sphere (radius = 1.0)
    • Preserves original directions
    • Good for attitude analysis
    
Min-Max Normalization:
    • Data mapped to [0,1] cube
    • Preserves relative distances
    • Good for feature scaling
    
Offset-Norm Normalization:
    • Centers data then projects to unit sphere
    • Removes systematic bias first
    • Good for biased data"""
    
    fig.text(0.02, 0.02, stats_text, fontsize=9, 
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    plt.show()
    
    print("\n=== Visualization Complete ===")
    print("The plots show:")
    print("• Top row: Unit vector normalization (projects to unit sphere)")
    print("• Middle row: Min-Max normalization (maps to [0,1] cube)")
    print("• Bottom row: Offset-Norm normalization (centers then projects to unit sphere)")
    print("• Dashed circles: Unit sphere projections")
    print("• Dashed squares: Unit cube projections")
    
    return fig