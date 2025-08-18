#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Offset-Norm归一化覆盖范围分析
通过将三个投影圆分成扇面，统计有落点的扇面占比来评估覆盖范围
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
import math

def offset_norm_normalize(data):
    """
    Offset-Norm归一化：先减去偏移量，再除以范数
    
    Parameters:
        data: numpy数组，形状为(n, 3)的磁力计数据
    
    Returns:
        normalized_data: 归一化后的数据
        offset: 计算出的偏移量
    """
    # 计算偏移量（均值）
    offset = np.mean(data, axis=0)
    
    # 减去偏移量
    centered_data = data - offset
    
    # 计算范数并归一化
    norms = np.linalg.norm(centered_data, axis=1, keepdims=True)
    norms[norms == 0] = 1  # 避免除零
    normalized_data = centered_data / norms
    
    return normalized_data, offset

def calculate_projection_radius(data, plane_indices):
    """
    计算指定平面投影的有效半径
    
    Parameters:
        data: 归一化后的数据
        plane_indices: 投影平面的轴索引，如(0,1)表示XY平面
    
    Returns:
        radius: 投影圆的有效半径
    """
    projection = data[:, plane_indices]
    distances = np.linalg.norm(projection, axis=1)
    
    # 使用95%分位数作为有效半径，排除异常值
    radius = np.percentile(distances, 95)
    
    return radius

def analyze_sector_coverage(data, plane_indices, num_sectors=8, min_radius_threshold=0.3):
    """
    分析指定投影平面的扇面覆盖情况
    
    Parameters:
        data: 归一化后的数据
        plane_indices: 投影平面的轴索引，如(0,1)表示XY平面
        num_sectors: 扇面数量，默认8个
        min_radius_threshold: 最小有效半径阈值
    
    Returns:
        coverage_info: 包含覆盖信息的字典
    """
    projection = data[:, plane_indices]
    
    # 计算投影半径
    radius = calculate_projection_radius(data, plane_indices)
    
    # 检查半径是否足够大
    if radius < min_radius_threshold:
        return {
            'plane_name': f'{"XYZ"[plane_indices[0]]}{"XYZ"[plane_indices[1]]}',
            'radius': radius,
            'valid': False,
            'reason': f'Projection radius {radius:.3f} < threshold {min_radius_threshold}',
            'covered_sectors': 0,
            'total_sectors': num_sectors,
            'coverage_ratio': 0.0,
            'sector_counts': [0] * num_sectors
        }
    
    # 计算每个点的角度
    angles = np.arctan2(projection[:, 1], projection[:, 0])
    # 将角度转换到[0, 2π]范围
    angles = (angles + 2 * np.pi) % (2 * np.pi)
    
    # 计算扇面大小
    sector_size = 2 * np.pi / num_sectors
    
    # 统计每个扇面的点数
    sector_counts = [0] * num_sectors
    for angle in angles:
        sector_idx = int(angle / sector_size)
        if sector_idx >= num_sectors:  # 处理边界情况
            sector_idx = num_sectors - 1
        sector_counts[sector_idx] += 1
    
    # 计算覆盖的扇面数量
    covered_sectors = sum(1 for count in sector_counts if count > 0)
    coverage_ratio = covered_sectors / num_sectors
    
    return {
        'plane_name': f'{"XYZ"[plane_indices[0]]}{"XYZ"[plane_indices[1]]}',
        'radius': radius,
        'valid': True,
        'covered_sectors': covered_sectors,
        'total_sectors': num_sectors,
        'coverage_ratio': coverage_ratio,
        'sector_counts': sector_counts,
        'angles': angles,
        'projection': projection
    }

def comprehensive_coverage_analysis(data, num_sectors=8, min_radius_threshold=0.3):
    """
    对三个投影平面进行全面的覆盖范围分析
    
    Parameters:
        data: 原始磁力计数据
        num_sectors: 扇面数量
        min_radius_threshold: 最小有效半径阈值
    
    Returns:
        analysis_results: 分析结果字典
    """
    # 进行Offset-Norm归一化
    normalized_data, offset = offset_norm_normalize(data)
    
    # 定义三个投影平面
    planes = [
        (0, 1, 'XY'),  # XY平面
        (0, 2, 'XZ'),  # XZ平面
        (1, 2, 'YZ')   # YZ平面
    ]
    
    plane_results = {}
    valid_planes = []
    total_coverage_score = 0
    
    print("=" * 70)
    print("Offset-Norm Normalization Coverage Analysis")
    print("=" * 70)
    
    print(f"\nOriginal data points: {len(data)}")
    print(f"Calculated offset: [{offset[0]:.3f}, {offset[1]:.3f}, {offset[2]:.3f}]")
    print(f"Sector division: {num_sectors} sectors per plane")
    print(f"Minimum radius threshold: {min_radius_threshold}")
    
    print("\n" + "="*50)
    print("Projection Plane Analysis")
    print("="*50)
    
    for plane_idx, (i, j, name) in enumerate(planes):
        result = analyze_sector_coverage(normalized_data, (i, j), num_sectors, min_radius_threshold)
        plane_results[name] = result
        
        print(f"\n【{name} Plane】")
        print(f"  Projection radius: {result['radius']:.3f}")
        
        if result['valid']:
            print(f"  ✅ Valid projection")
            print(f"  Covered sectors: {result['covered_sectors']}/{result['total_sectors']}")
            print(f"  Coverage ratio: {result['coverage_ratio']:.1%}")
            
            # 显示每个扇面的点数
            print(f"  Sector distribution:")
            for sector_idx, count in enumerate(result['sector_counts']):
                angle_start = sector_idx * 360 / num_sectors
                angle_end = (sector_idx + 1) * 360 / num_sectors
                status = "✓" if count > 0 else "✗"
                print(f"    Sector {sector_idx+1} ({angle_start:3.0f}°-{angle_end:3.0f}°): {status} {count:2d} points")
            
            valid_planes.append(name)
            total_coverage_score += result['coverage_ratio']
        else:
            print(f"  ❌ Invalid projection: {result['reason']}")
    
    # 计算综合覆盖评分
    if valid_planes:
        average_coverage = total_coverage_score / len(valid_planes)
        coverage_quality = "Excellent" if average_coverage >= 0.875 else \
                          "Good" if average_coverage >= 0.75 else \
                          "Fair" if average_coverage >= 0.625 else "Poor"
    else:
        average_coverage = 0
        coverage_quality = "No valid projections"
    
    print("\n" + "="*50)
    print("Coverage Summary")
    print("="*50)
    print(f"Valid projection planes: {len(valid_planes)}/3 ({', '.join(valid_planes)})")
    if valid_planes:
        print(f"Average coverage ratio: {average_coverage:.1%}")
        print(f"Coverage quality: {coverage_quality}")
    
    # 覆盖建议
    print(f"\n💡 Coverage Assessment:")
    if len(valid_planes) == 3 and average_coverage >= 0.75:
        print(f"   ✅ Excellent coverage - Ready for calibration")
    elif len(valid_planes) >= 2 and average_coverage >= 0.625:
        print(f"   ⚠️  Acceptable coverage - Can proceed with caution")
    else:
        print(f"   ❌ Insufficient coverage - Need more data")
        
        if len(valid_planes) < 2:
            print(f"   • Collect data with larger attitude variations")
        if average_coverage < 0.625:
            print(f"   • Ensure more uniform distribution across all directions")
    
    return {
        'original_data': data,
        'normalized_data': normalized_data,
        'offset': offset,
        'plane_results': plane_results,
        'valid_planes': valid_planes,
        'average_coverage': average_coverage,
        'coverage_quality': coverage_quality,
        'num_sectors': num_sectors,
        'min_radius_threshold': min_radius_threshold
    }

def visualize_coverage_analysis(analysis_results, save_path=None):
    """
    可视化覆盖范围分析结果
    
    Parameters:
        analysis_results: comprehensive_coverage_analysis返回的结果
        save_path: 保存路径，如果为None则不保存
    """
    original_data = analysis_results['original_data']
    normalized_data = analysis_results['normalized_data']
    plane_results = analysis_results['plane_results']
    num_sectors = analysis_results['num_sectors']
    
    # 创建图表 - 2行3列布局
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Magnetometer Data: Before and After Offset-Norm Normalization', fontsize=16, fontweight='bold')
    
    planes = ['XY', 'XZ', 'YZ']
    colors = ['blue', 'red', 'green']
    
    # 第一行：显示原始数据投影
    for idx, plane_name in enumerate(planes):
        # 从plane_results中获取对应的投影索引
        result = plane_results[plane_name]
        if result['valid']:
            # 根据平面名称确定索引
            if plane_name == 'XY':
                i, j = 0, 1
            elif plane_name == 'XZ':
                i, j = 0, 2
            else:  # YZ
                i, j = 1, 2
        else:
            # 对于无效平面，仍然显示原始数据
            if plane_name == 'XY':
                i, j = 0, 1
            elif plane_name == 'XZ':
                i, j = 0, 2
            else:  # YZ
                i, j = 1, 2
        ax = axes[0, idx]
        
        # 绘制真正的原始数据投影（未经任何处理）
        ax.scatter(original_data[:, i], original_data[:, j], 
                  c=colors[idx], alpha=0.6, s=8, 
                  label=f'Raw Original Data ({len(original_data)} points)')
        
        ax.set_title(f'Original Data - {plane_name} Plane', fontsize=12, fontweight='bold')
        ax.set_xlabel(f'{plane_name[0]} axis')
        ax.set_ylabel(f'{plane_name[1]} axis')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_aspect('equal')
    
    # 第二行：显示归一化后数据和扇面分析
    for idx, plane_name in enumerate(planes):
        ax = axes[1, idx]
        result = plane_results[plane_name]
        
        if result['valid']:
            projection = result['projection']
            angles = result['angles']
            sector_counts = result['sector_counts']
            
            # 绘制扇面分割线和覆盖情况
            sector_size = 2 * np.pi / num_sectors
            for sector_idx in range(num_sectors):
                angle_start = sector_idx * sector_size
                angle_end = (sector_idx + 1) * sector_size
                
                # 扇面颜色：有数据点的为绿色，无数据点的为红色
                sector_color = 'lightgreen' if sector_counts[sector_idx] > 0 else 'lightcoral'
                sector_alpha = 0.3
                
                # 绘制扇面
                wedge = Wedge((0, 0), 1.2, 
                             np.degrees(angle_start), np.degrees(angle_end),
                             facecolor=sector_color, alpha=sector_alpha, 
                             edgecolor='black', linewidth=0.5)
                ax.add_patch(wedge)
                
                # 在扇面中心添加扇面编号
                mid_angle = (angle_start + angle_end) / 2
                text_radius = 0.9
                text_x = text_radius * np.cos(mid_angle)
                text_y = text_radius * np.sin(mid_angle)
                ax.text(text_x, text_y, str(sector_idx + 1), 
                       ha='center', va='center', fontsize=10, fontweight='bold')
            
            # 绘制归一化后的数据投影点
            ax.scatter(projection[:, 0], projection[:, 1], 
                      c=colors[idx], alpha=0.6, s=8, 
                      label=f'Normalized Data ({len(projection)} points)')
            
            # 绘制单位圆
            circle = plt.Circle((0, 0), 1, fill=False, color='black', 
                               linestyle='--', linewidth=2, alpha=0.7, label='Unit sphere projection')
            ax.add_patch(circle)
            
            # 设置标题和标签
            coverage_ratio = result['coverage_ratio']
            ax.set_title(f'Normalized Data - {plane_name} Plane\nCoverage: {result["covered_sectors"]}/{num_sectors} ({coverage_ratio:.1%})\nRadius: {result["radius"]:.3f}')
            
        else:
            # 即使无效，也显示归一化后的数据点
            if plane_name == 'XY':
                i, j = 0, 1
            elif plane_name == 'XZ':
                i, j = 0, 2
            else:  # YZ
                i, j = 1, 2
            
            # 显示归一化数据的投影
            norm_projection = normalized_data[:, [i, j]]
            ax.scatter(norm_projection[:, 0], norm_projection[:, 1], 
                      c=colors[idx], alpha=0.4, s=8, 
                      label=f'Normalized Data ({len(norm_projection)} points)')
            
            # 绘制单位圆
            circle = plt.Circle((0, 0), 1, fill=False, color='black', 
                               linestyle='--', linewidth=2, alpha=0.5)
            ax.add_patch(circle)
            
            # 添加无效原因的文本
            ax.text(0, -1.4, f'❌ Invalid: {result["reason"]}', 
                   ha='center', va='center', fontsize=10, 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7))
            
            ax.set_title(f'Normalized Data - {plane_name} Plane\nRadius: {result["radius"]:.3f} (Invalid)', 
                        fontsize=12, fontweight='bold', color='red')
            ax.legend(loc='upper right')
        
        ax.set_xlabel(f'{plane_name[0]} axis (normalized)')
        ax.set_ylabel(f'{plane_name[1]} axis (normalized)')
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        if result['valid']:
            ax.legend(loc='upper right', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    plt.show()
    
    return fig

def generate_test_data():
    """
    生成测试数据：模拟有偏差的磁力计数据
    """
    np.random.seed(42)
    
    # 生成球面上的随机点
    n_points = 200
    
    # 使用球坐标生成均匀分布的点
    u = np.random.uniform(0, 1, n_points)
    v = np.random.uniform(0, 1, n_points)
    
    theta = 2 * np.pi * u  # 方位角
    phi = np.arccos(2 * v - 1)  # 极角
    
    # 转换为笛卡尔坐标
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    
    # 添加椭球变形（软铁效应）
    scale = np.array([2.0, 1.5, 1.2])
    data = np.column_stack([x, y, z]) * scale
    
    # 添加硬铁偏差
    hard_iron_bias = np.array([3.0, 2.0, 1.0])
    data += hard_iron_bias
    
    # 添加噪声
    noise = np.random.normal(0, 0.05, data.shape)
    data += noise
    
    return data

def main():
    """
    主函数：演示Offset-Norm归一化覆盖范围分析
    """
    # 生成测试数据
    test_data = generate_test_data()
    
    print("Generated test data with:")
    print(f"  • {len(test_data)} data points")
    print(f"  • Hard iron bias: [3.0, 2.0, 1.0]")
    print(f"  • Soft iron scaling: [2.0, 1.5, 1.2]")
    print(f"  • Added noise: σ=0.05")
    
    # 进行覆盖范围分析
    results = comprehensive_coverage_analysis(
        test_data, 
        num_sectors=8,  # 8个扇面
        min_radius_threshold=0.3  # 最小半径阈值
    )
    
    # 可视化结果
    print("\n" + "="*50)
    print("Generating visualization...")
    print("="*50)
    
    visualize_coverage_analysis(results)
    
    print("\n=== Analysis Complete ===")
    print("The visualization shows:")
    print("• Green sectors: Have data points (covered)")
    print("• Red sectors: No data points (uncovered)")
    print("• Numbers in sectors: Sector indices (1-8)")
    print("• Dashed circle: Unit sphere projection")
    print("\nUse this analysis to assess magnetometer calibration data quality!")
    
    return results

if __name__ == "__main__":
    results = main()