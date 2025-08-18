#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速磁力计数据覆盖范围分析工具
使用Offset-Norm归一化和扇面分析方法
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from offset_norm_coverage_analysis import comprehensive_coverage_analysis, visualize_coverage_analysis

def quick_analyze(file_path, mag_columns=None, num_sectors=8, min_radius_threshold=0.3, show_details=True):
    """
    快速分析磁力计数据覆盖范围
    
    Parameters:
        file_path: 数据文件路径
        mag_columns: 磁力计数据列名或索引，如['mag_x', 'mag_y', 'mag_z']或[0,1,2]
        num_sectors: 扇面数量 (默认8)
        min_radius_threshold: 最小半径阈值 (默认0.3)
        show_details: 是否显示详细信息 (默认True)
    
    Returns:
        analysis_results: 分析结果字典
    """
    
    print(f"🔍 Analyzing: {os.path.basename(file_path)}")
    print("-" * 60)
    
    # 1. 加载数据
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_table(file_path, sep='\s+|,|;', engine='python')
        
        if show_details:
            print(f"📊 Data shape: {df.shape}")
            print(f"📋 Columns: {list(df.columns)}")
        
        # 自动检测磁力计数据列
        if mag_columns is None:
            # 首先尝试按列名匹配
            possible_patterns = [
                ['mag_x', 'mag_y', 'mag_z'],
                ['magX', 'magY', 'magZ'], 
                ['mx', 'my', 'mz'],
                ['x', 'y', 'z']
            ]
            
            for pattern in possible_patterns:
                try:
                    if all(col in df.columns for col in pattern):
                        mag_columns = pattern
                        break
                except:
                    continue
            
            # 如果没有找到匹配的列名，尝试智能检测
            if mag_columns is None:
                # 检查是否有时间列，如果有则跳过
                time_patterns = ['time', 'timestamp', 'collecting']
                non_time_columns = []
                for col in df.columns:
                    col_lower = str(col).lower()
                    if not any(pattern in col_lower for pattern in time_patterns):
                        non_time_columns.append(col)
                
                # 如果有足够的非时间列，使用前3列
                if len(non_time_columns) >= 3:
                    mag_columns = non_time_columns[:3]
                # 否则假设后3列是磁力计数据（跳过第一列时间戳）
                elif df.shape[1] >= 4:
                    mag_columns = [1, 2, 3]  # 使用索引
                elif df.shape[1] >= 3:
                    mag_columns = [0, 1, 2]  # 使用索引
                else:
                    raise ValueError("Cannot detect magnetometer columns automatically")
            
            if mag_columns is None:
                raise ValueError("Cannot detect magnetometer columns automatically")
        
        # 提取数据
        if isinstance(mag_columns[0], str):
            data = df[mag_columns].values
        else:
            data = df.iloc[:, mag_columns].values
        
        # 清理数据
        valid_mask = ~(np.isnan(data).any(axis=1) | np.isinf(data).any(axis=1))
        data = data[valid_mask]
        
        if len(data) == 0:
            raise ValueError("No valid data points found")
        
        if show_details:
            print(f"✅ Valid data points: {len(data)}")
            print(f"📏 Data ranges:")
            for i, axis in enumerate(['X', 'Y', 'Z']):
                min_val, max_val = data[:, i].min(), data[:, i].max()
                print(f"   {axis}: [{min_val:8.3f}, {max_val:8.3f}] (range: {max_val-min_val:8.3f})")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None
    
    # 2. 进行覆盖分析
    print(f"\n🎯 Coverage Analysis (sectors={num_sectors}, threshold={min_radius_threshold})")
    print("-" * 60)
    
    try:
        results = comprehensive_coverage_analysis(
            data, 
            num_sectors=num_sectors,
            min_radius_threshold=min_radius_threshold
        )
        
        # 3. 显示关键结果
        print(f"\n📈 Results Summary:")
        print(f"   Valid projection planes: {len(results['valid_planes'])}/3")
        print(f"   Average coverage: {results['average_coverage']:.1%}")
        print(f"   Coverage quality: {results['coverage_quality']}")
        
        if show_details:
            print(f"\n📊 Plane Details:")
            for plane_name in ['XY', 'XZ', 'YZ']:
                plane_data = results['plane_results'][plane_name]
                radius = plane_data['radius']
                is_valid = plane_data['valid']
                
                if is_valid:
                    coverage = plane_data['coverage_ratio']
                    covered_sectors = plane_data['covered_sectors']
                    print(f"   {plane_name}: ✅ R={radius:.3f}, Coverage={coverage:.1%} ({covered_sectors}/{num_sectors} sectors)")
                else:
                    print(f"   {plane_name}: ❌ R={radius:.3f} < {min_radius_threshold} (invalid)")
        
        # 4. 生成可视化
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        save_path = f"{base_name}_quick_analysis.png"
        
        print(f"\n🎨 Generating visualization: {save_path}")
        visualize_coverage_analysis(results, save_path)
        
        # 5. 给出建议
        print(f"\n💡 Recommendations:")
        if results['coverage_quality'] == 'Excellent':
            print("   🟢 Excellent coverage! Ready for calibration.")
        elif results['coverage_quality'] == 'Good':
            print("   🟡 Good coverage. Can proceed with calibration.")
        elif results['coverage_quality'] == 'Fair':
            print("   🟠 Fair coverage. Consider collecting more data in uncovered areas.")
        else:
            print("   🔴 Poor coverage. More data collection recommended.")
        
        if len(results['valid_planes']) < 3:
            print("   ⚠️  Some projection planes have insufficient data spread.")
            print("   💭 Try rotating the device in more orientations.")
        
        return results
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return None

def main():
    """
    命令行接口
    """
    if len(sys.argv) < 2:
        print("Usage: python quick_coverage_analysis.py <data_file> [options]")
        print("")
        print("Options:")
        print("  --sectors N        Number of sectors (default: 8)")
        print("  --threshold T      Minimum radius threshold (default: 0.3)")
        print("  --columns C1,C2,C3 Magnetometer column names or indices")
        print("  --quiet            Show only summary results")
        print("")
        print("Examples:")
        print("  python quick_coverage_analysis.py data.csv")
        print("  python quick_coverage_analysis.py data.csv --sectors 12 --threshold 0.2")
        print("  python quick_coverage_analysis.py data.csv --columns mx,my,mz")
        print("  python quick_coverage_analysis.py data.csv --columns 0,1,2 --quiet")
        return
    
    file_path = sys.argv[1]
    
    # 解析参数
    num_sectors = 8
    min_radius_threshold = 0.3
    mag_columns = None
    show_details = True
    
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == '--sectors' and i + 1 < len(sys.argv):
            num_sectors = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == '--threshold' and i + 1 < len(sys.argv):
            min_radius_threshold = float(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == '--columns' and i + 1 < len(sys.argv):
            cols = sys.argv[i + 1].split(',')
            # 尝试转换为整数索引
            try:
                mag_columns = [int(c) for c in cols]
            except ValueError:
                mag_columns = cols
            i += 2
        elif sys.argv[i] == '--quiet':
            show_details = False
            i += 1
        else:
            print(f"Unknown option: {sys.argv[i]}")
            return
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return
    
    # 执行分析
    print("=" * 80)
    print("🚀 Quick Magnetometer Coverage Analysis")
    print("=" * 80)
    
    result = quick_analyze(
        file_path=file_path,
        mag_columns=mag_columns,
        num_sectors=num_sectors,
        min_radius_threshold=min_radius_threshold,
        show_details=show_details
    )
    
    if result:
        print("\n" + "=" * 80)
        print("✅ Analysis completed successfully!")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("❌ Analysis failed!")
        print("=" * 80)

if __name__ == "__main__":
    main()