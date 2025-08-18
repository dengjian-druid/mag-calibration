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
import argparse
from sklearn.model_selection import train_test_split
from offset_norm_coverage_analysis import comprehensive_coverage_analysis, visualize_coverage_analysis

def quick_analyze(file_path, mag_columns=None, num_sectors=8, min_radius=0.3, show_details=True, 
                 start_row=None, end_row=None, train_size=None):
    """
    快速分析磁力计数据覆盖范围
    
    Parameters:
        file_path: 数据文件路径
        mag_columns: 磁力计数据列名或索引，如['mag_x', 'mag_y', 'mag_z']或[0,1,2]
        num_sectors: 扇面数量 (默认8)
        min_radius: 最小半径阈值 (默认0.3)
        show_details: 是否显示详细信息 (默认True)
        start_row: 数据起始行
        end_row: 数据结束行
        train_size: 训练集大小
    
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
            df = pd.read_table(file_path, sep=r'\s+|,|;', engine='python')
        
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
        

        
        # 数据截取
        if start_row is not None or end_row is not None:
            start_idx = start_row if start_row is not None else 0
            end_idx = end_row if end_row is not None else len(data)
            data = data[start_idx:end_idx]
            if show_details:
                print(f"📏 Data slice: [{start_idx}:{end_idx}] -> {len(data)} points")
        
        # 数据分割为训练集和测试集
        if train_size is not None:
            if train_size >= len(data):
                print(f"⚠️  Warning: train_size ({train_size}) >= total data ({len(data)}), using all data as training set")
                train_data = data
                test_data = None
            else:
                # 随机分割数据
                indices = np.arange(len(data))
                train_indices, test_indices = train_test_split(
                    indices, train_size=train_size, random_state=42, shuffle=True
                )
                train_data = data[train_indices]
                test_data = data[test_indices]
                
                if show_details:
                    print(f"🔀 Data split: {len(train_data)} training + {len(test_data)} testing")
        else:
            train_data = data
            test_data = None
        
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
    print(f"\n🎯 Coverage Analysis (sectors={num_sectors}, min_radius={min_radius})")
    print("-" * 60)
    
    def analyze_dataset(dataset, dataset_name):
        """分析单个数据集"""
        print(f"\n📊 {dataset_name} Analysis:")
        print("-" * 40)
        
        results = comprehensive_coverage_analysis(
            dataset, 
            num_sectors=num_sectors,
            min_radius=min_radius
        )
        
        # 显示关键结果
        print(f"   Valid projection planes: {len(results['valid_planes'])}/3")
        print(f"   Average coverage: {results['average_coverage']:.1%}")
        print(f"   Coverage quality: {results['coverage_quality']}")
        
        if show_details:
            print(f"   Plane Details:")
            for plane_name in ['XY', 'XZ', 'YZ']:
                plane_data = results['plane_results'][plane_name]
                radius = plane_data['radius']
                is_valid = plane_data['valid']
                
                if is_valid:
                    coverage = plane_data['coverage_ratio']
                    covered_sectors = plane_data['covered_sectors']
                    print(f"     {plane_name}: ✅ R={radius:.3f}, Coverage={coverage:.1%} ({covered_sectors}/{num_sectors} sectors)")
                else:
                    print(f"     {plane_name}: ❌ R={radius:.3f} < {min_radius} (invalid)")
        
        return results
    
    try:
        # 分析训练集
        train_results = analyze_dataset(train_data, "Training Set")
        
        # 分析测试集（如果存在）
        test_results = None
        if test_data is not None:
            test_results = analyze_dataset(test_data, "Test Set")
        
        # 4. 生成可视化
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # 为训练集生成可视化
        train_save_path = f"{base_name}_train_analysis.png"
        print(f"\n🎨 Generating training set visualization: {train_save_path}")
        visualize_coverage_analysis(train_results, train_save_path)
        
        # 为测试集生成可视化（如果存在）
        if test_results is not None:
            test_save_path = f"{base_name}_test_analysis.png"
            print(f"🎨 Generating test set visualization: {test_save_path}")
            visualize_coverage_analysis(test_results, test_save_path)
        
        # 5. 给出建议
        print(f"\n💡 Recommendations:")
        
        def print_recommendations(results, dataset_name):
            print(f"   {dataset_name}:")
            if results['coverage_quality'] == 'Excellent':
                print(f"     🟢 Excellent coverage! Ready for calibration.")
            elif results['coverage_quality'] == 'Good':
                print(f"     🟡 Good coverage. Can proceed with calibration.")
            elif results['coverage_quality'] == 'Fair':
                print(f"     🟠 Fair coverage. Consider collecting more data in uncovered areas.")
            else:
                print(f"     🔴 Poor coverage. More data collection recommended.")
            
            if len(results['valid_planes']) < 3:
                print(f"     ⚠️  Some projection planes have insufficient data spread.")
                print(f"     💭 Try rotating the device in more orientations.")
        
        print_recommendations(train_results, "Training Set")
        if test_results is not None:
            print_recommendations(test_results, "Test Set")
        
        # 返回结果
        if test_results is not None:
            return {
                'train_results': train_results,
                'test_results': test_results,
                'train_data': train_data,
                'test_data': test_data
            }
        else:
            return {
                'train_results': train_results,
                'test_results': None,
                'train_data': train_data,
                'test_data': None
            }
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return None

def main():
    """
    命令行接口
    """
    parser = argparse.ArgumentParser(
        description='Quick Magnetometer Coverage Analysis Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python quick_coverage_analysis.py data.csv
  python quick_coverage_analysis.py data.csv --sectors 12 --min-radius 0.2
  python quick_coverage_analysis.py data.csv --columns mx,my,mz
  python quick_coverage_analysis.py data.csv -s 100 -e 1000 -n 500
  python quick_coverage_analysis.py -f data.csv -s 0 -e 2000 -n 800 --quiet
        """
    )
    
    # 位置参数
    parser.add_argument('file', nargs='?', help='Data file path')
    
    # 新增的参数
    parser.add_argument('-f', '--file', dest='file_path', help='Data file path (alternative to positional argument)')
    parser.add_argument('-s', '--start', type=int, help='Start row index for data slice')
    parser.add_argument('-e', '--end', type=int, help='End row index for data slice')
    parser.add_argument('-n', '--train-size', type=int, help='Number of samples for training set')
    
    # 原有参数
    parser.add_argument('--sectors', type=int, default=8, help='Number of sectors (default: 8)')
    parser.add_argument('--min-radius', type=float, default=0.3, help='Minimum radius for valid points (default: 0.3)')
    parser.add_argument('--columns', help='Magnetometer column names or indices (comma-separated)')
    parser.add_argument('--quiet', action='store_true', help='Show only summary results')
    
    args = parser.parse_args()
    
    # 确定文件路径
    file_path = args.file_path or args.file
    if not file_path:
        parser.error('Data file path is required (use positional argument or -f/--file)')
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return
    
    # 处理列参数
    mag_columns = None
    if args.columns:
        cols = args.columns.split(',')
        try:
            mag_columns = [int(c) for c in cols]
        except ValueError:
            mag_columns = cols
    
    # 执行分析
    print("=" * 80)
    print("🚀 Quick Magnetometer Coverage Analysis")
    if args.start is not None or args.end is not None:
        print(f"📏 Data slice: [{args.start or 0}:{args.end or 'end'}]")
    if args.train_size is not None:
        print(f"🔀 Training set size: {args.train_size}")
    print("=" * 80)
    
    result = quick_analyze(
        file_path=file_path,
        mag_columns=mag_columns,
        num_sectors=args.sectors,
        min_radius=args.min_radius,
        show_details=not args.quiet,
        start_row=args.start,
        end_row=args.end,
        train_size=args.train_size
    )
    
    if result:
        print("\n" + "=" * 80)
        print("✅ Analysis completed successfully!")
        if result.get('test_results') is not None:
            print("📊 Both training and test sets analyzed")
        else:
            print("📊 Single dataset analyzed")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("❌ Analysis failed!")
        print("=" * 80)

if __name__ == "__main__":
    main()