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