"""
Final Figures Generator

This script generates the final figures for the bPAC analysis.
The first figure is an example figure showing:
1. Average image with ROI zoom-in
2. GCaMP6s z-scored trace
3. PinkFlamindo z-scored trace
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from statsmodels.nonparametric.smoothers_lowess import lowess

def load_pickle_data(pickle_path):
    """
    Load the ROI data from the pickle file.
    
    Args:
        pickle_path (str): Path to the pickle file
        
    Returns:
        pandas.DataFrame: DataFrame containing the ROI data
    """
    return pd.read_pickle(pickle_path)

def create_average_image(stack):
    """
    Create an average image from the stack by averaging along the z-axis.
    
    Args:
        stack (numpy.ndarray): The image stack
        
    Returns:
        numpy.ndarray: The average image
    """
    return np.mean(stack, axis=0)

def enhance_contrast(image):
    """
    Enhance the contrast of the image using percentile-based scaling.
    
    Args:
        image (numpy.ndarray): The input image
        
    Returns:
        tuple: (enhanced_image, vmin, vmax) - Enhanced image and its value range
    """
    p05, p995 = np.percentile(image, (0.5, 99.5))  # Using 0.5th to 99.5th percentile for gentler enhancement
    return image, p05, p995

def smooth_trace(trace, is_gcamp=False):
    """
    Smooth the trace using LOWESS (Locally Weighted Scatterplot Smoothing), ignoring frames 15-18.
    
    Args:
        trace (numpy.ndarray): The original trace
        is_gcamp (bool): Whether this is a GCaMP6s trace (uses stronger smoothing)
        
    Returns:
        numpy.ndarray: The smoothed trace
    """
    # Create a copy of the trace
    smoothed = trace.copy()
    
    # Store the values for frames 15-18
    frames_to_ignore = slice(15, 19)  # 15-18 inclusive
    ignored_values = trace[frames_to_ignore].copy()
    
    # Set frames 15-18 to NaN for smoothing
    smoothed[frames_to_ignore] = np.nan
    
    # Create a mask for valid values
    valid_mask = ~np.isnan(smoothed)
    x = np.arange(len(trace))
    
    # Apply LOWESS smoothing with different parameters for GCaMP and PinkFlamindo
    if is_gcamp:
        frac = 0.3  # Stronger smoothing for GCaMP (30% of data used for each estimation)
    else:
        frac = 0.2  # Gentler smoothing for PinkFlamindo (20% of data used for each estimation)
    
    # Apply LOWESS only to valid points
    x_valid = x[valid_mask]
    y_valid = smoothed[valid_mask]
    
    # Perform LOWESS smoothing
    smoothed_valid = lowess(
        y_valid,
        x_valid,
        frac=frac,          # The fraction of data used for smoothing
        it=1,              # Number of robustifying iterations
        delta=0.1,         # Distance within which to use linear-interpolation instead of weighted regression
        return_sorted=False # Return values at input x points
    )
    
    # Create output array
    result = smoothed.copy()
    result[valid_mask] = smoothed_valid
    
    # Restore the original values for frames 15-18
    result[frames_to_ignore] = ignored_values
    
    return result

def generate_example_figure(df, parent_directory, output_path):
    """
    Generate the example figure showing ROI and traces.
    
    Args:
        df (pandas.DataFrame): DataFrame containing ROI data
        parent_directory (str): Path to the parent directory
        output_path (str): Path to save the figure
    """
    # Find the specific entry
    entry = df[
        (df['MOUSE'] == 'MLV') & 
        (df['EXP'] == 'LEDx15pls_MV_protocol') & 
        (df['ROI#'] == 4)
    ].iloc[0]
    
    # Load and process the stack
    stks_dir = os.path.join(parent_directory, entry['MOUSE'], entry['EXP'], 'STKS')
    tif_path = os.path.join(stks_dir, 'ChanB_stk.tif')
    stack = imread(tif_path)
    avg_image = create_average_image(stack)
    enhanced_image, vmin, vmax = enhance_contrast(avg_image)
    
    # Load ROI coordinates
    roi_path = os.path.join(stks_dir, 'ROIs', f"ROI#{entry['ROI#']}.npy")
    roi_coords = np.load(roi_path)
    
    # Calculate ROI dimensions and center
    center_x = np.mean(roi_coords[:, 0])
    center_y = np.mean(roi_coords[:, 1])
    width = np.max(roi_coords[:, 0]) - np.min(roi_coords[:, 0])
    height = np.max(roi_coords[:, 1]) - np.min(roi_coords[:, 1])
    max_dim = max(width, height) * 1.2  # 20% larger
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1], hspace=0.05)
    
    # 1. Average image with zoom-in
    ax1 = fig.add_subplot(gs[:, 0])
    im = ax1.imshow(enhanced_image, cmap='viridis', vmin=vmin, vmax=vmax)
    
    # Add zoom-in box
    y_min = max(0, int(center_y - max_dim/2))
    y_max = min(avg_image.shape[0], int(center_y + max_dim/2))
    x_min = max(0, int(center_x - max_dim/2))
    x_max = min(avg_image.shape[1], int(center_x + max_dim/2))
    
    # Draw dotted rectangle for zoom area
    rect = plt.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min,
                        fill=False, edgecolor='red', linestyle=':', linewidth=1)
    ax1.add_patch(rect)
    
    # Add zoom-in insert
    ax_inset = ax1.inset_axes([0.6, 0.1, 0.35, 0.35])
    ax_inset.imshow(enhanced_image[y_min:y_max, x_min:x_max], cmap='viridis', vmin=vmin, vmax=vmax)
    ax_inset.set_xticks([])
    ax_inset.set_yticks([])
    ax_inset.spines['bottom'].set_color('black')
    ax_inset.spines['top'].set_color('black')
    ax_inset.spines['left'].set_color('black')
    ax_inset.spines['right'].set_color('black')
    
    ax1.set_title('Example of ROI')
    ax1.axis('off')
    
    # 2. GCaMP6s trace (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    x = np.arange(len(entry['ChanB_Z_trc']))
    y_smooth = smooth_trace(entry['ChanB_Z_trc'], is_gcamp=True)  # Use stronger smoothing for GCaMP
    
    # Add baseline at y=0
    ax2.plot([0, len(entry['ChanB_Z_trc'])], [0, 0], 'k--', linewidth=0.5)
    
    # Add vertical cyan line from frame 15 to 16
    ax2.fill_betweenx([0, 35], 15, 16, color='cyan', alpha=0.3)
    
    ax2.scatter(x, entry['ChanB_Z_trc'], color='green', s=10, alpha=0.5)
    ax2.plot(x, y_smooth, 'k-', linewidth=1)
    ax2.set_title('GCaMP6s')
    ax2.set_xlabel('Frames')
    ax2.set_ylabel('Z-scored values')
    ax2.set_ylim([-1, 35])  # Set y-axis range
    
    # Remove top and right spines
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # 3. PinkFlamindo trace (bottom right)
    ax3 = fig.add_subplot(gs[1, 1])
    x = np.arange(len(entry['ChanA_Z_trc']))
    y_smooth = smooth_trace(entry['ChanA_Z_trc'], is_gcamp=False)  # Use original smoothing for PinkFlamindo
    
    # Add baseline at y=0
    ax3.plot([0, len(entry['ChanA_Z_trc'])], [0, 0], 'k--', linewidth=0.5)
    
    # Add vertical cyan line from frame 15 to 16
    ax3.fill_betweenx([0, 4], 15, 16, color='cyan', alpha=0.3)
    
    ax3.scatter(x, entry['ChanA_Z_trc'], color='red', s=10, alpha=0.5)
    ax3.plot(x, y_smooth, 'k-', linewidth=1)
    ax3.set_title('PinkFlamindo')
    ax3.set_xlabel('Frames')
    ax3.set_ylabel('Z-scored values')
    ax3.set_ylim([-1, 4])  # Set y-axis range
    
    # Remove top and right spines
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # Adjust subplot positions to align with image edges and prevent overlap
    pos1 = ax1.get_position()
    total_height = pos1.height
    subplot_height = total_height * 0.4  # Make each subplot 40% of the total height
    
    # Set new positions
    ax2.set_position([pos1.x1 + 0.05, pos1.y1 - subplot_height, pos1.width, subplot_height])
    ax3.set_position([pos1.x1 + 0.05, pos1.y0, pos1.width, subplot_height])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main(parent_directory):
    """
    Main function to generate the final figures.
    
    Args:
        parent_directory (str): Path to the parent directory
    """
    # Load the pickle file
    pickle_path = os.path.join(parent_directory, 'ROI_quant_overview.pkl')
    df = load_pickle_data(pickle_path)
    
    # Generate the example figure
    output_path = os.path.join(parent_directory, 'example_figure.png')
    generate_example_figure(df, parent_directory, output_path)
    print(f"\nExample figure generated: {output_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python final_figs.py <parent_directory>")
        sys.exit(1)
    
    parent_directory = sys.argv[1]
    main(parent_directory) 