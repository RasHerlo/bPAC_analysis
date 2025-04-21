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

def smooth_trace(trace):
    """
    Smooth the trace using a Savitzky-Golay filter, ignoring frames 15-18.
    
    Args:
        trace (numpy.ndarray): The original trace
        
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
    
    # Find valid indices (not NaN)
    valid_indices = ~np.isnan(smoothed)
    
    # Create x values for valid points
    x = np.arange(len(trace))[valid_indices]
    y = smoothed[valid_indices]
    
    # Apply Savitzky-Golay filter to valid points
    window_length = min(15, len(y) - 1 if len(y) % 2 == 0 else len(y) - 2)
    poly_order = 3
    
    # Smooth the valid points
    smoothed[valid_indices] = savgol_filter(y, window_length, poly_order)
    
    # Restore the original values for frames 15-18
    smoothed[frames_to_ignore] = ignored_values
    
    return smoothed

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
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
    
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
    y_smooth = smooth_trace(entry['ChanB_Z_trc'])
    
    # Add baseline at y=0
    ax2.plot([0, len(entry['ChanB_Z_trc'])], [0, 0], 'k--', linewidth=0.5)
    
    ax2.scatter(x, entry['ChanB_Z_trc'], color='green', s=10, alpha=0.5)
    ax2.plot(x, y_smooth, 'k-', linewidth=1)
    ax2.set_title('GCaMP6s')
    ax2.set_xlabel('Frames')
    ax2.set_ylabel('Z-scored values')
    ax2.set_ylim([-1, None])  # Set lower limit to -1
    
    # 3. PinkFlamindo trace (bottom right)
    ax3 = fig.add_subplot(gs[1, 1])
    x = np.arange(len(entry['ChanA_Z_trc']))
    y_smooth = smooth_trace(entry['ChanA_Z_trc'])
    
    # Add baseline at y=0
    ax3.plot([0, len(entry['ChanA_Z_trc'])], [0, 0], 'k--', linewidth=0.5)
    
    ax3.scatter(x, entry['ChanA_Z_trc'], color='red', s=10, alpha=0.5)
    ax3.plot(x, y_smooth, 'k-', linewidth=1)
    ax3.set_title('PinkFlamindo')
    ax3.set_xlabel('Frames')
    ax3.set_ylabel('Z-scored values')
    ax3.set_ylim([-1, None])  # Set lower limit to -1
    
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