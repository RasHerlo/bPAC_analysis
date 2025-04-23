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
    # Find the four specific entries
    entries = df[
        ((df['MOUSE'] == 'MLV') & 
         (df['EXP'] == 'LEDx15pls_MV_protocol') & 
         (df['ROI#'].isin([4, 5]))) |
        ((df['MOUSE'] == 'MLV') & 
         (df['EXP'] == 'LEDx15pls_MV_protocol_004') & 
         (df['ROI#'].isin([4, 5])))
    ]
    
    # Calculate average traces
    gcamp_traces = np.array([entry['ChanB_Z_trc'] for _, entry in entries.iterrows()])
    pinkflamindo_traces = np.array([entry['ChanA_Z_trc'] for _, entry in entries.iterrows()])
    
    avg_gcamp = np.mean(gcamp_traces, axis=0)
    avg_pinkflamindo = np.mean(pinkflamindo_traces, axis=0)
    sem_gcamp = np.std(gcamp_traces, axis=0) / np.sqrt(len(gcamp_traces))
    sem_pinkflamindo = np.std(pinkflamindo_traces, axis=0) / np.sqrt(len(pinkflamindo_traces))
    
    # Get the first entry for ROI coordinates and image
    first_entry = entries.iloc[0]
    
    # Load and process the stack
    stks_dir = os.path.join(parent_directory, first_entry['MOUSE'], first_entry['EXP'], 'STKS')
    tif_path = os.path.join(stks_dir, 'ChanB_stk.tif')
    stack = imread(tif_path)
    avg_image = create_average_image(stack)
    enhanced_image, vmin, vmax = enhance_contrast(avg_image)
    
    # Load ROI coordinates for ROI#4
    roi_path = os.path.join(stks_dir, 'ROIs', "ROI#4.npy")
    roi_coords = np.load(roi_path)
    
    # Calculate ROI center
    center_x = np.mean(roi_coords[:, 0])
    center_y = np.mean(roi_coords[:, 1])
    
    # Find the shortest distance to any edge
    distances_to_edges = [
        center_y,  # distance to top
        avg_image.shape[0] - center_y,  # distance to bottom
        center_x,  # distance to left
        avg_image.shape[1] - center_x  # distance to right
    ]
    max_dim = 2 * min(distances_to_edges)  # Total size of the square
    
    # Calculate square boundaries
    y_min = max(0, int(center_y - max_dim/2))
    y_max = min(avg_image.shape[0], int(center_y + max_dim/2))
    x_min = max(0, int(center_x - max_dim/2))
    x_max = min(avg_image.shape[1], int(center_x + max_dim/2))
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1], hspace=0.05)
    
    # 1. Average image (cropped square)
    ax1 = fig.add_subplot(gs[:, 0])
    cropped_image = enhanced_image[y_min:y_max, x_min:x_max]
    im = ax1.imshow(cropped_image, cmap='viridis', vmin=vmin, vmax=vmax)
    ax1.set_title('Average Image')
    ax1.axis('off')
    
    # 2. GCaMP6s trace (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    frames = np.arange(len(avg_gcamp))
    time = frames * 2.2 - 33  # Convert frames to seconds and shift so frame 15 is at 0
    
    # Add baseline at y=0
    ax2.plot([time[0], time[-1]], [0, 0], 'k--', linewidth=0.5)
    
    # Add vertical cyan line from frame 15 to 16
    stim_start = 0  # Now at 0 seconds
    stim_end = 2.2  # 1 frame later
    ax2.fill_betweenx([0, 35], stim_start, stim_end, color='cyan', alpha=0.3)
    
    # Plot average trace with error bands
    ax2.plot(time, avg_gcamp, 'k-', linewidth=1)
    ax2.fill_between(time, avg_gcamp - sem_gcamp, avg_gcamp + sem_gcamp, color='green', alpha=0.2)
    
    ax2.set_title('GCaMP6s')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Z-scored values')
    ax2.set_ylim([-1, 35])
    ax2.set_xlim([-30, 150])
    
    # Add ticks at specified time points
    ax2.set_xticks([-30, 0, 30, 60, 90, 120, 150])
    ax2.set_xticklabels(['-30', '0', '30', '60', '90', '120', '150'])
    
    # Remove top and right spines
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # 3. PinkFlamindo trace (bottom right)
    ax3 = fig.add_subplot(gs[1, 1])
    
    # Add baseline at y=0
    ax3.plot([time[0], time[-1]], [0, 0], 'k--', linewidth=0.5)
    
    # Add vertical cyan line from frame 15 to 16
    ax3.fill_betweenx([0, 4], stim_start, stim_end, color='cyan', alpha=0.3)
    
    # Plot average trace with error bands
    ax3.plot(time, avg_pinkflamindo, 'k-', linewidth=1)
    ax3.fill_between(time, avg_pinkflamindo - sem_pinkflamindo, avg_pinkflamindo + sem_pinkflamindo, color='red', alpha=0.2)
    
    ax3.set_title('PinkFlamindo')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Z-scored values')
    ax3.set_ylim([-1, 4])
    ax3.set_xlim([-30, 150])
    
    # Add ticks at specified time points
    ax3.set_xticks([-30, 0, 30, 60, 90, 120, 150])
    ax3.set_xticklabels(['-30', '0', '30', '60', '90', '120', '150'])
    
    # Remove top and right spines
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def generate_mean_traces_figure(df, output_path):
    """
    Generate the mean traces figure with bar plots for specific time points.
    
    Args:
        df (pandas.DataFrame): DataFrame containing ROI data
        output_path (str): Path to save the figure
    """
    # Group by MOUSE and EXP to get mean traces
    grouped = df.groupby(['MOUSE', 'EXP'])
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[2, 1], height_ratios=[1, 1], hspace=0.05)
    
    # Time points for bar plots
    time_points = [-10, 8, 40, 80]  # seconds
    frame_points = [(t + 33) / 2.2 for t in time_points]  # Convert to frame numbers
    
    # Calculate mean and SEM for each group
    gcamp_means = []
    gcamp_sems = []
    pinkflamindo_means = []
    pinkflamindo_sems = []
    
    for (mouse, exp), group in grouped:
        gcamp_traces = np.array([entry['ChanB_Z_trc'] for _, entry in group.iterrows()])
        pinkflamindo_traces = np.array([entry['ChanA_Z_trc'] for _, entry in group.iterrows()])
        
        gcamp_mean = np.mean(gcamp_traces, axis=0)
        pinkflamindo_mean = np.mean(pinkflamindo_traces, axis=0)
        
        gcamp_sem = np.std(gcamp_traces, axis=0) / np.sqrt(len(gcamp_traces))
        pinkflamindo_sem = np.std(pinkflamindo_traces, axis=0) / np.sqrt(len(pinkflamindo_traces))
        
        gcamp_means.append(gcamp_mean)
        gcamp_sems.append(gcamp_sem)
        pinkflamindo_means.append(pinkflamindo_mean)
        pinkflamindo_sems.append(pinkflamindo_sem)
    
    # Calculate overall mean and SEM
    gcamp_overall_mean = np.mean(gcamp_means, axis=0)
    gcamp_overall_sem = np.sqrt(np.sum(np.array(gcamp_sems)**2, axis=0)) / len(gcamp_sems)
    
    pinkflamindo_overall_mean = np.mean(pinkflamindo_means, axis=0)
    pinkflamindo_overall_sem = np.sqrt(np.sum(np.array(pinkflamindo_sems)**2, axis=0)) / len(pinkflamindo_sems)
    
    # Convert frame numbers to time
    frames = np.arange(len(gcamp_overall_mean))
    time = frames * 2.2 - 33
    
    # 1. GCaMP6s trace (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(time, gcamp_overall_mean, 'k-', linewidth=1)
    ax1.fill_between(time, gcamp_overall_mean - gcamp_overall_sem, gcamp_overall_mean + gcamp_overall_sem, color='green', alpha=0.2)
    
    # Add baseline at y=0
    ax1.plot([time[0], time[-1]], [0, 0], 'k--', linewidth=0.5)
    
    # Add vertical cyan line for stimulation
    stim_start = 0
    stim_end = 2.2
    ax1.fill_betweenx([0, 35], stim_start, stim_end, color='cyan', alpha=0.3)
    
    ax1.set_title('GCaMP6s')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Z-scored values')
    ax1.set_ylim([-1, 35])
    ax1.set_xlim([-30, 150])
    
    # Add ticks at specified time points
    ax1.set_xticks([-30, 0, 30, 60, 90, 120, 150])
    ax1.set_xticklabels(['-30', '0', '30', '60', '90', '120', '150'])
    
    # Remove top and right spines
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # 2. GCaMP6s bar plot (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    bar_width = 0.6
    x_pos = np.arange(len(time_points))
    
    # Get values at specified time points
    gcamp_values = [gcamp_overall_mean[int(f)] for f in frame_points]
    gcamp_errors = [gcamp_overall_sem[int(f)] for f in frame_points]
    
    bars = ax2.bar(x_pos, gcamp_values, yerr=gcamp_errors, width=bar_width, color='green', alpha=0.7)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([str(t) for t in time_points])
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Z-scored values')
    ax2.set_ylim([-1, 35])
    
    # Remove top and right spines
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # 3. PinkFlamindo trace (bottom left)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(time, pinkflamindo_overall_mean, 'k-', linewidth=1)
    ax3.fill_between(time, pinkflamindo_overall_mean - pinkflamindo_overall_sem, pinkflamindo_overall_mean + pinkflamindo_overall_sem, color='red', alpha=0.2)
    
    # Add baseline at y=0
    ax3.plot([time[0], time[-1]], [0, 0], 'k--', linewidth=0.5)
    
    # Add vertical cyan line for stimulation
    ax3.fill_betweenx([0, 4], stim_start, stim_end, color='cyan', alpha=0.3)
    
    ax3.set_title('PinkFlamindo')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Z-scored values')
    ax3.set_ylim([-1, 4])
    ax3.set_xlim([-30, 150])
    
    # Add ticks at specified time points
    ax3.set_xticks([-30, 0, 30, 60, 90, 120, 150])
    ax3.set_xticklabels(['-30', '0', '30', '60', '90', '120', '150'])
    
    # Remove top and right spines
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # 4. PinkFlamindo bar plot (bottom right)
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Get values at specified time points
    pinkflamindo_values = [pinkflamindo_overall_mean[int(f)] for f in frame_points]
    pinkflamindo_errors = [pinkflamindo_overall_sem[int(f)] for f in frame_points]
    
    bars = ax4.bar(x_pos, pinkflamindo_values, yerr=pinkflamindo_errors, width=bar_width, color='red', alpha=0.7)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([str(t) for t in time_points])
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Z-scored values')
    ax4.set_ylim([-1, 4])
    
    # Remove top and right spines
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    
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
    example_output_path = os.path.join(parent_directory, 'example_figure.png')
    generate_example_figure(df, parent_directory, example_output_path)
    print(f"\nExample figure generated: {example_output_path}")
    
    # Generate the mean traces figure
    mean_traces_output_path = os.path.join(parent_directory, 'mean_traces_figure.png')
    generate_mean_traces_figure(df, mean_traces_output_path)
    print(f"Mean traces figure generated: {mean_traces_output_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python final_figs.py <parent_directory>")
        sys.exit(1)
    
    parent_directory = sys.argv[1]
    main(parent_directory) 