"""
Final Figures Generator

This script generates the final figures for the bPAC analysis.
The first figure is an example figure showing:
1. Average image with ROI zoom-in
2. GCaMP6s dF/F trace
3. PinkFlamindo dF/F trace
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread, imwrite
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

def generate_example_figure(df, parent_directory, output_path, results_dir):
    """
    Generate the example figure showing ROI and traces.
    
    Args:
        df (pandas.DataFrame): DataFrame containing ROI data
        parent_directory (str): Path to the parent directory
        output_path (str): Path to save the figure
        results_dir (str): Path to the results directory
        
    Returns:
        dict: Dictionary containing the data for export
    """
    # Find the four specific entries to average
    entries = df[
        ((df['MOUSE'] == 'MLV') & 
         (df['EXP'] == 'LEDx15pls_MV_protocol') & 
         (df['ROI#'].isin([4, 5]))) |
        ((df['MOUSE'] == 'MLV') & 
         (df['EXP'] == 'LEDx15pls_MV_protocol_004') & 
         (df['ROI#'].isin([4, 5])))
    ]
    
    # Use the first entry for the image and ROI coordinates
    entry = entries.iloc[0]
    
    # Load and process the stack
    stks_dir = os.path.join(parent_directory, entry['MOUSE'], entry['EXP'], 'STKS')
    tif_path = os.path.join(stks_dir, 'ChanB_stk.tif')
    stack = imread(tif_path)
    avg_image = create_average_image(stack)
    enhanced_image, vmin, vmax = enhance_contrast(avg_image)
    
    # Take the top-right 200x150 pixels
    height, width = enhanced_image.shape
    top_right_corner = enhanced_image[0:150, width-200:width]
    
    # Save the image cutout
    imwrite(os.path.join(results_dir, 'example_roi.tif'), top_right_corner.astype(np.float32))
    
    # Load ROI coordinates
    roi_path = os.path.join(stks_dir, 'ROIs', f"ROI#{entry['ROI#']}.npy")
    roi_coords = np.load(roi_path)
    
    # Calculate ROI dimensions and center
    center_x = np.mean(roi_coords[:, 0])
    center_y = np.mean(roi_coords[:, 1])
    roi_width = np.max(roi_coords[:, 0]) - np.min(roi_coords[:, 0])
    roi_height = np.max(roi_coords[:, 1]) - np.min(roi_coords[:, 1])
    max_dim = max(roi_width, roi_height) * 1.2  # 20% larger
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1], hspace=0.05)
    
    # 1. Average image with zoom-in
    ax1 = fig.add_subplot(gs[:, 0])
    im = ax1.imshow(top_right_corner, cmap='viridis', vmin=vmin, vmax=vmax)
    
    # Calculate ROI position in the new image section
    roi_x_min = np.min(roi_coords[:, 0]) - (width-200)
    roi_y_min = np.min(roi_coords[:, 1])
    
    # Draw dotted rectangle for ROI
    rect = plt.Rectangle((roi_x_min, roi_y_min), roi_width, roi_height,
                        fill=False, edgecolor='red', linestyle=':', linewidth=1)
    ax1.add_patch(rect)
    
    ax1.set_title('Example of ROI')
    ax1.axis('off')
    
    # 2. GCaMP6s trace (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    frames = np.arange(len(entry['ChanB_dFF']))
    time = frames * 2.2 - 33  # Convert frames to seconds and shift so frame 15 is at 0
    
    # Get all GCaMP6s traces and set frames 15-17 to NaN
    gcamp_traces = []
    for _, row in entries.iterrows():
        trace = row['ChanB_dFF'].copy()
        trace[15:18] = np.nan
        gcamp_traces.append(trace)
    
    # Calculate average trace
    gcamp_traces = np.array(gcamp_traces)
    gcamp_avg = np.nanmean(gcamp_traces, axis=0)
    
    y_smooth = smooth_trace(gcamp_avg, is_gcamp=True)  # Use stronger smoothing for GCaMP
    
    # Add baseline at y=1
    ax2.plot([time[0], time[-1]], [1, 1], 'k--', linewidth=0.5)
    
    # Add vertical cyan line from frame 15 to 16
    stim_start = 0  # Now at 0 seconds
    stim_end = 2.2  # 1 frame later
    gcamp_max = np.nanmax(gcamp_avg)  # Get maximum value for cyan box height
    ax2.fill_betweenx([1, gcamp_max], stim_start, stim_end, color='cyan', alpha=0.3)
    
    ax2.scatter(time, gcamp_avg, color='green', s=10, alpha=0.5)
    ax2.plot(time, y_smooth, 'k-', linewidth=1)
    ax2.set_title('GCaMP6s')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('dF/F values')
    ax2.set_xlim([-30, 150])  # Set x-axis limit
    
    # Add ticks at specified time points
    ax2.set_xticks([-30, 0, 30, 60, 90, 120, 150])
    ax2.set_xticklabels(['-30', '0', '30', '60', '90', '120', '150'])
    
    # Remove top and right spines
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # 3. PinkFlamindo trace (bottom right)
    ax3 = fig.add_subplot(gs[1, 1])
    
    # Get all PinkFlamindo traces and set frames 15-17 to NaN
    pinkflamindo_traces = []
    for _, row in entries.iterrows():
        trace = row['ChanA_dFF'].copy()
        trace[15:18] = np.nan
        pinkflamindo_traces.append(trace)
    
    # Calculate average trace
    pinkflamindo_traces = np.array(pinkflamindo_traces)
    pinkflamindo_avg = np.nanmean(pinkflamindo_traces, axis=0)
    
    y_smooth = smooth_trace(pinkflamindo_avg, is_gcamp=False)  # Use original smoothing for PinkFlamindo
    
    # Add baseline at y=1
    ax3.plot([time[0], time[-1]], [1, 1], 'k--', linewidth=0.5)
    
    # Add vertical cyan line from frame 15 to 16
    pinkflamindo_max = np.nanmax(pinkflamindo_avg)  # Get maximum value for cyan box height
    ax3.fill_betweenx([1, pinkflamindo_max], stim_start, stim_end, color='cyan', alpha=0.3)
    
    ax3.scatter(time, pinkflamindo_avg, color='red', s=10, alpha=0.5)
    ax3.plot(time, y_smooth, 'k-', linewidth=1)
    ax3.set_title('PinkFlamindo')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('dF/F values')
    ax3.set_xlim([-30, 150])  # Set x-axis limit
    
    # Add ticks at specified time points
    ax3.set_xticks([-30, 0, 30, 60, 90, 120, 150])
    ax3.set_xticklabels(['-30', '0', '30', '60', '90', '120', '150'])
    
    # Remove top and right spines
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Return the data for export
    return {
        'time': time,
        'gcamp': gcamp_avg,
        'pinkflamindo': pinkflamindo_avg
    }

def generate_mean_traces_figure(df, output_path):
    """
    Generate the mean traces figure showing average GCaMP6s and PinkFlamindo traces.
    
    Args:
        df (pandas.DataFrame): DataFrame containing ROI data
        output_path (str): Path to save the figure
        
    Returns:
        dict: Dictionary containing the data for export
    """
    # Filter for specific mouse and experiment
    filtered_df = df[
        (df['MOUSE'] == 'MLV') & 
        (df['EXP'].str.contains('LEDx15pls_MV_protocol'))
    ]
    
    # Get all GCaMP6s traces
    gcamp_traces = []
    for _, row in filtered_df.iterrows():
        trace = row['ChanB_dFF'].copy()
        trace[15:18] = np.nan  # Set frames 15-17 to NaN
        gcamp_traces.append(trace)
    
    # Get all PinkFlamindo traces
    pinkflamindo_traces = []
    for _, row in filtered_df.iterrows():
        trace = row['ChanA_dFF'].copy()
        trace[15:18] = np.nan  # Set frames 15-17 to NaN
        pinkflamindo_traces.append(trace)
    
    # Convert to numpy arrays
    gcamp_traces = np.array(gcamp_traces)
    pinkflamindo_traces = np.array(pinkflamindo_traces)
    
    # Calculate mean and SEM
    gcamp_mean = np.nanmean(gcamp_traces, axis=0)
    gcamp_sem = np.nanstd(gcamp_traces, axis=0) / np.sqrt(np.sum(~np.isnan(gcamp_traces), axis=0))
    
    pinkflamindo_mean = np.nanmean(pinkflamindo_traces, axis=0)
    pinkflamindo_sem = np.nanstd(pinkflamindo_traces, axis=0) / np.sqrt(np.sum(~np.isnan(pinkflamindo_traces), axis=0))
    
    # Create time array
    frames = np.arange(len(gcamp_mean))
    time = frames * 2.2 - 33  # Convert frames to seconds and shift so frame 15 is at 0
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    
    # Plot GCaMP6s
    ax1.plot(time, gcamp_mean, 'g-', linewidth=1)
    ax1.fill_between(time, gcamp_mean - gcamp_sem, gcamp_mean + gcamp_sem, color='g', alpha=0.2)
    ax1.set_title('GCaMP6s')
    ax1.set_ylabel('dF/F values')
    ax1.set_xlim([-30, 150])
    ax1.set_xticks([-30, 0, 30, 60, 90, 120, 150])
    ax1.set_xticklabels(['-30', '0', '30', '60', '90', '120', '150'])
    
    # Plot PinkFlamindo
    ax2.plot(time, pinkflamindo_mean, 'r-', linewidth=1)
    ax2.fill_between(time, pinkflamindo_mean - pinkflamindo_sem, pinkflamindo_mean + pinkflamindo_sem, color='r', alpha=0.2)
    ax2.set_title('PinkFlamindo')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('dF/F values')
    
    # Add vertical cyan line from frame 15 to 16
    stim_start = 0  # Now at 0 seconds
    stim_end = 2.2  # 1 frame later
    
    # Add cyan box to both plots
    for ax in [ax1, ax2]:
        y_min, y_max = ax.get_ylim()
        ax.fill_betweenx([y_min, y_max], stim_start, stim_end, color='cyan', alpha=0.3)
        ax.set_ylim([y_min, y_max])
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Return the data for export
    return {
        'time': time,
        'gcamp_mean': gcamp_mean,
        'gcamp_sem': gcamp_sem,
        'pinkflamindo_mean': pinkflamindo_mean,
        'pinkflamindo_sem': pinkflamindo_sem
    }

def generate_temporal_overlay_figure(mean_traces_data, output_path):
    """
    Generate a figure showing normalized mean traces for GCaMP6s and PinkFlamindo.
    
    Args:
        mean_traces_data (dict): Dictionary containing mean traces data
        output_path (str): Path to save the figure
        
    Returns:
        dict: Dictionary containing the normalized traces data
    """
    # Get the time points and mean traces
    time = mean_traces_data['time']
    gcamp_mean = mean_traces_data['gcamp_mean']
    pinkflamindo_mean = mean_traces_data['pinkflamindo_mean']
    
    # Normalize GCaMP6s
    gcamp_norm = (gcamp_mean - 1) / (np.nanmax(gcamp_mean) - 1)
    
    # Smooth the traces using LOWESS
    gcamp_smooth = smooth_trace(gcamp_norm, is_gcamp=True)
    pinkflamindo_smooth = smooth_trace(pinkflamindo_mean, is_gcamp=False)
    
    # Normalize PinkFlamindo based on smoothed curve maximum
    pinkflamindo_norm = (pinkflamindo_mean - 1) / (np.nanmax(pinkflamindo_smooth) - 1)
    pinkflamindo_smooth = (pinkflamindo_smooth - 1) / (np.nanmax(pinkflamindo_smooth) - 1)
    
    # Create the figure
    plt.figure(figsize=(8, 6))
    
    # Plot the scatter points
    plt.scatter(time, gcamp_norm, color='green', s=10, alpha=0.5, label='GCaMP6s')
    plt.scatter(time, pinkflamindo_norm, color='red', s=10, alpha=0.5, label='PinkFlamindo')
    
    # Plot the smoothed curves
    plt.plot(time, gcamp_smooth, 'k-', linewidth=1)
    plt.plot(time, pinkflamindo_smooth, 'k-', linewidth=1)
    
    # Add the cyan box
    plt.fill_betweenx([0, 1], 0, 2.2, color='cyan', alpha=0.3)
    
    # Set plot properties
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized dF/F')
    plt.title('Normalized Temporal Overlay')
    plt.legend()
    plt.grid(True)
    
    # Set x-axis limits and ticks
    plt.xlim([-30, 150])
    plt.xticks([-30, 0, 30, 60, 90, 120, 150])
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Return the normalized traces data including smoothed values
    return {
        'time': time,
        'gcamp_norm': gcamp_norm,
        'pinkflamindo_norm': pinkflamindo_norm,
        'gcamp_smooth': gcamp_smooth,
        'pinkflamindo_smooth': pinkflamindo_smooth
    }

def create_results_directory(pickle_dir):
    """
    Create a results directory in the same location as the pickle file.
    
    Args:
        pickle_dir (str): Directory containing the pickle file
        
    Returns:
        str: Path to the results directory
    """
    results_dir = os.path.join(pickle_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    return results_dir

def export_data_to_excel(example_data, mean_traces_data, norm_traces_data, output_path):
    """
    Export the data to an Excel file with three sheets.
    
    Args:
        example_data (dict): Dictionary containing example figure data
        mean_traces_data (dict): Dictionary containing mean traces data
        norm_traces_data (dict): Dictionary containing normalized traces data
        output_path (str): Path to save the Excel file
    """
    with pd.ExcelWriter(output_path) as writer:
        # Example figure data
        example_df = pd.DataFrame({
            'Time (s)': example_data['time'],
            'GCaMP6s': example_data['gcamp'],
            'PinkFlamindo': example_data['pinkflamindo']
        })
        example_df.to_excel(writer, sheet_name='example_figure', index=False)
        
        # Mean traces data
        mean_traces_df = pd.DataFrame({
            'Time (s)': mean_traces_data['time'],
            'Mean_GCaMP6s': mean_traces_data['gcamp_mean'],
            'SEM_GCaMP6s': mean_traces_data['gcamp_sem'],
            'Mean_PinkFlamindo': mean_traces_data['pinkflamindo_mean'],
            'SEM_PinkFlamindo': mean_traces_data['pinkflamindo_sem']
        })
        mean_traces_df.to_excel(writer, sheet_name='mean_traces_figure', index=False)
        
        # Normalized traces data
        norm_traces_df = pd.DataFrame({
            'Time (s)': norm_traces_data['time'],
            'GCaMP6s_norm': norm_traces_data['gcamp_norm'],
            'PinkFl_norm': norm_traces_data['pinkflamindo_norm'],
            'GCaMP6s_smooth': norm_traces_data['gcamp_smooth'],
            'PinkFl_smooth': norm_traces_data['pinkflamindo_smooth']
        })
        norm_traces_df.to_excel(writer, sheet_name='Norm_Traces', index=False)

def main(parent_directory):
    """
    Main function to generate all figures and export data.
    
    Args:
        parent_directory (str): Path to the parent directory containing the pickle file
    """
    # Find the pickle file
    pickle_path = None
    for root, _, files in os.walk(parent_directory):
        for file in files:
            if file == 'ROI_quant_overview.pkl':
                pickle_path = os.path.join(root, file)
                break
        if pickle_path:
            break
    
    if not pickle_path:
        print("Error: Could not find ROI_quant_overview.pkl in the specified directory")
        return
    
    # Load the data
    df = load_pickle_data(pickle_path)
    
    # Create results directory
    results_dir = create_results_directory(os.path.dirname(pickle_path))
    
    # Generate the example figure
    example_output_path = os.path.join(results_dir, 'example_figure.png')
    example_data = generate_example_figure(df, parent_directory, example_output_path, results_dir)
    print(f"\nExample figure generated: {example_output_path}")
    
    # Generate the mean traces figure
    mean_traces_output_path = os.path.join(results_dir, 'mean_traces_figure.png')
    mean_traces_data = generate_mean_traces_figure(df, mean_traces_output_path)
    print(f"Mean traces figure generated: {mean_traces_output_path}")
    
    # Generate the temporal overlay figure
    temporal_overlay_path = os.path.join(results_dir, 'temporal_overlay.png')
    norm_traces_data = generate_temporal_overlay_figure(mean_traces_data, temporal_overlay_path)
    print(f"Temporal overlay figure generated: {temporal_overlay_path}")
    
    # Export data to Excel
    output_excel_path = os.path.join(results_dir, 'figures_data.xlsx')
    export_data_to_excel(example_data, mean_traces_data, norm_traces_data, output_excel_path)
    print(f"Data exported to Excel: {output_excel_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python final_figs.py <parent_directory>")
        sys.exit(1)
    
    parent_directory = sys.argv[1]
    main(parent_directory) 