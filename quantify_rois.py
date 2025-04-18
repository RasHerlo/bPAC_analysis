import os
import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread
import argparse
import sys
from skimage.draw import polygon
from matplotlib.widgets import Slider

def load_tif_stack(directory, channel):
    """
    Load a tif stack from the given directory and channel.
    
    Parameters:
    -----------
    directory : str
        Directory containing the TIFF stack (STKS folder)
    channel : str
        Channel name ('ChanA' or 'ChanB')
        
    Returns:
    --------
    numpy.ndarray
        The loaded tif stack
    """
    tif_path = os.path.join(directory, f'{channel}_stk.tif')
    print(f"Loading tif stack from: {tif_path}")
    stack = imread(tif_path)
    print(f"Loaded stack with shape: {stack.shape}")
    return stack

def create_average_image(stack):
    """
    Create an average image from the stack by averaging along the z-axis.
    
    Parameters:
    -----------
    stack : numpy.ndarray
        The image stack
        
    Returns:
    --------
    numpy.ndarray
        The average image
    """
    return np.mean(stack, axis=0)

def load_roi(roi_path):
    """
    Load ROI coordinates from a .npy file.
    
    Parameters:
    -----------
    roi_path : str
        Path to the ROI file
        
    Returns:
    --------
    numpy.ndarray
        The ROI coordinates
    """
    return np.load(roi_path)

def calculate_roi_trace(stack, roi_coords):
    """
    Calculate the average intensity trace for pixels within an ROI.
    
    Args:
        stack (numpy.ndarray): The image stack (z, y, x)
        roi_coords (numpy.ndarray): Coordinates of the ROI vertices
        
    Returns:
        numpy.ndarray: Average intensity trace over time
    """
    # Create a mask for the ROI
    mask = np.zeros(stack.shape[1:], dtype=bool)
    rr, cc = polygon(roi_coords[:, 1], roi_coords[:, 0], stack.shape[1:])
    mask[rr, cc] = True
    
    # Calculate average intensity for each frame
    trace = np.mean(stack[:, mask], axis=1)
    return trace

def normalize_trace_excluding_stim(trace, z_stim_start, z_stim_end):
    """
    Normalize trace to 0-1 range using only values outside the stimulation range.
    
    Args:
        trace (numpy.ndarray): The trace to normalize
        z_stim_start (int): Start of stimulation range
        z_stim_end (int): End of stimulation range
        
    Returns:
        numpy.ndarray: Normalized trace
    """
    # Create a mask for non-stimulation frames
    non_stim_mask = np.ones_like(trace, dtype=bool)
    non_stim_mask[z_stim_start:z_stim_end] = False
    
    # Get min and max from non-stimulation frames
    min_val = np.min(trace[non_stim_mask])
    max_val = np.max(trace[non_stim_mask])
    
    # Normalize using these values
    return (trace - min_val) / (max_val - min_val)

def zscore_trace(trace, z_stim_start, z_stim_end):
    """
    Convert a trace to Z-scores using baseline mean and overall std.
    
    Args:
        trace (numpy.ndarray): The trace to Z-score
        z_stim_start (int): Start of stimulation range
        z_stim_end (int): End of stimulation range
        
    Returns:
        numpy.ndarray: Z-scored trace
    """
    # Create a mask for non-stimulation frames
    non_stim_mask = np.ones_like(trace, dtype=bool)
    non_stim_mask[z_stim_start:z_stim_end] = False
    
    # Calculate mean from non-stimulation frames
    baseline_mean = np.mean(trace[non_stim_mask])
    
    # Calculate std from entire trace
    trace_std = np.std(trace)
    
    # Z-score the trace
    return (trace - baseline_mean) / trace_std

def calculate_compensated_trace(trace_chanA, trace_chanB, compensation_percent):
    """
    Calculate compensated trace for channel A based on channel B.
    
    Args:
        trace_chanA (numpy.ndarray): Raw trace from channel A
        trace_chanB (numpy.ndarray): Raw trace from channel B
        compensation_percent (float): Compensation percentage (0-30)
        
    Returns:
        numpy.ndarray: Compensated trace for channel A
    """
    return trace_chanA - (compensation_percent/100 * trace_chanB)

def plot_rois_on_image(avg_image, rois, stackA, stackB, z_stim_start, z_stim_end):
    """
    Plot the average image with ROIs overlaid and create detailed subplots for each ROI.
    
    Args:
        avg_image (numpy.ndarray): The average image to plot
        rois (dict): Dictionary of ROIs with their coordinates
        stackA (numpy.ndarray): The image stack for channel A
        stackB (numpy.ndarray): The image stack for channel B
        z_stim_start (int): Start of stimulation range
        z_stim_end (int): End of stimulation range
    """
    # Calculate figure size based on number of ROIs
    n_rois = len(rois)
    n_rows = max(3, 3 + ((n_rois - 3) + 1) // 2)  # At least 3 rows, add rows for additional ROIs
    n_cols = 6  # 3 for main image + 3 for ROI details
    
    # Create figure with appropriate size
    fig = plt.figure(figsize=(n_cols * 3, n_rows * 3 + 1))  # Extra space for slider
    gs = plt.GridSpec(n_rows + 1, n_cols, figure=fig)  # Extra row for slider
    
    # Plot main image in upper left (3x3)
    ax_main = fig.add_subplot(gs[0:3, 0:3])
    ax_main.imshow(avg_image, cmap='viridis')
    ax_main.set_title('Average Image with ROIs')
    
    # Plot ROIs on main image
    for roi_name, roi_coords in rois.items():
        roi_coords_closed = np.vstack([roi_coords, roi_coords[0]])
        ax_main.plot(roi_coords_closed[:, 0], roi_coords_closed[:, 1], 'r-', linewidth=2)
        center_x = np.mean(roi_coords[:, 0])
        center_y = np.mean(roi_coords[:, 1])
        ax_main.text(center_x, center_y, roi_name, color='red',
                    horizontalalignment='center', verticalalignment='center',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    # Store raw traces and plot objects for updating
    raw_traces = {}
    plot_objects = {}
    
    # Plot each ROI and its trace
    for i, (roi_name, roi_coords) in enumerate(rois.items()):
        # Calculate row and column for this ROI
        if i < 3:
            row = i
            col = 3
        else:
            row = 3 + ((i - 3) // 2)
            col = 0 if (i - 3) % 2 == 0 else 3
        
        # Calculate ROI bounds with 5-pixel margin
        y_min = max(0, int(min(roi_coords[:, 1]) - 5))
        y_max = min(avg_image.shape[0], int(max(roi_coords[:, 1]) + 5))
        x_min = max(0, int(min(roi_coords[:, 0]) - 5))
        x_max = min(avg_image.shape[1], int(max(roi_coords[:, 0]) + 5))
        
        # Plot ROI cut-out
        ax_roi = fig.add_subplot(gs[row, col])
        roi_cutout = avg_image[y_min:y_max, x_min:x_max]
        ax_roi.imshow(roi_cutout, cmap='viridis')
        ax_roi.set_title(f'ROI: {roi_name}')
        ax_roi.set_aspect('equal')
        
        # Add thin red ROI lines to zoom window
        roi_coords_closed = np.vstack([roi_coords, roi_coords[0]])
        roi_coords_relative = roi_coords_closed - np.array([x_min, y_min])
        ax_roi.plot(roi_coords_relative[:, 0], roi_coords_relative[:, 1], 'r-', linewidth=1)
        
        # Calculate and plot traces from both channels
        ax_trace = fig.add_subplot(gs[row, col+1:col+3])
        trace_chanA = calculate_roi_trace(stackA, roi_coords)
        trace_chanB = calculate_roi_trace(stackB, roi_coords)
        
        # Store raw traces
        raw_traces[roi_name] = {'A': trace_chanA, 'B': trace_chanB}
        
        # Normalize traces excluding stimulation range
        norm_trace_chanA = normalize_trace_excluding_stim(trace_chanA, z_stim_start, z_stim_end)
        norm_trace_chanB = normalize_trace_excluding_stim(trace_chanB, z_stim_start, z_stim_end)
        
        # Plot traces and store plot objects
        lineA, = ax_trace.plot(norm_trace_chanA, 'r-', label=f'{roi_name} (ChanA)')
        lineB, = ax_trace.plot(norm_trace_chanB, 'g-', label=f'{roi_name} (ChanB)')
        plot_objects[roi_name] = {'A': lineA, 'B': lineB}
        
        ax_trace.legend()
        ax_trace.set_title(f'Normalized Traces: {roi_name}')
        ax_trace.set_xlabel('Frame')
        ax_trace.set_ylabel('Normalized Intensity')
        
        # Add stimulation range indicator
        stim_rect = plt.Rectangle((z_stim_start, -0.25), 
                                z_stim_end - z_stim_start, 
                                1.5, 
                                facecolor='gray', alpha=0.2)
        ax_trace.add_patch(stim_rect)
        ax_trace.set_ylim(-0.25, 1.25)  # Set y-axis limits for normalized traces
    
    # Add compensation slider
    ax_slider = fig.add_subplot(gs[n_rows, :])
    slider = Slider(
        ax=ax_slider,
        label='Channel A Compensation (%)',
        valmin=0,
        valmax=30,
        valinit=0,
        valstep=1
    )
    
    # Add tick marks and labels
    slider.ax.set_xticks(range(0, 31, 1))  # Tick marks every 1%
    slider.ax.set_xticklabels([str(x) if x % 5 == 0 else '' for x in range(0, 31, 1)])  # Labels every 5%
    
    # Add text label for current value
    text_label = ax_slider.text(0.5, 0.5, 'Compensation: 0%', 
                              transform=ax_slider.transAxes,
                              ha='center', va='center')
    
    def update(val):
        compensation = slider.val
        text_label.set_text(f'Compensation: {compensation:.0f}%')
        
        for roi_name, traces in raw_traces.items():
            # Calculate compensated trace
            compensated_trace = calculate_compensated_trace(traces['A'], traces['B'], compensation)
            
            # Normalize compensated trace
            norm_compensated = normalize_trace_excluding_stim(compensated_trace, z_stim_start, z_stim_end)
            
            # Update plot
            plot_objects[roi_name]['A'].set_ydata(norm_compensated)
        
        fig.canvas.draw_idle()
    
    slider.on_changed(update)
    
    plt.tight_layout()
    plt.show()

def plot_summary_traces(rois, stackA, stackB, z_stim_start, z_stim_end):
    """
    Create a summary figure showing all normalized and Z-scored traces.
    
    Args:
        rois (dict): Dictionary of ROIs with their coordinates
        stackA (numpy.ndarray): The image stack for channel A
        stackB (numpy.ndarray): The image stack for channel B
        z_stim_start (int): Start of stimulation range
        z_stim_end (int): End of stimulation range
    """
    # Create figure with two subplots and extra space for slider
    fig = plt.figure(figsize=(15, 6))
    
    # Create a 2x1 grid for the plots and slider
    gs = plt.GridSpec(2, 1, height_ratios=[5, 1], figure=fig)
    
    # Create a 1x2 grid for the two plots within the top space
    gs_top = gs[0].subgridspec(1, 2)
    
    # Create subplots
    ax1 = fig.add_subplot(gs_top[0])  # First subplot
    ax2 = fig.add_subplot(gs_top[1])  # Second subplot
    
    # Lists to store all traces for averaging
    all_norm_traces_chanA = []
    all_norm_traces_chanB = []
    all_zscore_traces_chanA = []
    all_zscore_traces_chanB = []
    
    # Store raw traces and plot objects
    raw_traces = {}
    plot_objects = {}
    
    # Process each ROI
    for roi_name, roi_coords in rois.items():
        # Calculate traces
        trace_chanA = calculate_roi_trace(stackA, roi_coords)
        trace_chanB = calculate_roi_trace(stackB, roi_coords)
        
        # Store raw traces
        raw_traces[roi_name] = {'A': trace_chanA, 'B': trace_chanB}
        
        # Normalize traces
        norm_trace_chanA = normalize_trace_excluding_stim(trace_chanA, z_stim_start, z_stim_end)
        norm_trace_chanB = normalize_trace_excluding_stim(trace_chanB, z_stim_start, z_stim_end)
        
        # Z-score traces
        zscore_trace_chanA = zscore_trace(trace_chanA, z_stim_start, z_stim_end)
        zscore_trace_chanB = zscore_trace(trace_chanB, z_stim_start, z_stim_end)
        
        # Store traces for averaging
        all_norm_traces_chanA.append(norm_trace_chanA)
        all_norm_traces_chanB.append(norm_trace_chanB)
        all_zscore_traces_chanA.append(zscore_trace_chanA)
        all_zscore_traces_chanB.append(zscore_trace_chanB)
        
        # Plot individual traces
        lineA1, = ax1.plot(norm_trace_chanA, 'r:', alpha=0.3)
        lineB1, = ax1.plot(norm_trace_chanB, 'g:', alpha=0.3)
        lineA2, = ax2.plot(zscore_trace_chanA, 'r:', alpha=0.3)
        lineB2, = ax2.plot(zscore_trace_chanB, 'g:', alpha=0.3)
        
        plot_objects[roi_name] = {
            'norm': {'A': lineA1, 'B': lineB1},
            'zscore': {'A': lineA2, 'B': lineB2}
        }
    
    # Calculate and plot average traces
    avg_norm_chanA = np.mean(all_norm_traces_chanA, axis=0)
    avg_norm_chanB = np.mean(all_norm_traces_chanB, axis=0)
    avg_zscore_chanA = np.mean(all_zscore_traces_chanA, axis=0)
    avg_zscore_chanB = np.mean(all_zscore_traces_chanB, axis=0)
    
    # Plot average traces
    avg_lineA1, = ax1.plot(avg_norm_chanA, 'r-', linewidth=2, label='Avg ChanA')
    avg_lineB1, = ax1.plot(avg_norm_chanB, 'g-', linewidth=2, label='Avg ChanB')
    avg_lineA2, = ax2.plot(avg_zscore_chanA, 'r-', linewidth=2, label='Avg ChanA')
    avg_lineB2, = ax2.plot(avg_zscore_chanB, 'g-', linewidth=2, label='Avg ChanB')
    
    plot_objects['avg'] = {
        'norm': {'A': avg_lineA1, 'B': avg_lineB1},
        'zscore': {'A': avg_lineA2, 'B': avg_lineB2}
    }
    
    # Set titles and labels
    ax1.set_title('normalized traces')
    ax2.set_title('Z-scored traces')
    ax1.set_xlabel('Frame')
    ax2.set_xlabel('Frame')
    ax1.set_ylabel('Normalized Intensity')
    ax2.set_ylabel('Z-score')
    
    # Add legends
    ax1.legend()
    ax2.legend()
    
    # Add compensation slider
    ax_slider = fig.add_subplot(gs[1])
    slider = Slider(
        ax=ax_slider,
        label='Channel A Compensation (%)',
        valmin=0,
        valmax=30,
        valinit=0,
        valstep=1
    )
    
    # Add tick marks and labels
    slider.ax.set_xticks(range(0, 31, 1))  # Tick marks every 1%
    slider.ax.set_xticklabels([str(x) if x % 5 == 0 else '' for x in range(0, 31, 1)])  # Labels every 5%
    
    # Add text label for current value
    text_label = ax_slider.text(0.5, 0.5, 'Compensation: 0%', 
                              transform=ax_slider.transAxes,
                              ha='center', va='center')
    
    def update(val):
        compensation = slider.val
        text_label.set_text(f'Compensation: {compensation:.0f}%')
        
        # Update normalized traces
        all_norm_compensated = []
        all_zscore_compensated = []
        
        for roi_name, traces in raw_traces.items():
            # Calculate compensated trace
            compensated_trace = calculate_compensated_trace(traces['A'], traces['B'], compensation)
            
            # Normalize and Z-score compensated trace
            norm_compensated = normalize_trace_excluding_stim(compensated_trace, z_stim_start, z_stim_end)
            zscore_compensated = zscore_trace(compensated_trace, z_stim_start, z_stim_end)
            
            # Update individual traces
            plot_objects[roi_name]['norm']['A'].set_ydata(norm_compensated)
            plot_objects[roi_name]['zscore']['A'].set_ydata(zscore_compensated)
            
            all_norm_compensated.append(norm_compensated)
            all_zscore_compensated.append(zscore_compensated)
        
        # Update average traces
        avg_norm_compensated = np.mean(all_norm_compensated, axis=0)
        avg_zscore_compensated = np.mean(all_zscore_compensated, axis=0)
        
        plot_objects['avg']['norm']['A'].set_ydata(avg_norm_compensated)
        plot_objects['avg']['zscore']['A'].set_ydata(avg_zscore_compensated)
        
        fig.canvas.draw_idle()
    
    slider.on_changed(update)
    
    plt.tight_layout()
    plt.show()

def main():
    # Get directory path from command line argument
    if len(sys.argv) != 2:
        print("Usage: python quantify_rois.py <stks_directory_path>")
        sys.exit(1)
    
    stks_dir = sys.argv[1]
    roi_dir = os.path.join(stks_dir, 'ROIs')  # ROIs are inside STKS folder
    
    # Check if directories exist
    if not os.path.exists(stks_dir):
        print(f"Error: STKS directory not found at {stks_dir}")
        sys.exit(1)
    
    if not os.path.exists(roi_dir):
        print(f"Error: ROIs directory not found at {roi_dir}")
        sys.exit(1)
    
    # Load TIFF stacks
    chanA_stack = load_tif_stack(stks_dir, 'ChanA')
    chanB_stack = load_tif_stack(stks_dir, 'ChanB')
    
    # Create average images
    avg_chanA = create_average_image(chanA_stack)
    avg_chanB = create_average_image(chanB_stack)
    
    # Load all ROIs
    rois = {}
    for roi_file in os.listdir(roi_dir):
        if roi_file.endswith('.npy'):
            roi_path = os.path.join(roi_dir, roi_file)
            roi_name = os.path.splitext(roi_file)[0]
            rois[roi_name] = load_roi(roi_path)
    
    # Define stimulation range (you can make these command line arguments if needed)
    z_stim_start = 35  # Default stimulation start
    z_stim_end = 40    # Default stimulation end
    
    # Plot ROIs on average image with traces
    plot_rois_on_image(avg_chanB, rois, chanA_stack, chanB_stack, z_stim_start, z_stim_end)
    
    # Plot summary traces
    plot_summary_traces(rois, chanA_stack, chanB_stack, z_stim_start, z_stim_end)

if __name__ == '__main__':
    main() 