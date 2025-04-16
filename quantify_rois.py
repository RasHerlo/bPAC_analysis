import os
import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread
import argparse
import sys
from skimage.draw import polygon

def load_tif_stack(directory, channel):
    """
    Load a tif stack from the given directory and channel.
    
    Parameters:
    -----------
    directory : str
        Directory containing the TIFF stack
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

def plot_rois_on_image(avg_image, rois, stack):
    """
    Plot the average image with ROIs overlaid and create detailed subplots for each ROI.
    
    Args:
        avg_image (numpy.ndarray): The average image to plot
        rois (dict): Dictionary of ROIs with their coordinates
        stack (numpy.ndarray): The image stack for calculating traces
    """
    # Calculate figure size based on number of ROIs
    n_rois = len(rois)
    n_rows = max(3, 3 + ((n_rois - 3) + 1) // 2)  # At least 3 rows, add rows for additional ROIs
    n_cols = 6  # 3 for main image + 3 for ROI details
    
    # Create figure with appropriate size
    fig = plt.figure(figsize=(n_cols * 3, n_rows * 3))
    gs = plt.GridSpec(n_rows, n_cols, figure=fig)
    
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
        
        # Calculate and plot trace
        ax_trace = fig.add_subplot(gs[row, col+1:col+3])
        trace = calculate_roi_trace(stack, roi_coords)
        ax_trace.plot(trace, 'k-', label=roi_name)
        ax_trace.legend()
        ax_trace.set_title(f'Trace: {roi_name}')
        ax_trace.set_xlabel('Frame')
        ax_trace.set_ylabel('Average Intensity')
    
    plt.tight_layout()
    plt.show()

def main():
    # Get directory path from command line argument
    if len(sys.argv) != 2:
        print("Usage: python quantify_rois.py <directory_path>")
        sys.exit(1)
    
    directory = sys.argv[1]
    stks_dir = os.path.join(directory, 'STKS')
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
    
    # Plot ROIs on average image with traces
    plot_rois_on_image(avg_chanB, rois, chanB_stack)  # Using ChanB as it's typically the signal channel

if __name__ == '__main__':
    main() 