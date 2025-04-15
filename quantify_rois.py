import os
import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread
import argparse

def load_tif_stack(tif_path):
    """
    Load a tif stack from the given path.
    
    Parameters:
    -----------
    tif_path : str
        Path to the tif stack file
        
    Returns:
    --------
    numpy.ndarray
        The loaded tif stack
    """
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

def plot_rois_on_image(average_image, roi_dir):
    """
    Plot the average image with ROIs overlaid.
    
    Parameters:
    -----------
    average_image : numpy.ndarray
        The average image
    roi_dir : str
        Directory containing ROI files
    """
    # Create figure
    plt.figure(figsize=(10, 10))
    
    # Plot average image
    plt.imshow(average_image, cmap='viridis')
    plt.colorbar(label='Intensity')
    
    # Load and plot each ROI
    for roi_file in os.listdir(roi_dir):
        if roi_file.endswith('.npy'):
            roi_path = os.path.join(roi_dir, roi_file)
            roi_coords = load_roi(roi_path)
            
            # Plot ROI as a closed polygon
            roi_coords_closed = np.vstack([roi_coords, roi_coords[0]])  # Close the polygon
            plt.plot(roi_coords_closed[:, 0], roi_coords_closed[:, 1], 'r-', linewidth=2)
            
            # Add label (filename without extension)
            roi_name = os.path.splitext(roi_file)[0]
            # Calculate center of ROI for label placement
            center_x = np.mean(roi_coords[:, 0])
            center_y = np.mean(roi_coords[:, 1])
            plt.text(center_x, center_y, roi_name, color='red', 
                    horizontalalignment='center', verticalalignment='center',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    plt.title('Average Image with ROIs')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.tight_layout()
    plt.show()

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Quantify ROIs from image stacks')
    parser.add_argument('directory', type=str, help='Directory containing STKS folder')
    
    args = parser.parse_args()
    
    # Construct paths
    stks_dir = os.path.join(args.directory, 'STKS')
    roi_dir = os.path.join(stks_dir, 'ROIs')  # ROIs folder is inside STKS
    
    # Check if directories exist
    if not os.path.exists(stks_dir):
        raise ValueError(f"STKS directory not found: {stks_dir}")
    if not os.path.exists(roi_dir):
        raise ValueError(f"ROIs directory not found: {roi_dir}")
    
    # Load ChanB stack
    chan_b_path = os.path.join(stks_dir, 'ChanB_stk.tif')
    stack_b = load_tif_stack(chan_b_path)
    
    # Create average image
    average_image = create_average_image(stack_b)
    
    # Plot ROIs on average image
    plot_rois_on_image(average_image, roi_dir)

if __name__ == '__main__':
    main() 