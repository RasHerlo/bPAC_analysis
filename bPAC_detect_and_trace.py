import os
import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread
from skimage import exposure
import argparse

def load_tif_stack(tif_path):
    """
    Load a tif stack from the specified path.
    
    Parameters:
    -----------
    tif_path : str
        Path to the tif stack file
    
    Returns:
    --------
    numpy.ndarray
        The loaded tif stack
    """
    if not os.path.isfile(tif_path):
        raise FileNotFoundError(f"Tif file not found at: {tif_path}")
    
    print(f"Loading tif stack from: {tif_path}")
    stack = imread(tif_path)
    print(f"Loaded stack with shape: {stack.shape}")
    return stack

def create_heatmaps(stack, z_range1, z_range2):
    """
    Create average and ratio heatmaps from the tif stack.
    
    Parameters:
    -----------
    stack : numpy.ndarray
        The loaded tif stack
    z_range1 : tuple
        (start, end) for the first z-range
    z_range2 : tuple
        (start, end) for the second z-range
    
    Returns:
    --------
    tuple
        (average_heatmap, ratio_heatmap)
    """
    # Create average heatmap (average across all z)
    average_heatmap = np.mean(stack, axis=0)
    
    # Create ratio heatmap
    avg1 = np.mean(stack[z_range1[0]:z_range1[1]], axis=0)
    avg2 = np.mean(stack[z_range2[0]:z_range2[1]], axis=0)
    ratio_heatmap = avg1 / avg2
    
    # Handle division by zero
    ratio_heatmap[~np.isfinite(ratio_heatmap)] = 0
    
    return average_heatmap, ratio_heatmap

def stretch_heatmaps(average_heatmap, ratio_heatmap):
    """
    Stretch the heatmaps using skimage exposure.
    
    Parameters:
    -----------
    average_heatmap : numpy.ndarray
        The average heatmap
    ratio_heatmap : numpy.ndarray
        The ratio heatmap
    
    Returns:
    --------
    tuple
        (stretched_average, stretched_ratio)
    """
    # Stretch the heatmaps
    p2_average, p98_average = np.percentile(average_heatmap, (2, 98))
    p2_ratio, p98_ratio = np.percentile(ratio_heatmap, (2, 98))
    
    stretched_average = exposure.rescale_intensity(average_heatmap, 
                                                 in_range=(p2_average, p98_average))
    stretched_ratio = exposure.rescale_intensity(ratio_heatmap, 
                                               in_range=(p2_ratio, p98_ratio))
    
    return stretched_average, stretched_ratio

def get_top_pixels(ratio_heatmap, n_pixels=1000):
    """
    Get the coordinates of the top n pixels with highest ratio values.
    
    Parameters:
    -----------
    ratio_heatmap : numpy.ndarray
        The ratio heatmap
    n_pixels : int
        Number of top pixels to select
    
    Returns:
    --------
    tuple
        (y_coords, x_coords) of the top n pixels
    """
    # Flatten the ratio heatmap and get indices of top n pixels
    flat_indices = np.argsort(ratio_heatmap.flatten())[-n_pixels:]
    
    # Convert flat indices to 2D coordinates
    y_coords, x_coords = np.unravel_index(flat_indices, ratio_heatmap.shape)
    
    return y_coords, x_coords

def extract_and_normalize_traces(stack, y_coords, x_coords):
    """
    Extract and normalize traces for the specified coordinates.
    
    Parameters:
    -----------
    stack : numpy.ndarray
        The loaded tif stack
    y_coords : numpy.ndarray
        Y coordinates of pixels
    x_coords : numpy.ndarray
        X coordinates of pixels
    
    Returns:
    --------
    numpy.ndarray
        Normalized traces for the specified pixels
    """
    # Extract traces
    traces = stack[:, y_coords, x_coords]
    
    # Normalize each trace to [0, 1]
    trace_mins = np.min(traces, axis=0)
    trace_maxs = np.max(traces, axis=0)
    normalized_traces = (traces - trace_mins) / (trace_maxs - trace_mins)
    
    return normalized_traces

def create_mask_image(shape, y_coords, x_coords):
    """
    Create a binary mask image for the top pixels.
    
    Parameters:
    -----------
    shape : tuple
        Shape of the output mask image
    y_coords : numpy.ndarray
        Y coordinates of pixels
    x_coords : numpy.ndarray
        X coordinates of pixels
    
    Returns:
    --------
    numpy.ndarray
        Binary mask image
    """
    mask = np.zeros(shape, dtype=np.uint8)
    mask[y_coords, x_coords] = 1
    return mask

def plot_results(stretched_average, stretched_ratio, normalized_traces, mask_image):
    """
    Create a figure with four subplots showing the results.
    
    Parameters:
    -----------
    stretched_average : numpy.ndarray
        The stretched average heatmap
    stretched_ratio : numpy.ndarray
        The stretched ratio heatmap
    normalized_traces : numpy.ndarray
        Normalized traces for top pixels
    mask_image : numpy.ndarray
        Binary mask image for top pixels
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot average heatmap
    im1 = ax1.imshow(stretched_average, cmap='viridis')
    ax1.set_title('Average Heatmap')
    plt.colorbar(im1, ax=ax1, label='Intensity')
    
    # Plot ratio heatmap
    im2 = ax2.imshow(stretched_ratio, cmap='viridis')
    ax2.set_title('Ratio Heatmap')
    plt.colorbar(im2, ax=ax2, label='Ratio')
    
    # Plot normalized traces (transposed)
    im3 = ax3.imshow(normalized_traces.T, aspect='auto', cmap='viridis')
    ax3.set_xlabel('Time (frames)')
    ax3.set_ylabel('Pixel Index')
    ax3.set_title('Normalized Traces (Top 1000)')
    plt.colorbar(im3, ax=ax3, label='Normalized Intensity')
    
    # Plot mask image (inverted colors)
    im4 = ax4.imshow(1 - mask_image, cmap='gray')
    ax4.set_title('Top 1000 Pixels Mask')
    
    plt.tight_layout()
    plt.show()

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process tif stack and create heatmaps')
    parser.add_argument('tif_path', type=str, help='Path to the tif stack file')
    parser.add_argument('--z1_start', type=int, required=True, help='Start of first z-range')
    parser.add_argument('--z1_end', type=int, required=True, help='End of first z-range')
    parser.add_argument('--z2_start', type=int, required=True, help='Start of second z-range')
    parser.add_argument('--z2_end', type=int, required=True, help='End of second z-range')
    
    args = parser.parse_args()
    
    # Load tif stack
    stack = load_tif_stack(args.tif_path)
    
    # Validate z-ranges
    if not (0 <= args.z1_start < args.z1_end <= stack.shape[0] and 
            0 <= args.z2_start < args.z2_end <= stack.shape[0]):
        raise ValueError("Invalid z-ranges. Must be within stack dimensions.")
    
    # Create heatmaps
    average_heatmap, ratio_heatmap = create_heatmaps(
        stack, 
        (args.z1_start, args.z1_end), 
        (args.z2_start, args.z2_end)
    )
    
    # Stretch heatmaps
    stretched_average, stretched_ratio = stretch_heatmaps(average_heatmap, ratio_heatmap)
    
    # Get top 1000 pixels
    y_coords, x_coords = get_top_pixels(ratio_heatmap)
    
    # Extract and normalize traces
    normalized_traces = extract_and_normalize_traces(stack, y_coords, x_coords)
    
    # Create mask image
    mask_image = create_mask_image(ratio_heatmap.shape, y_coords, x_coords)
    
    # Plot results
    plot_results(stretched_average, stretched_ratio, normalized_traces, mask_image)

if __name__ == "__main__":
    main() 