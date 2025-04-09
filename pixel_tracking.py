import os
import numpy as np
import matplotlib.pyplot as plt
import tifffile
from PIL import Image
from typing import Dict, Tuple
from dataclasses import dataclass
from tqdm import tqdm

@dataclass
class PixelTrace:
    xy: Tuple[int, int]
    trace: np.ndarray
    normalized_trace: np.ndarray

def load_and_process_data(tif_path: str):
    # Load tif stack
    stack = tifffile.imread(tif_path)
    print(f"Loaded tif stack with shape: {stack.shape}")
    
    # Calculate mean image
    mean_img = np.mean(stack, axis=0)
    print(f"Mean image shape: {mean_img.shape}")
    
    # Find and load binary mask
    mask_dir = os.path.dirname(tif_path)
    mask_files = [f for f in os.listdir(mask_dir) if f.startswith('Binary_Mask') and f.endswith('.png')]
    if not mask_files:
        raise FileNotFoundError("No binary mask file found in the directory")
    
    mask_path = os.path.join(mask_dir, mask_files[0])
    print(f"Using mask file: {mask_files[0]}")
    
    # Load mask and convert to binary
    mask_img = Image.open(mask_path)
    mask = np.array(mask_img)
    print(f"Original mask shape: {mask.shape}")
    
    # Convert to binary: 0 stays 0, any non-zero value becomes 1
    mask = np.any(mask > 0, axis=2).astype(np.uint8)
    print(f"Binary mask shape: {mask.shape}")
    print(f"Number of True pixels in mask: {np.sum(mask)}")
    
    # Apply mask to mean image
    masked_mean_img = mean_img.copy()
    masked_mean_img[~mask.astype(bool)] = 0
    
    # Get coordinates of masked pixels
    y_coords, x_coords = np.where(mask)
    num_pixels = len(y_coords)
    print(f"Processing {num_pixels} masked pixels...")
    
    # Extract traces for all masked pixels at once
    traces = stack[:, y_coords, x_coords]  # Shape: (frames, num_masked_pixels)
    
    # Normalize all traces at once
    trace_mins = np.min(traces, axis=0)
    trace_maxs = np.max(traces, axis=0)
    normalized_traces = (traces - trace_mins) / (trace_maxs - trace_mins)
    
    # Store traces in dictionary with progress bar
    pixel_traces = {}
    for idx, (y, x) in enumerate(tqdm(zip(y_coords, x_coords), total=num_pixels, desc="Processing pixels")):
        pixel_traces[(x, y)] = PixelTrace((x, y), traces[:, idx], normalized_traces[:, idx])
    
    print(f"Extracted {len(pixel_traces)} pixel traces")
    return masked_mean_img, pixel_traces

def plot_traces(masked_mean_img: np.ndarray, pixel_traces: Dict[Tuple[int, int], PixelTrace]):
    if not pixel_traces:
        print("No pixel traces available to plot")
        return
        
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot masked mean image
    im1 = ax1.imshow(masked_mean_img, cmap='coolwarm')
    ax1.set_title('Masked Mean Image')
    plt.colorbar(im1, ax=ax1)
    
    # Create raster plot of normalized traces
    traces = np.array([pt.normalized_trace for pt in pixel_traces.values()])
    im2 = ax2.imshow(traces, aspect='auto', cmap='jet')
    ax2.set_xlabel('Time (frames)')
    ax2.set_ylabel('Pixel Index')
    ax2.set_title('Normalized Pixel Traces')
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    plt.show()

def analyze_pixel_traces(tif_path: str):
    masked_mean_img, pixel_traces = load_and_process_data(tif_path)
    plot_traces(masked_mean_img, pixel_traces)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Analyze pixel traces from a tif stack')
    parser.add_argument('tif_path', type=str, help='Path to the tif stack file')
    args = parser.parse_args()
    
    if not os.path.exists(args.tif_path):
        print(f"Error: File {args.tif_path} does not exist")
    else:
        analyze_pixel_traces(args.tif_path) 