import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from tifffile import imread
from skimage.filters import threshold_otsu
from PIL import Image
import matplotlib.image as mpimg

def analyze_tif_stack(tif_path):
    """
    Analyze a .tif stack and create visualizations with interactive threshold adjustment.
    
    Parameters:
    -----------
    tif_path : str
        Path to the .tif stack file
    
    Returns:
    --------
    tuple
        (mask, chosen_threshold) where chosen_threshold is None if no threshold was selected
    """
    # Load the tif stack
    stack = imread(tif_path)
    
    # Calculate the average along the z-direction
    avg_image = np.mean(stack, axis=0)
    
    # Load the MeanImg.png from the same directory
    tif_dir = os.path.dirname(tif_path)
    mean_img_path = os.path.join(tif_dir, "MeanImg.png")
    
    if os.path.exists(mean_img_path):
        # Load the mean image as background
        mean_img = np.array(Image.open(mean_img_path))
        # Convert to grayscale if it's RGB
        if len(mean_img.shape) == 3:
            mean_img = np.mean(mean_img, axis=2)
            
        # Check if sizes match and resize if needed
        if mean_img.shape != avg_image.shape:
            print(f"Warning: MeanImg.png size {mean_img.shape} doesn't match tif stack size {avg_image.shape}")
            print("Resizing MeanImg.png to match tif stack size")
            from skimage.transform import resize
            mean_img = resize(mean_img, avg_image.shape, anti_aliasing=True)
    else:
        print(f"Warning: MeanImg.png not found in {tif_dir}. Using black background.")
        mean_img = np.zeros_like(avg_image)
    
    # Create figure with three subplots and space for slider and button
    fig = plt.figure(figsize=(18, 6))
    gs = fig.add_gridspec(2, 3, height_ratios=[4, 1])
    
    # Create subplots
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax_slider = fig.add_subplot(gs[1, 0:2])
    ax_button = fig.add_subplot(gs[1, 2])
    
    # Plot 1: Average image
    im1 = ax1.imshow(avg_image, cmap='gray')
    ax1.set_title('Average Image (Z-direction)')
    plt.colorbar(im1, ax=ax1, label='Intensity')
    
    # Plot 2: Histogram
    min_val = np.min(avg_image)
    max_val = np.max(avg_image)
    counts, bins, _ = ax2.hist(avg_image.flatten(), bins=100, range=(min_val, max_val))
    ax2.set_title('Histogram of Pixel Values')
    ax2.set_xlabel('Pixel Value')
    ax2.set_ylabel('Count')
    ax2.set_xlim(min_val, max_val)
    
    # Add legend with min and max values
    ax2.text(0.95, 0.95, f'Min: {min_val:.2f}\nMax: {max_val:.2f}', 
             transform=ax2.transAxes, 
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Calculate initial Otsu threshold
    initial_threshold = threshold_otsu(avg_image)
    
    # Create initial mask
    mask = avg_image > initial_threshold
    
    # Plot 3: Mask with background
    # Display the background image with coolwarm colormap
    ax3.imshow(mean_img, cmap='coolwarm')
    
    # Create a colored overlay for the mask
    colored_mask = np.zeros((*mask.shape, 4), dtype=np.uint8)  # RGBA
    colored_mask[mask, 0] = 255  # Red channel
    colored_mask[mask, 1] = 0    # Green channel
    colored_mask[mask, 2] = 0    # Blue channel
    colored_mask[mask, 3] = 128  # Alpha channel (semi-transparent)
    
    # Overlay the mask
    im3 = ax3.imshow(colored_mask)
    ax3.set_title(f'Mask (Threshold: {initial_threshold:.2f})')
    
    # Add vertical line in histogram to show threshold
    threshold_line = ax2.axvline(x=initial_threshold, color='r', linestyle='--', 
                               label=f'Threshold: {initial_threshold:.2f}')
    ax2.legend()
    
    # Create slider
    slider = Slider(
        ax=ax_slider,
        label='Threshold',
        valmin=min_val,
        valmax=max_val,
        valinit=initial_threshold
    )
    
    # Variable to store the chosen threshold
    chosen_threshold = [None]
    
    # Create button
    button = Button(ax_button, 'Save Threshold')
    
    # Update function for slider
    def update(val):
        threshold = slider.val
        mask = avg_image > threshold
        # Update the colored mask
        colored_mask = np.zeros((*mask.shape, 4), dtype=np.uint8)
        colored_mask[mask, 0] = 255  # Red channel
        colored_mask[mask, 1] = 0    # Green channel
        colored_mask[mask, 2] = 0    # Blue channel
        colored_mask[mask, 3] = 128  # Alpha channel (semi-transparent)
        im3.set_data(colored_mask)
        ax3.set_title(f'Mask (Threshold: {threshold:.2f})')
        threshold_line.set_xdata([threshold, threshold])
        fig.canvas.draw_idle()
    
    # Button click function
    def save_threshold(event):
        threshold = slider.val
        chosen_threshold[0] = threshold
        print(f"\nThreshold value {threshold:.2f} has been saved!")
        
        # Save the binary mask
        mask = avg_image > threshold
        rounded_threshold = int(round(threshold))
        mask_filename = f"Binary_Mask_Thr{rounded_threshold}.png"
        mask_path = os.path.join(tif_dir, mask_filename)
        
        # Convert boolean mask to uint8 (0 or 255)
        mask_img = np.zeros_like(avg_image, dtype=np.uint8)
        mask_img[mask] = 255
        
        # Save the mask
        mpimg.imsave(mask_path, mask_img, cmap='binary')
        print(f"Binary mask saved as: {mask_path}")
        
        button.label.set_text('Saved!')
        plt.draw()
    
    # Register the update function with the slider
    slider.on_changed(update)
    
    # Register the button click function
    button.on_clicked(save_threshold)
    
    plt.tight_layout()
    plt.show()
    
    return mask, chosen_threshold[0]

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze a .tif stack and create visualizations')
    parser.add_argument('tif_path', type=str, help='Path to the .tif stack file')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.tif_path):
        raise FileNotFoundError(f"The file {args.tif_path} does not exist")
    
    mask, threshold = analyze_tif_stack(args.tif_path)
    if threshold is not None:
        print(f"\nFinal threshold value: {threshold:.2f}") 