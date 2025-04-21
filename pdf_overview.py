"""
PDF Overview Generator

This script generates a PDF report showing ROI visualizations for each mouse.
For each mouse, it shows:
1. An average image of a representative stack
2. The same image with ROIs overlaid
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from skimage.draw import polygon
from matplotlib.backends.backend_pdf import PdfPages

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

def plot_rois_on_image(avg_image, rois):
    """
    Plot the average image with ROIs overlaid.
    
    Args:
        avg_image (numpy.ndarray): The average image
        rois (list): List of ROI coordinates
        
    Returns:
        matplotlib.figure.Figure: The figure with the plot
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(avg_image, cmap='viridis')
    
    # Plot each ROI
    for i, roi_coords in enumerate(rois):
        roi_coords_closed = np.vstack([roi_coords, roi_coords[0]])
        ax.plot(roi_coords_closed[:, 0], roi_coords_closed[:, 1], 'r-', linewidth=2)
        center_x = np.mean(roi_coords[:, 0])
        center_y = np.mean(roi_coords[:, 1])
        ax.text(center_x, center_y, f'ROI {i+1}', color='red',
                horizontalalignment='center', verticalalignment='center',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    ax.axis('off')
    return fig

def generate_pdf_report(df, parent_directory, output_path):
    """
    Generate a PDF report with ROI visualizations for each mouse.
    
    Args:
        df (pandas.DataFrame): DataFrame containing ROI data
        parent_directory (str): Path to the parent directory
        output_path (str): Path to save the PDF report
    """
    # Create PDF pages
    with PdfPages(output_path) as pdf:
        # Get unique mice
        unique_mice = df['MOUSE'].unique()
        print(f"\nFound {len(unique_mice)} unique mice: {unique_mice}")
        
        for mouse in unique_mice:
            print(f"\nProcessing mouse: {mouse}")
            # Get the first experiment for this mouse
            mouse_data = df[df['MOUSE'] == mouse]
            first_exp = mouse_data.iloc[0]
            print(f"First experiment for {mouse}: {first_exp['EXP']}")
            
            # Construct path to the TIFF stack
            stks_dir = os.path.join(parent_directory, mouse, first_exp['EXP'], 'STKS')
            tif_path = os.path.join(stks_dir, 'ChanB_stk.tif')
            print(f"Looking for TIFF stack at: {tif_path}")
            print(f"Directory exists: {os.path.exists(os.path.dirname(tif_path))}")
            print(f"File exists: {os.path.exists(tif_path)}")
            
            # Load and process the stack
            stack = imread(tif_path)
            avg_image = create_average_image(stack)
            
            # Create a new page for the average image
            fig = plt.figure(figsize=(12, 6))
            
            # Plot average image on the left
            ax1 = fig.add_subplot(121)
            ax1.imshow(avg_image, cmap='viridis')
            ax1.set_title(f'Average Image - {mouse}')
            ax1.axis('off')
            
            # Plot ROIs on the right
            ax2 = fig.add_subplot(122)
            ax2.imshow(avg_image, cmap='viridis')
            
            # Get ROI coordinates for this mouse
            rois = []
            for _, row in mouse_data.iterrows():
                roi_dir = os.path.join(stks_dir, 'ROIs')
                roi_path = os.path.join(roi_dir, f"ROI#{row['ROI#']}.npy")
                print(f"\nLooking for ROI file: {roi_path}")
                print(f"ROI directory exists: {os.path.exists(roi_dir)}")
                print(f"ROI file exists: {os.path.exists(roi_path)}")
                if os.path.exists(roi_path):
                    print(f"Found ROI file: {roi_path}")
                    roi_coords = np.load(roi_path)
                    print(f"ROI coordinates shape: {roi_coords.shape}")
                    rois.append(roi_coords)
                    roi_coords_closed = np.vstack([roi_coords, roi_coords[0]])
                    ax2.plot(roi_coords_closed[:, 0], roi_coords_closed[:, 1], 'r-', linewidth=2)
                    center_x = np.mean(roi_coords[:, 0])
                    center_y = np.mean(roi_coords[:, 1])
                    ax2.text(center_x, center_y, f'ROI {row["ROI#"]}', color='red',
                            horizontalalignment='center', verticalalignment='center',
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
                else:
                    print(f"ROI file not found: {roi_path}")
            
            print(f"Found {len(rois)} ROIs for mouse {mouse}")
            
            ax2.set_title(f'ROIs - {mouse}')
            ax2.axis('off')
            
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()
            
            # Create a new page for each entry's detailed plots
            for _, row in mouse_data.iterrows():
                # Create a new figure for this entry
                fig = plt.figure(figsize=(15, 5))
                
                # 1. ROI zoom-in
                ax1 = fig.add_subplot(131)
                roi_coords = np.load(os.path.join(stks_dir, 'ROIs', f"ROI#{row['ROI#']}.npy"))
                center_x = np.mean(roi_coords[:, 0])
                center_y = np.mean(roi_coords[:, 1])
                
                # Calculate ROI dimensions
                width = np.max(roi_coords[:, 0]) - np.min(roi_coords[:, 0])
                height = np.max(roi_coords[:, 1]) - np.min(roi_coords[:, 1])
                max_dim = max(width, height) * 1.2  # 20% larger
                
                # Create square crop
                y_min = max(0, int(center_y - max_dim/2))
                y_max = min(avg_image.shape[0], int(center_y + max_dim/2))
                x_min = max(0, int(center_x - max_dim/2))
                x_max = min(avg_image.shape[1], int(center_x + max_dim/2))
                
                # Ensure square dimensions
                crop_size = min(y_max - y_min, x_max - x_min)
                y_center = (y_min + y_max) // 2
                x_center = (x_min + x_max) // 2
                y_min = y_center - crop_size // 2
                y_max = y_center + crop_size // 2
                x_min = x_center - crop_size // 2
                x_max = x_center + crop_size // 2
                
                # Plot the zoomed ROI
                ax1.imshow(avg_image[y_min:y_max, x_min:x_max], cmap='viridis')
                ax1.set_title(f'ROI#{row["ROI#"]}')
                ax1.axis('off')
                
                # 2. Normalized traces
                ax2 = fig.add_subplot(132)
                ax2.plot(row['ChanA_norm_trc'], 'r-', label='ChanA')
                ax2.plot(row['ChanB_norm_trc'], 'g-', label='ChanB')
                ax2.set_title(row['EXP'])
                ax2.set_xlabel('Frames')
                ax2.set_ylabel('Normalized values')
                ax2.set_ylim([-0.5, 1.5])  # Set fixed y-axis limits for normalized traces
                ax2.legend()
                
                # 3. Z-scored traces
                ax3 = fig.add_subplot(133)
                ax3.plot(row['ChanA_Z_trc'], 'r-', label='ChanA')
                ax3.plot(row['ChanB_Z_trc'], 'g-', label='ChanB')
                ax3.set_xlabel('Frames')
                ax3.set_ylabel('Z-scored values')
                # Set y-axis limits for z-scored traces
                max_chanA = np.max(row['ChanA_Z_trc'])
                ax3.set_ylim([-1, 1.5 * max_chanA])
                ax3.legend()
                
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close()

def main(parent_directory):
    """
    Main function to generate the PDF report.
    
    Args:
        parent_directory (str): Path to the parent directory
    """
    print(f"\nParent directory: {parent_directory}")
    print(f"Directory exists: {os.path.exists(parent_directory)}")
    
    # Load the pickle file
    pickle_path = os.path.join(parent_directory, 'ROI_quant_overview.pkl')
    print(f"\nLoading pickle file from: {pickle_path}")
    print(f"Pickle file exists: {os.path.exists(pickle_path)}")
    df = load_pickle_data(pickle_path)
    
    # Generate the PDF report
    output_path = os.path.join(parent_directory, 'ROI_overview.pdf')
    generate_pdf_report(df, parent_directory, output_path)
    print(f"\nPDF report generated: {output_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python pdf_overview.py <parent_directory>")
        sys.exit(1)
    
    parent_directory = sys.argv[1]
    main(parent_directory) 