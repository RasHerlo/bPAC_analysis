import os
import numpy as np
from tifffile import imread, imwrite
import argparse
import sys
from pathlib import Path

def find_tif_files(directory, prefix):
    """
    Find all .tif files with the given prefix in the directory, sorted numerically.
    
    Args:
        directory (str): Directory to search in
        prefix (str): File prefix (e.g., 'ChanA' or 'ChanB')
        
    Returns:
        list: Sorted list of file paths
    """
    # Get all files with the prefix
    files = [f for f in os.listdir(directory) if f.startswith(prefix) and f.endswith('.tif')]
    
    # Sort files numerically based on the last number in the filename
    def get_numbers(filename):
        # Remove the extension first
        name_without_ext = os.path.splitext(filename)[0]
        # Get the last part after the last underscore
        last_part = name_without_ext.split('_')[-1]
        try:
            # Convert the last part to integer
            return int(last_part)
        except ValueError:
            # If conversion fails, return infinity to sort at the end
            return float('inf')
    
    files.sort(key=get_numbers)
    
    # Return full paths
    return [os.path.join(directory, f) for f in files]

def create_stack_from_files(file_paths):
    """
    Create a stack from a list of TIFF files.
    
    Args:
        file_paths (list): List of file paths to TIFF files
        
    Returns:
        numpy.ndarray: The combined stack
    """
    if not file_paths:
        return None
    
    print(f"\nInitiating stack creation from {len(file_paths)} files")
    print(f"First file: {os.path.basename(file_paths[0])}")
    print(f"Last file: {os.path.basename(file_paths[-1])}")
    
    # Read first image to get shape
    first_img = imread(file_paths[0], is_ome=False)
    stack_shape = (len(file_paths),) + first_img.shape
    print(f"Creating stack with shape: {stack_shape}")
    
    # Create empty stack
    stack = np.zeros(stack_shape, dtype=first_img.dtype)
    stack[0] = first_img  # Store the first image we already loaded
    
    # Fill stack with images
    for i, file_path in enumerate(file_paths[1:], start=1):  # Start from second image
        print(f"Reading file {i+1}/{len(file_paths)}: {os.path.basename(file_path)}", end='\r')
        try:
            stack[i] = imread(file_path, is_ome=False)
        except Exception as e:
            print(f"\nError reading file {file_path}: {str(e)}")
            return None
    
    print(f"\nCompleted stack creation with final shape: {stack.shape}")
    return stack

def process_directory(directory):
    """
    Process a directory to create stacks if needed.
    
    Args:
        directory (str): Directory to process
    """
    # Check if we have the trigger file
    trigger_file = "ChanA_001_001_001_001.tif"
    if not os.path.exists(os.path.join(directory, trigger_file)):
        return
    
    # Check if STKS folder exists
    stks_dir = os.path.join(directory, "STKS")
    if os.path.exists(stks_dir):
        print(f"STKS folder already exists in {directory}, skipping...")
        return
    
    print(f"\nProcessing directory: {directory}")
    
    # Create STKS folder
    os.makedirs(stks_dir)
    print("Created STKS folder")
    
    # Process Channel A
    print("\nProcessing Channel A files...")
    chanA_files = find_tif_files(directory, "ChanA")
    if chanA_files:
        print(f"Found {len(chanA_files)} ChanA files")
        chanA_stack = create_stack_from_files(chanA_files)
        if chanA_stack is not None:
            output_path = os.path.join(stks_dir, "ChanA_stk.tif")
            print(f"Saving ChanA stack to: {output_path}")
            imwrite(output_path, chanA_stack)
            print("ChanA stack saved successfully")
    
    # Process Channel B
    print("\nProcessing Channel B files...")
    chanB_files = find_tif_files(directory, "ChanB")
    if chanB_files:
        print(f"Found {len(chanB_files)} ChanB files")
        chanB_stack = create_stack_from_files(chanB_files)
        if chanB_stack is not None:
            output_path = os.path.join(stks_dir, "ChanB_stk.tif")
            print(f"Saving ChanB stack to: {output_path}")
            imwrite(output_path, chanB_stack)
            print("ChanB stack saved successfully")

def main():
    # Get directory path from command line argument
    if len(sys.argv) != 2:
        print("Usage: python generate_stks.py <parent_directory>")
        sys.exit(1)
    
    parent_dir = sys.argv[1]
    
    # Check if directory exists
    if not os.path.exists(parent_dir):
        print(f"Error: Directory not found at {parent_dir}")
        sys.exit(1)
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(parent_dir):
        process_directory(root)

if __name__ == '__main__':
    main() 