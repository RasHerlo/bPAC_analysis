"""
PDF ROI Quantification Overview Generator

This script generates a comprehensive PDF report of ROI quantification data from a parent directory.
The process involves several steps:

1) Directory and Data Import:
   - Takes a parent directory as input
   - Locates and imports ROI data from an Excel spreadsheet
   - Each row in the spreadsheet represents a single ROI with its associated metadata

2) Data Processing:
   - Processes the imported ROI data
   - Calculates additional metrics and statistics
   - Organizes data for visualization

3) Report Generation:
   - Creates a PDF report with multiple sections
   - Includes visualizations of ROI data
   - Provides statistical summaries
   - Organizes information in a clear, readable format

Dependencies:
- pandas: For data handling and Excel import
- matplotlib: For data visualization
- reportlab: For PDF generation
- numpy: For numerical calculations
- skimage: For image processing
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from skimage.draw import polygon

def find_excel_files(directory):
    """
    Search for Excel files in the given directory and its subdirectories.
    
    Args:
        directory (str): Path to the directory to search
        
    Returns:
        list: List of paths to Excel files found
    """
    excel_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(('.xlsx', '.xls')):
                excel_files.append(os.path.join(root, file))
    return excel_files

def import_excel_data(file_path):
    """
    Import data from an Excel file, first removing empty columns and rows,
    then using the remaining top row as column headers.
    
    Args:
        file_path (str): Path to the Excel file
        
    Returns:
        pandas.DataFrame: DataFrame containing the Excel data
    """
    try:
        # Skip the temporary Excel file that starts with ~$
        if os.path.basename(file_path).startswith('~$'):
            return None
            
        # First read the Excel file without headers
        df = pd.read_excel(file_path, header=None)
        
        # Find the first non-empty row that contains actual data
        # (skip any initial empty rows or header rows)
        start_row = None
        for i in range(len(df)):
            if not df.iloc[i].isna().all():
                start_row = i
                break
                
        if start_row is not None:
            # Use this row as the header
            df.columns = df.iloc[start_row]
            # Remove all rows before and including the header row
            df = df.iloc[start_row+1:]
            
            # Remove any completely empty columns
            df = df.dropna(axis=1, how='all')
            
            # Remove any completely empty rows
            df = df.dropna(axis=0, how='all')
            
            # Reset the index
            df = df.reset_index(drop=True)
        
        print(f"\nFound Excel file: {file_path}")
        print("\nDataFrame shape:", df.shape)
        print("\nColumns:", df.columns.tolist())
        print("\nFirst few rows:")
        print(df.head())
        
        return df
    except Exception as e:
        print(f"Error reading {file_path}: {str(e)}")
        return None

def visualize_dataframe(df):
    """
    Create a figure displaying the DataFrame as a table.
    
    Args:
        df (pandas.DataFrame): DataFrame to visualize
    """
    # Create figure and axis with more height per row
    fig, ax = plt.subplots(figsize=(15, len(df)*0.5 + 2))  # More height per row and extra space for header
    
    # Hide axes
    ax.axis('tight')
    ax.axis('off')
    
    # Create a copy of the DataFrame for display
    display_df = df.copy()
    
    # Convert trace columns to compact format
    for col in display_df.columns:
        if 'trc' in col.lower():  # If column contains traces
            display_df[col] = display_df[col].apply(lambda x: f"Trace [length: {len(x)}]" if isinstance(x, np.ndarray) else str(x))
    
    # Convert all values to strings and right-align numbers
    cell_text = display_df.applymap(str).values
    
    # Create table
    table = ax.table(cellText=cell_text,
                    colLabels=display_df.columns,
                    cellLoc='left',
                    loc='center',
                    colColours=['lightblue']*len(display_df.columns))
    
    # Adjust font size and cell size
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    
    # Make the rows taller
    table.scale(1.2, 2)
    
    # Color alternating rows for better readability
    for i in range(len(display_df)):
        for j in range(len(display_df.columns)):
            cell = table[(i+1, j)]
            if i % 2 == 0:
                cell.set_facecolor('#f0f0f0')
    
    plt.title('ROI Database Overview', pad=20, size=14)
    plt.tight_layout()
    plt.show()

def extract_roi_trace(tiff_path, roi_path):
    """
    Extract the average intensity trace from a TIFF stack for a given ROI.
    
    Args:
        tiff_path (str): Path to the TIFF stack
        roi_path (str): Path to the ROI coordinates file (.npy)
        
    Returns:
        numpy.ndarray: Average intensity trace across z-slices
    """
    # Load the TIFF stack
    stack = io.imread(tiff_path)
    
    # Load the ROI coordinates
    roi_coords = np.load(roi_path)
    
    # Create a mask for the ROI
    mask = np.zeros(stack.shape[1:], dtype=bool)
    rr, cc = polygon(roi_coords[:, 1], roi_coords[:, 0], stack.shape[1:])
    mask[rr, cc] = True
    
    # Calculate average intensity for each frame
    trace = np.mean(stack[:, mask], axis=1)
    
    return trace

def find_stimulation_point(trace_a, trace_b):
    """
    Find the stimulation point in the traces by identifying the minimum value.
    
    Args:
        trace_a (numpy.ndarray): Channel A trace
        trace_b (numpy.ndarray): Channel B trace
        
    Returns:
        int: Stimulation point frame number
    """
    # Find minimum values and their first occurrences
    min_a_idx = np.where(trace_a == np.min(trace_a))[0][0]
    min_b_idx = np.where(trace_b == np.min(trace_b))[0][0]
    
    # Set stimulation point one frame before minimum
    stim_a = min_a_idx - 1
    stim_b = min_b_idx - 1
    
    # Check if stimulation points are within 1 frame of each other
    if abs(stim_a - stim_b) > 1:
        raise ValueError(f"Stimulation points differ by more than 1 frame (A: {stim_a}, B: {stim_b})")
    
    # Return the earlier stimulation point
    return min(stim_a, stim_b)

def add_trace_columns(df, parent_directory):
    """
    Add ChanA and ChanB trace columns to the DataFrame.
    
    Args:
        df (pandas.DataFrame): DataFrame containing ROI information
        parent_directory (str): Path to the parent directory
        
    Returns:
        pandas.DataFrame: Updated DataFrame with trace columns
    """
    # Initialize new columns
    df['ChanA_raw_trc'] = None
    df['ChanB_raw_trc'] = None
    df['Stim'] = None
    df['ChanA_cut_trc'] = None
    df['ChanB_cut_trc'] = None
    
    # Process each row
    for idx, row in df.iterrows():
        # Construct paths
        exp_dir = os.path.join(parent_directory, row['MOUSE'], row['EXP'], 'STKS')
        roi_path = os.path.join(exp_dir, 'ROIs', f"ROI#{row['ROI#']}.npy")
        
        # Extract traces
        chan_a_trace = extract_roi_trace(os.path.join(exp_dir, 'ChanA_stk.tif'), roi_path)
        chan_b_trace = extract_roi_trace(os.path.join(exp_dir, 'ChanB_stk.tif'), roi_path)
        
        # Store traces
        df.at[idx, 'ChanA_raw_trc'] = chan_a_trace
        df.at[idx, 'ChanB_raw_trc'] = chan_b_trace
        
        try:
            # Find and store stimulation point
            stim_point = find_stimulation_point(chan_a_trace, chan_b_trace)
            df.at[idx, 'Stim'] = stim_point
            
            # Create cut traces (15 frames before, 85 frames after stimulation)
            start_idx = max(0, stim_point - 15)  # Ensure we don't go below 0
            end_idx = min(len(chan_a_trace), stim_point + 85)  # Ensure we don't exceed trace length
            
            # Store cut traces
            df.at[idx, 'ChanA_cut_trc'] = chan_a_trace[start_idx:end_idx]
            df.at[idx, 'ChanB_cut_trc'] = chan_b_trace[start_idx:end_idx]
            
        except ValueError as e:
            print(f"Warning: {e} for ROI {row['ROI#']} in {row['EXP']}")
            df.at[idx, 'Stim'] = None
            df.at[idx, 'ChanA_cut_trc'] = None
            df.at[idx, 'ChanB_cut_trc'] = None
    
    return df

def save_dataframe(df, parent_directory):
    """
    Save the DataFrame to a pickle file in the parent directory.
    
    Args:
        df (pandas.DataFrame): DataFrame to save
        parent_directory (str): Path to the parent directory
    """
    # Create the output path
    output_path = os.path.join(parent_directory, 'ROI_quant_overview.pkl')
    
    # Save the DataFrame with full trace data as pickle
    df.to_pickle(output_path)
    print(f"\nSaved DataFrame to: {output_path}")
    
    # Also save a readable Excel version for reference (without traces)
    excel_df = df.copy()
    for col in excel_df.columns:
        if 'trc' in col.lower():  # If column contains traces
            excel_df[col] = excel_df[col].apply(lambda x: f"Trace [length: {len(x)}]" if isinstance(x, np.ndarray) else str(x))
    excel_path = os.path.join(parent_directory, 'ROI_quant_overview.xlsx')
    excel_df.to_excel(excel_path, index=False)
    print(f"Also saved readable version to: {excel_path}")

def main(parent_directory):
    """
    Main function to generate the ROI quantification overview PDF.
    
    Args:
        parent_directory (str): Path to the parent directory containing the Excel data
    """
    # Step 1: Import and process data
    print(f"Searching for Excel files in: {parent_directory}")
    excel_files = find_excel_files(parent_directory)
    
    if not excel_files:
        print("No Excel files found in the specified directory.")
        return
    
    print(f"\nFound {len(excel_files)} Excel files:")
    for file in excel_files:
        print(f"- {file}")
    
    # Import data from each Excel file
    for file_path in excel_files:
        df = import_excel_data(file_path)
        if df is not None:
            # Add trace columns
            print("\nExtracting ROI traces...")
            df = add_trace_columns(df, parent_directory)
            
            # Save the updated DataFrame
            save_dataframe(df, parent_directory)
            
            # Visualize the DataFrame
            visualize_dataframe(df)
            break  # Only process the first valid Excel file
    
    # Step 2: Generate visualizations
    # TODO: Implement visualization generation
    
    # Step 3: Create PDF report
    # TODO: Implement PDF report generation

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generate ROI quantification overview PDF')
    parser.add_argument('parent_directory', type=str, help='Path to the parent directory containing the Excel data')
    args = parser.parse_args()
    
    main(args.parent_directory) 