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
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

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
    
    # Convert all values to strings and right-align numbers
    cell_text = df.applymap(str).values
    
    # Create table
    table = ax.table(cellText=cell_text,
                    colLabels=df.columns,
                    cellLoc='left',
                    loc='center',
                    colColours=['lightblue']*len(df.columns))
    
    # Adjust font size and cell size
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    
    # Make the rows taller
    table.scale(1.2, 2)
    
    # Color alternating rows for better readability
    for i in range(len(df)):
        for j in range(len(df.columns)):
            cell = table[(i+1, j)]
            if i % 2 == 0:
                cell.set_facecolor('#f0f0f0')
    
    plt.title('ROI Database Overview', pad=20, size=14)
    plt.tight_layout()
    plt.show()

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