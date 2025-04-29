"""
cAMP-Ca Correlation Analysis

This script analyzes the correlation between cAMP and Ca signals from ROI quantification data.
It loads a pickle file containing ROI data, extracts maximum values from ChanA_dFF (cAMP) and ChanB_dFF (Ca),
and creates a scatter plot of these values.

Dependencies:
- pandas: For data handling
- matplotlib: For data visualization
- numpy: For numerical calculations
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def find_pickle_file(directory):
    """
    Search for the ROI quantification pickle file in the given directory.
    
    Args:
        directory (str): Path to the directory to search
        
    Returns:
        str: Path to the pickle file if found, None otherwise
    """
    for root, _, files in os.walk(directory):
        for file in files:
            if file == 'ROI_quant_overview.pkl':
                return os.path.join(root, file)
    return None

def extract_max_values(df):
    """
    Extract maximum values from ChanA_dFF and ChanB_dFF columns.
    
    Args:
        df (pd.DataFrame): DataFrame containing ROI data
        
    Returns:
        tuple: Arrays of cAMP and Ca maximum values
    """
    # Extract maximum values from each row in ChanA_dFF and ChanB_dFF
    cAMP_values = df['ChanA_dFF'].apply(lambda x: np.max(x) if isinstance(x, (list, np.ndarray)) else x)
    Ca_values = df['ChanB_dFF'].apply(lambda x: np.max(x) if isinstance(x, (list, np.ndarray)) else x)
    
    return cAMP_values, Ca_values

def plot_correlation(cAMP_values, Ca_values, output_dir):
    """
    Create a scatter plot of cAMP vs Ca values and save it.
    
    Args:
        cAMP_values (np.ndarray): Array of cAMP maximum values
        Ca_values (np.ndarray): Array of Ca maximum values
        output_dir (str): Directory where the plot should be saved
    """
    plt.figure(figsize=(8, 6))
    
    # Create scatter plot
    plt.scatter(cAMP_values, Ca_values, alpha=0.6)
    
    # Add linear regression line
    slope, intercept = np.polyfit(cAMP_values, Ca_values, 1)
    plt.plot(cAMP_values, slope * cAMP_values + intercept, color='red', 
             label=f'y = {slope:.2f}x + {intercept:.2f}')
    
    # Add labels and title
    plt.xlabel('cAMP (ΔF/F)')
    plt.ylabel('Ca (ΔF/F)')
    plt.title('bPAC-induction')
    
    # Set axis limits
    plt.xlim(left=1)
    plt.ylim(bottom=1)
    
    # Add grid and legend
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save the plot
    output_path = os.path.join(output_dir, 'cAMP_Ca_correlation.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    # Show the plot
    plt.tight_layout()
    plt.show()

def main(directory):
    """
    Main function to run the analysis.
    
    Args:
        directory (str): Path to the directory containing the pickle file
    """
    # Find the pickle file
    pickle_path = find_pickle_file(directory)
    if not pickle_path:
        print(f"Error: Could not find ROI_quant_overview.pkl in {directory}")
        return
    
    # Get the directory containing the pickle file
    pickle_dir = os.path.dirname(pickle_path)
    
    # Load the data
    try:
        df = pd.read_pickle(pickle_path)
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        return
    
    # Extract maximum values
    cAMP_values, Ca_values = extract_max_values(df)
    
    # Plot the correlation
    plot_correlation(cAMP_values, Ca_values, pickle_dir)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python cAMP_Ca_correlations.py <directory>")
        sys.exit(1)
    
    directory = sys.argv[1]
    main(directory) 