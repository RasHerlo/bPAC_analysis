"""
cAMP-Ca Correlation Analysis

This script analyzes the correlation between cAMP and Ca signals from ROI quantification data.
It loads a pickle file containing ROI data, extracts maximum values from ChanA_dFF (cAMP) and ChanB_dFF (Ca),
and creates a scatter plot of these values.

Dependencies:
- pandas: For data handling
- matplotlib: For data visualization
- numpy: For numerical calculations
- scipy: For statistical calculations
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

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
    For cAMP (ChanA_dFF), only look between frames 15-60.
    Excludes ROI#8 from calculations.
    
    Args:
        df (pd.DataFrame): DataFrame containing ROI data
        
    Returns:
        tuple: Arrays of cAMP and Ca maximum values
    """
    # Filter out ROI#8
    df_filtered = df[df['ROI#'] != 8]
    
    # Extract maximum values from each row in ChanA_dFF and ChanB_dFF
    cAMP_values = df_filtered['ChanA_dFF'].apply(lambda x: np.max(x[15:60]) if isinstance(x, (list, np.ndarray)) else x)
    Ca_values = df_filtered['ChanB_dFF'].apply(lambda x: np.max(x) if isinstance(x, (list, np.ndarray)) else x)
    
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
    
    # Calculate linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(cAMP_values, Ca_values)
    r_squared = r_value ** 2
    
    # Add linear regression line
    plt.plot(cAMP_values, slope * cAMP_values + intercept, color='red', 
             label=f'R² = {r_squared:.3f}, p = {p_value:.3e}')
    
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

def plot_pairwise_traces(df, output_dir):
    """
    Create a scrollable figure showing pairwise traces of cAMP and Ca signals.
    Excludes ROI#8 from visualization.
    
    Args:
        df (pd.DataFrame): DataFrame containing ROI data
        output_dir (str): Directory where the plot should be saved
    """
    # Filter out ROI#8
    df_filtered = df[df['ROI#'] != 8]
    
    # Create a Tkinter window
    root = tk.Tk()
    root.title("cAMP and Ca Traces")
    
    # Create a frame for the canvas and scrollbar
    frame = tk.Frame(root)
    frame.pack(fill=tk.BOTH, expand=True)
    
    # Create a canvas
    canvas = tk.Canvas(frame)
    scrollbar = tk.Scrollbar(frame, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas)
    
    # Configure the canvas
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    
    # Create a window in the canvas
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    
    # Calculate the total height needed for all plots
    n_rows = len(df_filtered)
    plot_height = 2.5  # inches per plot
    total_height = n_rows * plot_height
    
    # Create the figure with appropriate size
    fig = Figure(figsize=(10, total_height))
    
    # Create subplots
    axes = fig.subplots(n_rows, 2)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Plot each pair of traces
    for idx, (_, row) in enumerate(df_filtered.iterrows()):
        # Get the traces
        cAMP_trace = row['ChanA_dFF']
        Ca_trace = row['ChanB_dFF']
        
        # Find maximum values and their indices (for cAMP, only between frames 15-60)
        cAMP_max_idx = np.argmax(cAMP_trace[15:60]) + 15  # Add 15 to get correct index
        Ca_max_idx = np.argmax(Ca_trace)
        
        # Plot cAMP trace
        axes[idx, 0].plot(cAMP_trace, color='blue')
        axes[idx, 0].scatter(cAMP_max_idx, cAMP_trace[cAMP_max_idx], color='red', s=50)
        axes[idx, 0].set_title(f'ROI {row["ROI#"]} - cAMP')
        axes[idx, 0].grid(True, alpha=0.3)
        # Add gray box for frames 15-60
        axes[idx, 0].axvspan(15, 60, color='gray', alpha=0.2)
        
        # Plot Ca trace
        axes[idx, 1].plot(Ca_trace, color='green')
        axes[idx, 1].scatter(Ca_max_idx, Ca_trace[Ca_max_idx], color='red', s=50)
        axes[idx, 1].set_title(f'ROI {row["ROI#"]} - Ca')
        axes[idx, 1].grid(True, alpha=0.3)
        # Add gray box for frames 15-60
        axes[idx, 1].axvspan(15, 60, color='gray', alpha=0.2)
    
    # Add x-axis labels to the bottom row only
    for ax in axes[-1]:
        ax.set_xlabel('Frame')
    
    # Adjust layout
    fig.tight_layout()
    
    # Create the canvas widget
    canvas_widget = FigureCanvasTkAgg(fig, master=scrollable_frame)
    canvas_widget.draw()
    canvas_widget.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    # Configure the scroll region
    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )
    
    # Add mousewheel scrolling
    def _on_mousewheel(event):
        canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    
    canvas.bind_all("<MouseWheel>", _on_mousewheel)
    
    # Save the plot
    output_path = os.path.join(output_dir, 'cAMP_Ca_traces.png')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Traces plot saved to: {output_path}")
    
    # Start the Tkinter event loop
    root.mainloop()

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

    # Plot pairwise traces
    plot_pairwise_traces(df, pickle_dir)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python cAMP_Ca_correlations.py <directory>")
        sys.exit(1)
    
    directory = sys.argv[1]
    main(directory) 