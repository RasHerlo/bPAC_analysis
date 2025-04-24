"""
Calcium Response Comparison

This script generates a barplot comparing calcium responses across three conditions:
1. bPAC_spont: Spontaneous activity
2. AP_clean: Action potential responses
3. bPAC_AP: bPAC-induced responses

The script extracts maximum z-scored values from Channel B traces for each condition
and plots them as a barplot with error bars representing standard error of the mean.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def extract_max_values(df, condition, range_type='stim', range_start=None, range_end=None):
    """
    Extract maximum values from Channel B z-scored traces based on specified range.
    
    Args:
        df (pandas.DataFrame): DataFrame containing the trace data
        condition (str): Name of the condition
        range_type (str): Either 'stim' for stimulation-based range or 'fixed' for fixed range
        range_start (int): Start of fixed range (only used if range_type='fixed')
        range_end (int): End of fixed range (only used if range_type='fixed')
        
    Returns:
        list: List of maximum values
    """
    max_values = []
    
    for idx, row in df.iterrows():
        if not isinstance(row['ChanB_Z_trc'], np.ndarray) or pd.isna(row['Stim']):
            continue
            
        trace = row['ChanB_Z_trc']
        
        if range_type == 'stim':
            # Use stimulation point + 15 frames
            start_idx = int(row['Stim'])
            end_idx = start_idx + 15
        else:
            # Use fixed range
            start_idx = range_start
            end_idx = range_end
            
        # Ensure indices are within trace bounds
        start_idx = max(0, start_idx)
        end_idx = min(len(trace), end_idx)
        
        # Extract maximum value in the range
        max_val = np.max(trace[start_idx:end_idx])
        max_values.append(max_val)
    
    print(f"Found {len(max_values)} values for {condition}")
    return max_values

def create_barplot(bpac_spont_values, ap_clean_values, bpac_ap_values):
    """
    Create a barplot comparing the three conditions.
    
    Args:
        bpac_spont_values (list): List of maximum values for bPAC_spont condition
        ap_clean_values (list): List of maximum values for AP_clean condition
        bpac_ap_values (list): List of maximum values for bPAC_AP condition
    """
    # Calculate means and standard errors
    means = [
        np.mean(bpac_spont_values),
        np.mean(ap_clean_values),
        np.mean(bpac_ap_values)
    ]
    
    sems = [
        np.std(bpac_spont_values) / np.sqrt(len(bpac_spont_values)),
        np.std(ap_clean_values) / np.sqrt(len(ap_clean_values)),
        np.std(bpac_ap_values) / np.sqrt(len(bpac_ap_values))
    ]
    
    # Create the barplot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Define bar positions and labels
    x = np.arange(3)
    labels = ['bPAC_spont', 'AP_clean', 'bPAC_AP']
    
    # Plot bars
    bars = ax.bar(x, means, yerr=sems, capsize=10, color='green')
    
    # Customize the plot
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Maximum Z-scored Value')
    ax.set_title('Calcium Response Comparison')
    
    # Add individual data points
    for i, values in enumerate([bpac_spont_values, ap_clean_values, bpac_ap_values]):
        x_points = np.random.normal(i, 0.04, size=len(values))
        ax.plot(x_points, values, 'o', color='black', alpha=0.5, markersize=4)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'calcium_comparison.png')
    plt.savefig(output_path, dpi=300)
    print(f"\nPlot saved to: {output_path}")
    
    # Show the plot
    plt.show()

def main():
    """
    Main function to load data and generate the comparison plot.
    """
    # Load bPAC_spont data
    bpac_spont_path = r"C:\Users\rasmu\Desktop\TEMP LOCAL FILES\bPAC_spont\ROI_quant_overview.pkl"
    print(f"\nLoading bPAC_spont data from: {bpac_spont_path}")
    bpac_spont_df = pd.read_pickle(bpac_spont_path)
    
    # Load bPAC_APs data
    bpac_aps_path = r"C:\Users\rasmu\Desktop\TEMP LOCAL FILES\bPAC_APs\ROI_quant_overview.pkl"
    print(f"Loading bPAC_APs data from: {bpac_aps_path}")
    bpac_aps_df = pd.read_pickle(bpac_aps_path)
    
    # Extract values for each condition
    print("\nExtracting maximum values...")
    bpac_spont_values = extract_max_values(bpac_spont_df, 'bPAC_spont', 'stim')
    ap_clean_values = extract_max_values(bpac_aps_df, 'AP_clean', 'fixed', 34, 44)
    bpac_ap_values = extract_max_values(bpac_aps_df, 'bPAC_AP', 'stim')
    
    # Create and save the barplot
    create_barplot(bpac_spont_values, ap_clean_values, bpac_ap_values)

if __name__ == "__main__":
    main() 