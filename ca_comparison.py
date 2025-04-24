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

def extract_max_values(df, condition, trace_type='ChanB_Z_trc'):
    """Extract maximum values from specified trace type based on condition"""
    max_values = []
    for idx, row in df.iterrows():
        if isinstance(row[trace_type], np.ndarray):
            if condition == 'bPAC_spont':
                # Use fixed range 15-25 for bPAC_spont
                start_idx = 15
                end_idx = 25
            elif condition == 'AP_clean':
                # Use fixed range 34-44 for AP_clean
                start_idx = 34
                end_idx = 44
            elif condition == 'bPAC_AP':
                # Use range from stim to stim+15 for bPAC_AP
                if not pd.isna(row['Stim']):
                    start_idx = int(row['Stim'])
                    end_idx = start_idx + 15
                else:
                    continue
            
            # Ensure indices are within bounds
            start_idx = max(0, start_idx)
            end_idx = min(len(row[trace_type]), end_idx)
            
            # Extract and store maximum value
            max_val = np.max(row[trace_type][start_idx:end_idx])
            max_values.append(max_val)
    
    return np.array(max_values)

def create_barplot(bpac_spont_values, ap_clean_values, bpac_ap_values, trace_type, output_path):
    """
    Create a barplot comparing the three conditions.
    
    Args:
        bpac_spont_values (list): List of maximum values for bPAC_spont condition
        ap_clean_values (list): List of maximum values for AP_clean condition
        bpac_ap_values (list): List of maximum values for bPAC_AP condition
        trace_type (str): Type of trace used ('ChanB_Z_trc' or 'ChanB_norm_trc')
        output_path (str): Path to save the plot
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
    ylabel = 'Maximum Z-scored Value' if trace_type == 'ChanB_Z_trc' else 'Maximum Normalized Value'
    ax.set_ylabel(ylabel)
    ax.set_title(f'Calcium Response Comparison ({trace_type})')
    
    # Add individual data points
    for i, values in enumerate([bpac_spont_values, ap_clean_values, bpac_ap_values]):
        x_points = np.random.normal(i, 0.04, size=len(values))
        ax.plot(x_points, values, 'o', color='black', alpha=0.5, markersize=4)
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path, dpi=300, format='pdf')
    print(f"\nPlot saved to: {output_path}")
    
    # Show the plot
    plt.show()

def main():
    """
    Main function to load data and generate the comparison plots.
    """
    # Load bPAC_spont data
    bpac_spont_path = r"C:\Users\rasmu\Desktop\TEMP LOCAL FILES\bPAC_spont\ROI_quant_overview.pkl"
    print(f"\nLoading bPAC_spont data from: {bpac_spont_path}")
    bpac_spont_df = pd.read_pickle(bpac_spont_path)
    
    # Load bPAC_APs data
    bpac_aps_path = r"C:\Users\rasmu\Desktop\TEMP LOCAL FILES\bPAC_APs\ROI_quant_overview.pkl"
    print(f"Loading bPAC_APs data from: {bpac_aps_path}")
    bpac_aps_df = pd.read_pickle(bpac_aps_path)
    
    # Create output directory based on first pickle file location
    output_dir = os.path.dirname(bpac_spont_path)
    
    # Process both trace types
    for trace_type in ['ChanB_Z_trc', 'ChanB_norm_trc']:
        print(f"\nProcessing {trace_type}...")
        
        # Extract values for each condition
        bpac_spont_values = extract_max_values(bpac_spont_df, 'bPAC_spont', trace_type)
        ap_clean_values = extract_max_values(bpac_aps_df, 'AP_clean', trace_type)
        bpac_ap_values = extract_max_values(bpac_aps_df, 'bPAC_AP', trace_type)
        
        # Create output path
        output_path = os.path.join(output_dir, f'calcium_comparison_{trace_type}.pdf')
        
        # Create and save the barplot
        create_barplot(bpac_spont_values, ap_clean_values, bpac_ap_values, trace_type, output_path)

if __name__ == "__main__":
    main() 