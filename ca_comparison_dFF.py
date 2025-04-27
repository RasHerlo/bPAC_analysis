"""
Calcium Response Comparison

This script generates a barplot comparing calcium responses across three conditions:
1. bPAC_spont: Spontaneous activity
2. AP_clean: Action potential responses
3. bPAC_AP: bPAC-induced responses

The script extracts maximum dF/F values from Channel B traces for each condition
and plots them as a barplot with error bars representing standard error of the mean.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def extract_max_values(df, condition, trace_type='ChanB_dFF'):
    """Extract maximum values from specified trace type based on condition"""
    max_values = []
    for idx, row in df.iterrows():
        if isinstance(row[trace_type], np.ndarray):
            if condition == 'bPAC_spont':
                # Use range from stim to stim+15 for bPAC_spont
                if not pd.isna(row['Stim']):
                    start_idx = int(row['Stim'])
                    end_idx = start_idx + 15
                else:
                    continue
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
        trace_type (str): Type of trace used ('ChanB_dFF')
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
    ax.set_ylabel('Maximum dF/F Value')
    ax.set_title('Calcium Response Comparison (dF/F)')
    
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

def create_trace_overview_bpac_spont(df, output_path):
    """
    Create an overview of all bPAC_spont traces with their maximum values and search ranges.
    
    Args:
        df (DataFrame): DataFrame containing the bPAC_spont data
        output_path (str): Path to save the plot
    """
    # Filter out traces that are not numpy arrays and have a valid Stim value
    valid_traces = [idx for idx, row in df.iterrows() 
                   if isinstance(row['ChanB_dFF'], np.ndarray) and not pd.isna(row['Stim'])]
    
    # Calculate number of rows needed (3 columns)
    n_traces = len(valid_traces)
    n_rows = (n_traces + 2) // 3  # Ceiling division
    
    # Create figure with appropriate size
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5*n_rows))
    axes = axes.flatten()
    
    for i, idx in enumerate(valid_traces):
        trace = df.iloc[idx]['ChanB_dFF']
        
        # Plot trace
        axes[i].plot(trace, 'k-', linewidth=1)
        
        # Define search range based on Stim
        start_idx = int(df.iloc[idx]['Stim'])
        end_idx = start_idx + 15
        
        # Find maximum in search range
        search_range = trace[start_idx:end_idx]
        max_idx = np.argmax(search_range) + start_idx
        max_val = trace[max_idx]
        
        # Highlight search range
        axes[i].axvspan(start_idx, end_idx, color='gray', alpha=0.2)
        
        # Mark maximum value
        axes[i].plot(max_idx, max_val, 'ro', markersize=4)
        
        # Add text with maximum value
        axes[i].text(max_idx + 1, max_val, f'{max_val:.2f}', 
                    fontsize=8, verticalalignment='center')
        
        # Set title and labels
        mouse = df.iloc[idx]['MOUSE']
        roi = df.iloc[idx]['ROI#']
        axes[i].set_title(f'{mouse} - ROI#{roi}')
        axes[i].set_xlabel('Frame')
        axes[i].set_ylabel('dF/F Value')
        axes[i].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(len(valid_traces), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, format='pdf')
    print(f"\nTrace overview for bPAC_spont saved to: {output_path}")

def create_trace_overview_ap_conditions(df, output_path):
    """
    Create an overview of all AP traces with both AP_clean and bPAC_AP ranges and maximum values.
    
    Args:
        df (DataFrame): DataFrame containing the AP_clean and bPAC_AP data
        output_path (str): Path to save the plot
    """
    # Get valid traces (those that have both conditions)
    valid_traces = [idx for idx, row in df.iterrows() 
                   if isinstance(row['ChanB_dFF'], np.ndarray) and not pd.isna(row['Stim'])]
    
    n_traces = len(valid_traces)
    n_rows = (n_traces + 2) // 3  # Ceiling division
    
    # Create figure with appropriate size
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5*n_rows))
    axes = axes.flatten()
    
    for i, idx in enumerate(valid_traces):
        trace = df.iloc[idx]['ChanB_dFF']
        
        # Plot trace
        axes[i].plot(trace, 'k-', linewidth=1)
        
        # AP_clean range: 34-44
        ap_clean_start = 34
        ap_clean_end = 44
        ap_clean_range = trace[ap_clean_start:ap_clean_end]
        ap_clean_max_idx = np.argmax(ap_clean_range) + ap_clean_start
        ap_clean_max_val = trace[ap_clean_max_idx]
        
        # bPAC_AP range: stim to stim+15
        bpac_ap_start = int(df.iloc[idx]['Stim'])
        bpac_ap_end = bpac_ap_start + 15
        bpac_ap_range = trace[bpac_ap_start:bpac_ap_end]
        bpac_ap_max_idx = np.argmax(bpac_ap_range) + bpac_ap_start
        bpac_ap_max_val = trace[bpac_ap_max_idx]
        
        # Highlight search ranges with different colors
        axes[i].axvspan(ap_clean_start, ap_clean_end, color='gray', alpha=0.2, label='AP_clean range')
        axes[i].axvspan(bpac_ap_start, bpac_ap_end, color='blue', alpha=0.1, label='bPAC_AP range')
        
        # Mark maximum values with different colors
        axes[i].plot(ap_clean_max_idx, ap_clean_max_val, 'ro', markersize=4, label='AP_clean max')
        axes[i].plot(bpac_ap_max_idx, bpac_ap_max_val, 'bo', markersize=4, label='bPAC_AP max')
        
        # Add text with maximum values
        axes[i].text(ap_clean_max_idx + 1, ap_clean_max_val, f'{ap_clean_max_val:.2f}', 
                    fontsize=8, verticalalignment='center', color='red')
        axes[i].text(bpac_ap_max_idx + 1, bpac_ap_max_val, f'{bpac_ap_max_val:.2f}', 
                    fontsize=8, verticalalignment='center', color='blue')
        
        # Set title and labels
        mouse = df.iloc[idx]['MOUSE']
        roi = df.iloc[idx]['ROI#']
        axes[i].set_title(f'{mouse} - ROI#{roi}')
        axes[i].set_xlabel('Frame')
        axes[i].set_ylabel('dF/F Value')
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()
    
    # Hide unused subplots
    for i in range(len(valid_traces), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, format='pdf')
    print(f"\nTrace overview for AP conditions saved to: {output_path}")

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
    
    # Create trace overview plots
    create_trace_overview_bpac_spont(bpac_spont_df, 
                                   os.path.join(output_dir, 'bpac_spont_traces_dFF.pdf'))
    create_trace_overview_ap_conditions(bpac_aps_df, 
                                      os.path.join(output_dir, 'ap_conditions_traces_dFF.pdf'))
    
    # Extract values for bar plot using dF/F traces
    print("\nProcessing dF/F traces for bar plot...")
    bpac_spont_values = extract_max_values(bpac_spont_df, 'bPAC_spont', 'ChanB_dFF')
    ap_clean_values = extract_max_values(bpac_aps_df, 'AP_clean', 'ChanB_dFF')
    bpac_ap_values = extract_max_values(bpac_aps_df, 'bPAC_AP', 'ChanB_dFF')
    
    # Create and save the barplot
    output_path = os.path.join(output_dir, 'calcium_comparison_dFF.pdf')
    create_barplot(bpac_spont_values, ap_clean_values, bpac_ap_values, 'ChanB_dFF', output_path)

if __name__ == "__main__":
    main() 