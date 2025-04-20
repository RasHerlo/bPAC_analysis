import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data from pickle file
df = pd.read_pickle('C:/Users/rasmu/OneDrive/Desktop/TEMP LOCAL FILES/ROI_quant_overview.pkl')

# Get the first row's traces
raw_trace_a = df['ChanA_raw_trc'].iloc[0]
raw_trace_b = df['ChanB_raw_trc'].iloc[0]
cut_trace_a = df['ChanA_cut_trc'].iloc[0]
cut_trace_b = df['ChanB_cut_trc'].iloc[0]
stim_point = df['Stim'].iloc[0]

# Create subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Plot raw traces
ax1.plot(raw_trace_a, 'r-', label='Channel A')
ax1.plot(raw_trace_b, 'g-', label='Channel B')
ax1.axvline(x=stim_point, color='k', linestyle='--', label='Stim')
ax1.set_xlabel('Frame')
ax1.set_ylabel('Intensity')
ax1.set_title(f"Raw Traces for ROI #{df['ROI#'].iloc[0]}")
ax1.legend()
ax1.grid(True)

# Plot cut traces
ax2.plot(cut_trace_a, 'r-', label='Channel A')
ax2.plot(cut_trace_b, 'g-', label='Channel B')
ax2.axvline(x=15, color='k', linestyle='--', label='Stim')  # Stim point should be at frame 15
ax2.set_xlabel('Frame')
ax2.set_ylabel('Intensity')
ax2.set_title('Cut Traces (15 frames before, 85 after stim)')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show() 