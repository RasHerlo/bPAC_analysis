import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
df = pd.read_pickle(r"C:\Users\rasmu\OneDrive\Desktop\TEMP LOCAL FILES\ROI_quant_overview.pkl")

# Get the first row's traces
row = df.iloc[0]
chan_a_raw = row['ChanA_raw_trc']
chan_a_comp = row['ChanA_comp_trc']
chan_b_raw = row['ChanB_raw_trc']
chan_a_cut = row['ChanA_cut_trc']
chan_b_cut = row['ChanB_cut_trc']
chan_a_norm = row['ChanA_norm_trc']
chan_b_norm = row['ChanB_norm_trc']
chan_a_z = row['ChanA_Z_trc']
chan_b_z = row['ChanB_Z_trc']
stim_point = row['Stim']

# Create figure with four subplots
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 16))

# Plot raw traces
ax1.plot(chan_a_raw, 'r-', label='Channel A (raw)')
ax1.plot(chan_a_comp, 'r--', label='Channel A (compensated)')
ax1.plot(chan_b_raw, 'g-', label='Channel B')
if stim_point is not None:
    ax1.axvline(x=stim_point, color='k', linestyle='--', label='Stimulation')
ax1.set_xlabel('Frame')
ax1.set_ylabel('Intensity')
ax1.set_title('Raw Traces')
ax1.legend()
ax1.grid(True)

# Plot cut traces
ax2.plot(chan_a_cut, 'r-', label='Channel A')
ax2.plot(chan_b_cut, 'g-', label='Channel B')
if stim_point is not None:
    ax2.axvline(x=15, color='k', linestyle='--', label='Stimulation')
ax2.set_xlabel('Frame')
ax2.set_ylabel('Intensity')
ax2.set_title('Cut Traces (15 frames before, 85 frames after stimulation)')
ax2.legend()
ax2.grid(True)

# Plot normalized traces
ax3.plot(chan_a_norm, 'r-', label='Channel A')
ax3.plot(chan_b_norm, 'g-', label='Channel B')
if stim_point is not None:
    ax3.axvline(x=15, color='k', linestyle='--', label='Stimulation')
ax3.set_xlabel('Frame')
ax3.set_ylabel('Normalized Intensity')
ax3.set_title('Normalized Traces')
ax3.set_ylim(-0.25, 1.25)
ax3.legend()
ax3.grid(True)

# Plot z-scored traces
ax4.plot(chan_a_z, 'r-', label='Channel A')
ax4.plot(chan_b_z, 'g-', label='Channel B')
if stim_point is not None:
    ax4.axvline(x=15, color='k', linestyle='--', label='Stimulation')
ax4.set_xlabel('Frame')
ax4.set_ylabel('Z-score')
ax4.set_title('Z-scored Traces')
ax4.set_ylim(-1, None)
ax4.legend()
ax4.grid(True)

plt.tight_layout()
plt.show() 