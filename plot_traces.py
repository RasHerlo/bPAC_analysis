import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data from pickle file
df = pd.read_pickle('C:/Users/rasmu/OneDrive/Desktop/TEMP LOCAL FILES/ROI_quant_overview.pkl')

# Get the first row's traces
trace_a = df['ChanA_raw_trc'].iloc[0]
trace_b = df['ChanB_raw_trc'].iloc[0]

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(trace_a, 'r-', label='Channel A')
plt.plot(trace_b, 'g-', label='Channel B')
plt.xlabel('Frame')
plt.ylabel('Intensity')
plt.title(f"Traces for ROI #{df['ROI#'].iloc[0]}")
plt.legend()
plt.grid(True)
plt.show() 