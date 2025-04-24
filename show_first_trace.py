import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the pickle file
df = pd.read_pickle(r"C:\Users\rasmu\Desktop\TEMP LOCAL FILES\bPAC_spont\ROI_quant_overview.pkl")

# Get the first trace
first_trace = df.iloc[0]['ChanB_Z_trc']

# Print the trace data
print("First trace data:")
print(first_trace)

# Plot the trace
plt.figure(figsize=(10, 4))
plt.plot(first_trace)
plt.title('First ChanB_Z_trc')
plt.xlabel('Frame')
plt.ylabel('Z-score')
plt.grid(True)
plt.show() 