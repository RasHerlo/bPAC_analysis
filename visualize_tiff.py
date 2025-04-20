import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.filters import threshold_otsu
import os

# Load the TIFF stack
file_path = r"C:\Users\rasmu\OneDrive\Desktop\TEMP LOCAL FILES\FXX\980nm_PreStim60s_pls15_fps6_w05avg\STKS\ChanB_stk.tif"
stack = io.imread(file_path)

# Calculate the average projection
avg_projection = np.mean(stack, axis=0)

# Create figure with three subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

# Plot 1: Average projection
im1 = ax1.imshow(avg_projection, cmap='viridis')
ax1.set_title('Average Projection')
plt.colorbar(im1, ax=ax1)

# Plot 2: Enhanced contrast
# Calculate 1st and 99th percentiles for better contrast
p1, p99 = np.percentile(avg_projection, (1, 99))
im2 = ax2.imshow(avg_projection, cmap='viridis', vmin=p1, vmax=p99)
ax2.set_title('Enhanced Contrast')
plt.colorbar(im2, ax=ax2)

# Plot 3: Otsu thresholding
# Apply Otsu thresholding
thresh = threshold_otsu(avg_projection)
binary = avg_projection > thresh
im3 = ax3.imshow(binary, cmap='viridis')
ax3.set_title('Otsu Thresholding')
plt.colorbar(im3, ax=ax3)

# Adjust layout and show
plt.tight_layout()
plt.show() 