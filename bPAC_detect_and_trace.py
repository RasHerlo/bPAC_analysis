import os
import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread
from skimage import exposure
import argparse
from matplotlib.widgets import Button
from matplotlib.patches import Polygon

def load_tif_stack(tif_path):
    """
    Load a tif stack from the specified path.
    
    Parameters:
    -----------
    tif_path : str
        Path to the tif stack file
    
    Returns:
    --------
    numpy.ndarray
        The loaded tif stack
    """
    if not os.path.isfile(tif_path):
        raise FileNotFoundError(f"Tif file not found at: {tif_path}")
    
    print(f"Loading tif stack from: {tif_path}")
    stack = imread(tif_path)
    print(f"Loaded stack with shape: {stack.shape}")
    return stack

def create_heatmaps(stack, z_range1, z_range2):
    """
    Create average and ratio heatmaps from the tif stack.
    
    Parameters:
    -----------
    stack : numpy.ndarray
        The loaded tif stack
    z_range1 : tuple
        (start, end) for the first z-range
    z_range2 : tuple
        (start, end) for the second z-range
    
    Returns:
    --------
    tuple
        (average_heatmap, ratio_heatmap)
    """
    # Create average heatmap (average across all z)
    average_heatmap = np.mean(stack, axis=0)
    
    # Create ratio heatmap
    avg1 = np.mean(stack[z_range1[0]:z_range1[1]], axis=0)
    avg2 = np.mean(stack[z_range2[0]:z_range2[1]], axis=0)
    ratio_heatmap = avg1 / avg2
    
    # Handle division by zero
    ratio_heatmap[~np.isfinite(ratio_heatmap)] = 0
    
    return average_heatmap, ratio_heatmap

def stretch_heatmaps(average_heatmap, ratio_heatmap):
    """
    Stretch the heatmaps using skimage exposure.
    
    Parameters:
    -----------
    average_heatmap : numpy.ndarray
        The average heatmap
    ratio_heatmap : numpy.ndarray
        The ratio heatmap
    
    Returns:
    --------
    tuple
        (stretched_average, stretched_ratio)
    """
    # Stretch the heatmaps
    p2_average, p98_average = np.percentile(average_heatmap, (2, 98))
    p2_ratio, p98_ratio = np.percentile(ratio_heatmap, (2, 98))
    
    stretched_average = exposure.rescale_intensity(average_heatmap, 
                                                 in_range=(p2_average, p98_average))
    stretched_ratio = exposure.rescale_intensity(ratio_heatmap, 
                                               in_range=(p2_ratio, p98_ratio))
    
    return stretched_average, stretched_ratio

def get_top_pixels(ratio_heatmap, n_pixels=1000):
    """
    Get the coordinates of the top n pixels with highest ratio values.
    
    Parameters:
    -----------
    ratio_heatmap : numpy.ndarray
        The ratio heatmap
    n_pixels : int
        Number of top pixels to select
    
    Returns:
    --------
    tuple
        (y_coords, x_coords) of the top n pixels
    """
    # Flatten the ratio heatmap and get indices of top n pixels
    flat_indices = np.argsort(ratio_heatmap.flatten())[-n_pixels:]
    
    # Convert flat indices to 2D coordinates
    y_coords, x_coords = np.unravel_index(flat_indices, ratio_heatmap.shape)
    
    return y_coords, x_coords

def extract_and_normalize_traces(stack, y_coords, x_coords):
    """
    Extract and normalize traces for the specified coordinates.
    
    Parameters:
    -----------
    stack : numpy.ndarray
        The loaded tif stack
    y_coords : numpy.ndarray
        Y coordinates of pixels
    x_coords : numpy.ndarray
        X coordinates of pixels
    
    Returns:
    --------
    numpy.ndarray
        Normalized traces for the specified pixels
    """
    # Extract traces
    traces = stack[:, y_coords, x_coords]
    
    # Normalize each trace to [0, 1]
    trace_mins = np.min(traces, axis=0)
    trace_maxs = np.max(traces, axis=0)
    normalized_traces = (traces - trace_mins) / (trace_maxs - trace_mins)
    
    return normalized_traces

def create_mask_image(shape, y_coords, x_coords):
    """
    Create a binary mask image for the top pixels.
    
    Parameters:
    -----------
    shape : tuple
        Shape of the output mask image
    y_coords : numpy.ndarray
        Y coordinates of pixels
    x_coords : numpy.ndarray
        X coordinates of pixels
    
    Returns:
    --------
    numpy.ndarray
        Binary mask image
    """
    mask = np.zeros(shape, dtype=np.uint8)
    mask[y_coords, x_coords] = 1
    return mask

def plot_results(stretched_average, stretched_ratio, normalized_traces, mask_image, y_coords, x_coords):
    """
    Create a figure with four subplots showing the results.
    
    Parameters:
    -----------
    stretched_average : numpy.ndarray
        The stretched average heatmap
    stretched_ratio : numpy.ndarray
        The stretched ratio heatmap
    normalized_traces : numpy.ndarray
        Normalized traces for top pixels
    mask_image : numpy.ndarray
        Binary mask image for top pixels
    y_coords : numpy.ndarray
        Y coordinates of top pixels
    x_coords : numpy.ndarray
        X coordinates of top pixels
    """
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(2, 2)
    
    # Create subplots in new arrangement
    ax1 = fig.add_subplot(gs[0, 0])  # Ratio heatmap
    ax2 = fig.add_subplot(gs[0, 1])  # Mask
    ax3 = fig.add_subplot(gs[1, 0])  # Traces
    ax4 = fig.add_subplot(gs[1, 1])  # Average heatmap with red dots
    
    # Plot ratio heatmap
    im1 = ax1.imshow(stretched_ratio, cmap='viridis')
    ax1.set_title('Ratio Heatmap')
    plt.colorbar(im1, ax=ax1, label='Ratio')
    
    # Plot mask image (inverted colors)
    im2 = ax2.imshow(1 - mask_image, cmap='gray')
    ax2.set_title('Top 1000 Pixels Mask')
    
    # Plot normalized traces (transposed)
    im3 = ax3.imshow(normalized_traces.T, aspect='auto', cmap='viridis')
    ax3.set_xlabel('Time (frames)')
    ax3.set_ylabel('Pixel Index')
    ax3.set_title('Normalized Traces (Top 1000)')
    plt.colorbar(im3, ax=ax3, label='Normalized Intensity')
    
    # Plot average heatmap with red dots
    im4 = ax4.imshow(stretched_average, cmap='viridis')
    ax4.set_title('Average Heatmap with Top 1000 Pixels')
    plt.colorbar(im4, ax=ax4, label='Intensity')
    
    # Add red dots for top pixels
    ax4.scatter(x_coords, y_coords, c='red', s=1, alpha=0.5)
    
    plt.tight_layout()
    plt.show()

def extract_traces_in_polygon(stack, vertices):
    """
    Extract traces for all pixels within a polygon.
    
    Parameters:
    -----------
    stack : numpy.ndarray
        The image stack
    vertices : list of [x, y] coordinates
        The vertices of the polygon
        
    Returns:
    --------
    numpy.ndarray
        Array of traces for pixels within the polygon
    """
    from matplotlib.path import Path
    import numpy as np
    
    # Create a grid of all pixel coordinates
    y, x = np.mgrid[0:stack.shape[1], 0:stack.shape[2]]
    points = np.vstack((x.flatten(), y.flatten())).T
    
    # Create polygon path
    polygon = Path(vertices)
    
    # Find points inside polygon
    mask = polygon.contains_points(points)
    mask = mask.reshape(stack.shape[1], stack.shape[2])
    
    # Get coordinates of points inside polygon
    y_coords, x_coords = np.where(mask)
    
    # Extract traces
    traces = []
    for y, x in zip(y_coords, x_coords):
        trace = stack[:, y, x]
        traces.append(trace)
    
    return np.array(traces)

class PolygonDrawer:
    def __init__(self, ax, image, stack, trace_ax, sum_ax):
        self.ax = ax
        self.image = image
        self.stack = stack
        self.trace_ax = trace_ax
        self.sum_ax = sum_ax
        self.polygon = None
        self.vertices = []
        self.polygon_patch = None
        self.lines = []  # Store all line segments
        self.vertex_markers = []  # Store vertex markers
        self.dragged_vertex = None
        self.drag_index = None
        self.is_finished = False
        self.trace_line = None
        
        # Connect events
        self.cidpress = ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cidrelease = ax.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmotion = ax.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)
        
        # Add clear button
        self.clear_button_ax = plt.axes([0.81, 0.05, 0.1, 0.04])
        self.clear_button = Button(self.clear_button_ax, 'Clear')
        self.clear_button.on_clicked(self.clear_polygon)
        
        # Add finish button
        self.finish_button_ax = plt.axes([0.7, 0.05, 0.1, 0.04])
        self.finish_button = Button(self.finish_button_ax, 'Finish')
        self.finish_button.on_clicked(self.finish_polygon)
        
        self.is_drawing = False
        self.temp_line = None
        
    def add_vertex_marker(self, x, y):
        """Add a draggable vertex marker"""
        marker, = self.ax.plot(x, y, 'ro', markersize=8, picker=5)
        self.vertex_markers.append(marker)
        return marker
        
    def update_traces(self):
        """Update the trace plot with current polygon"""
        if len(self.vertices) < 3 or not self.is_finished:
            return
            
        # Clear previous traces
        self.trace_ax.clear()
        self.sum_ax.clear()
        
        # Extract and plot traces
        traces = extract_traces_in_polygon(self.stack, self.vertices)
        if len(traces) > 0:
            # Normalize traces
            traces = (traces - traces.mean(axis=1, keepdims=True)) / traces.std(axis=1, keepdims=True)
            
            # Plot trace matrix
            self.trace_ax.imshow(traces, aspect='auto', cmap='viridis')
            self.trace_ax.set_title(f'Traces within polygon ({len(traces)} pixels)')
            self.trace_ax.set_xlabel('Frame')
            self.trace_ax.set_ylabel('Pixel')
            
            # Plot sum of traces
            sum_trace = np.sum(traces, axis=0)
            self.sum_ax.plot(sum_trace, 'r-', linewidth=1)
            self.sum_ax.set_title('Sum of all traces')
            self.sum_ax.set_xlabel('Frame')
            self.sum_ax.set_ylabel('Sum intensity')
            self.sum_ax.grid(True)
        else:
            self.trace_ax.text(0.5, 0.5, 'No pixels in polygon', 
                             horizontalalignment='center',
                             verticalalignment='center',
                             transform=self.trace_ax.transAxes)
            self.sum_ax.text(0.5, 0.5, 'No pixels in polygon', 
                           horizontalalignment='center',
                           verticalalignment='center',
                           transform=self.sum_ax.transAxes)
        
        self.trace_ax.figure.canvas.draw_idle()
        self.sum_ax.figure.canvas.draw_idle()
        
    def update_polygon(self):
        """Update the polygon and its lines based on current vertices"""
        # Clear existing lines
        for line in self.lines:
            line.remove()
        self.lines = []
        
        # Clear existing vertex markers
        for marker in self.vertex_markers:
            marker.remove()
        self.vertex_markers = []
        
        # Clear existing polygon patch
        if self.polygon_patch:
            self.polygon_patch.remove()
            self.polygon_patch = None
        
        # Draw new lines and markers
        if len(self.vertices) > 1:
            # Create closed polygon vertices
            closed_vertices = self.vertices + [self.vertices[0]] if self.is_finished else self.vertices
            
            # Draw polygon patch if finished
            if self.is_finished:
                self.polygon_patch = Polygon(closed_vertices, fill=False, color='red')
                self.ax.add_patch(self.polygon_patch)
            
            # Add vertex markers and line segments
            for i in range(len(self.vertices)):
                # Add vertex marker
                self.add_vertex_marker(self.vertices[i][0], self.vertices[i][1])
                
                # Add line segment
                if i < len(self.vertices) - 1:
                    line, = self.ax.plot([self.vertices[i][0], self.vertices[i+1][0]],
                                       [self.vertices[i][1], self.vertices[i+1][1]], 'r-')
                    self.lines.append(line)
            
            # Add closing line if finished
            if self.is_finished:
                line, = self.ax.plot([self.vertices[-1][0], self.vertices[0][0]],
                                   [self.vertices[-1][1], self.vertices[0][1]], 'r-')
                self.lines.append(line)
        
        # Update traces if polygon is finished
        if self.is_finished:
            self.update_traces()
            
        self.ax.figure.canvas.draw_idle()
        
    def on_press(self, event):
        if event.inaxes != self.ax:
            return
            
        # Check if we're clicking on a vertex marker
        if event.button == 1:  # Left click
            for i, marker in enumerate(self.vertex_markers):
                if marker.contains(event)[0]:
                    self.dragged_vertex = marker
                    self.drag_index = i
                    return
                    
            # If not clicking on a vertex, add new vertex
            self.is_drawing = True
            self.vertices.append([event.xdata, event.ydata])
            self.update_polygon()
        
    def on_motion(self, event):
        if event.inaxes != self.ax:
            return
            
        if self.dragged_vertex is not None:
            # Update vertex position
            self.vertices[self.drag_index] = [event.xdata, event.ydata]
            self.update_polygon()
        elif self.is_drawing and len(self.vertices) > 0:
            # Show temporary line while drawing
            if self.temp_line:
                self.temp_line.remove()
            self.temp_line, = self.ax.plot([self.vertices[-1][0], event.xdata],
                                         [self.vertices[-1][1], event.ydata], 'r-')
            self.ax.figure.canvas.draw_idle()
            
    def on_release(self, event):
        if event.inaxes != self.ax:
            return
        self.is_drawing = False
        self.dragged_vertex = None
        self.drag_index = None
        
    def clear_polygon(self, event):
        self.vertices = []
        self.is_finished = False
        if self.polygon_patch:
            self.polygon_patch.remove()
            self.polygon_patch = None
        if self.temp_line:
            self.temp_line.remove()
            self.temp_line = None
        for line in self.lines:
            line.remove()
        self.lines = []
        for marker in self.vertex_markers:
            marker.remove()
        self.vertex_markers = []
        self.ax.figure.canvas.draw_idle()
        
    def finish_polygon(self, event):
        if len(self.vertices) < 3:
            return
        self.is_finished = True
        self.update_polygon()

def create_interactive_window(stretched_average, y_coords, x_coords, stack):
    """
    Create an interactive window for polygon drawing on the average heatmap.
    
    Parameters:
    -----------
    stretched_average : numpy.ndarray
        The stretched average heatmap
    y_coords : numpy.ndarray
        Y coordinates of top pixels
    x_coords : numpy.ndarray
        X coordinates of top pixels
    stack : numpy.ndarray
        The image stack for extracting traces
    """
    fig = plt.figure(figsize=(15, 10))
    
    # Create subplots
    gs = plt.GridSpec(2, 2, figure=fig)
    ax1 = fig.add_subplot(gs[:, 0])  # Left side: polygon
    ax2 = fig.add_subplot(gs[0, 1])  # Top right: traces
    ax3 = fig.add_subplot(gs[1, 1])  # Bottom right: sum of traces
    
    # Plot heatmap and dots
    ax1.imshow(stretched_average, cmap='viridis')
    ax1.scatter(x_coords, y_coords, c='red', s=1, alpha=0.5)
    ax1.set_title('Draw polygon by clicking. Press "Finish" to close polygon.')
    
    # Initialize polygon drawer
    drawer = PolygonDrawer(ax1, stretched_average, stack, ax2, ax3)
    plt.show()

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process tif stack and create heatmaps')
    parser.add_argument('tif_path', type=str, help='Path to the tif stack file')
    parser.add_argument('--z1_start', type=int, required=True, help='Start of first z-range')
    parser.add_argument('--z1_end', type=int, required=True, help='End of first z-range')
    parser.add_argument('--z2_start', type=int, required=True, help='Start of second z-range')
    parser.add_argument('--z2_end', type=int, required=True, help='End of second z-range')
    
    args = parser.parse_args()
    
    # Load tif stack
    stack = load_tif_stack(args.tif_path)
    
    # Validate z-ranges
    if not (0 <= args.z1_start < args.z1_end <= stack.shape[0] and 
            0 <= args.z2_start < args.z2_end <= stack.shape[0]):
        raise ValueError("Invalid z-ranges. Must be within stack dimensions.")
    
    # Create heatmaps
    average_heatmap, ratio_heatmap = create_heatmaps(
        stack, 
        (args.z1_start, args.z1_end), 
        (args.z2_start, args.z2_end)
    )
    
    # Stretch heatmaps
    stretched_average, stretched_ratio = stretch_heatmaps(average_heatmap, ratio_heatmap)
    
    # Get top 1000 pixels
    y_coords, x_coords = get_top_pixels(ratio_heatmap)
    
    # Extract and normalize traces
    normalized_traces = extract_and_normalize_traces(stack, y_coords, x_coords)
    
    # Create mask image
    mask_image = create_mask_image(ratio_heatmap.shape, y_coords, x_coords)
    
    # Plot results
    plot_results(stretched_average, stretched_ratio, normalized_traces, mask_image, y_coords, x_coords)
    
    # Create interactive window for polygon drawing
    create_interactive_window(stretched_average, y_coords, x_coords, stack)

if __name__ == "__main__":
    main() 