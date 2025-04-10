import os
import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread
from skimage import exposure
import argparse
from matplotlib.widgets import Button
from matplotlib.patches import Polygon
from matplotlib.path import Path

def load_tif_stack(tif_path):
    """
    Load a tif stack from the given path.
    
    Parameters:
    -----------
    tif_path : str
        Path to the tif stack file
        
    Returns:
    --------
    numpy.ndarray
        The loaded tif stack
    """
    print(f"Loading tif stack from: {tif_path}")
    stack = imread(tif_path)
    print(f"Loaded stack with shape: {stack.shape}")
    return stack

def create_heatmaps(stack, z1_range, z2_range):
    """
    Create average and ratio heatmaps from the stack.
    
    Parameters:
    -----------
    stack : numpy.ndarray
        The image stack
    z1_range : tuple
        (start, end) of first z-range
    z2_range : tuple
        (start, end) of second z-range
        
    Returns:
    --------
    tuple
        (average_heatmap, ratio_heatmap)
    """
    z1_start, z1_end = z1_range
    z2_start, z2_end = z2_range
    
    # Calculate average for each z-range
    avg1 = np.mean(stack[z1_start:z1_end], axis=0)
    avg2 = np.mean(stack[z2_start:z2_end], axis=0)
    
    # Calculate ratio (z1_range divided by z2_range)
    ratio = avg1 / (avg2 + 1e-10)  # Add small constant to avoid division by zero
    
    return avg1, ratio

def stretch_heatmaps(average_heatmap, ratio_heatmap):
    """
    Stretch the heatmaps to use the full range.
    
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
    # Stretch average heatmap
    p2, p98 = np.percentile(average_heatmap, (2, 98))
    stretched_average = exposure.rescale_intensity(average_heatmap, in_range=(p2, p98))
    
    # Stretch ratio heatmap
    p2, p98 = np.percentile(ratio_heatmap, (2, 98))
    stretched_ratio = exposure.rescale_intensity(ratio_heatmap, in_range=(p2, p98))
    
    return stretched_average, stretched_ratio

def get_top_pixels(ratio_heatmap, n=1000):
    """
    Get the coordinates of the top n pixels in the ratio heatmap.
    
    Parameters:
    -----------
    ratio_heatmap : numpy.ndarray
        The ratio heatmap
    n : int
        Number of top pixels to select
        
    Returns:
    --------
    tuple
        (y_coords, x_coords) of the top n pixels
    """
    # Flatten the heatmap and get indices of top n pixels
    flat_indices = np.argsort(ratio_heatmap.flatten())[-n:]
    
    # Convert to 2D coordinates
    y_coords, x_coords = np.unravel_index(flat_indices, ratio_heatmap.shape)
    
    return y_coords, x_coords

def extract_and_normalize_traces(stack, y_coords, x_coords):
    """
    Extract and normalize traces for the given coordinates.
    
    Parameters:
    -----------
    stack : numpy.ndarray
        The image stack
    y_coords : numpy.ndarray
        Y coordinates of pixels
    x_coords : numpy.ndarray
        X coordinates of pixels
        
    Returns:
    --------
    numpy.ndarray
        Normalized traces
    """
    # Extract traces
    traces = []
    for y, x in zip(y_coords, x_coords):
        trace = stack[:, y, x]
        traces.append(trace)
    
    # Convert to array and normalize
    traces = np.array(traces)
    traces = (traces - traces.mean(axis=1, keepdims=True)) / traces.std(axis=1, keepdims=True)
    
    return traces

def create_mask_image(shape, y_coords, x_coords):
    """
    Create a mask image with the selected pixels.
    
    Parameters:
    -----------
    shape : tuple
        Shape of the mask image
    y_coords : numpy.ndarray
        Y coordinates of pixels
    x_coords : numpy.ndarray
        X coordinates of pixels
        
    Returns:
    --------
    numpy.ndarray
        The mask image
    """
    mask = np.zeros(shape, dtype=bool)
    mask[y_coords, x_coords] = True
    return mask

def plot_results(stretched_average, stretched_ratio, normalized_traces, mask_image, y_coords, x_coords, channel_name):
    """
    Plot the results.
    
    Parameters:
    -----------
    stretched_average : numpy.ndarray
        The stretched average heatmap
    stretched_ratio : numpy.ndarray
        The stretched ratio heatmap
    normalized_traces : numpy.ndarray
        The normalized traces
    mask_image : numpy.ndarray
        The mask image
    y_coords : numpy.ndarray
        Y coordinates of top pixels
    x_coords : numpy.ndarray
        X coordinates of top pixels
    channel_name : str
        Name of the channel (ChanA or ChanB)
    """
    # Create figure with GridSpec for custom layout
    fig = plt.figure(figsize=(15, 10))
    gs = plt.GridSpec(2, 2, figure=fig)
    
    # Place subplots
    ax1 = fig.add_subplot(gs[0, 0])  # Ratio heatmap
    ax2 = fig.add_subplot(gs[0, 1])  # Mask image
    ax3 = fig.add_subplot(gs[1, 0])  # Normalized traces
    ax4 = fig.add_subplot(gs[1, 1])  # Average heatmap with red dots
    
    # Plot ratio heatmap
    im1 = ax1.imshow(stretched_ratio, cmap='viridis')
    ax1.set_title('Ratio Heatmap')
    plt.colorbar(im1, ax=ax1)
    
    # Plot mask image (inverted colors)
    ax2.imshow(1 - mask_image, cmap='gray')
    ax2.set_title('Mask Image')
    
    # Plot normalized traces
    ax3.imshow(normalized_traces, aspect='auto', cmap='viridis')
    ax3.set_title('Normalized Traces')
    ax3.set_xlabel('Frame')
    ax3.set_ylabel('Pixel')
    
    # Plot average heatmap with red dots
    ax4.imshow(stretched_average, cmap='viridis')
    ax4.scatter(x_coords, y_coords, c='red', s=1, alpha=0.5)
    ax4.set_title('Average Heatmap with Top 1000 Pixels')
    
    # Add overall title
    fig.suptitle(f'{channel_name} Analysis', fontsize=16)
    
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

class DualChannelPolygonDrawer:
    def __init__(self, ax_a, ax_b, image_a, image_b, stack_a, stack_b, trace_ax_a, trace_ax_b, sum_ax,
                 z1_start_a, z1_end_a, z2_start_a, z2_end_a,
                 z1_start_b, z1_end_b, z2_start_b, z2_end_b,
                 z_stim_start, z_stim_end):
        # Store z-range boundaries first
        self.z1_start_a = z1_start_a
        self.z1_end_a = z1_end_a
        self.z2_start_a = z2_start_a
        self.z2_end_a = z2_end_a
        self.z1_start_b = z1_start_b
        self.z1_end_b = z1_end_b
        self.z2_start_b = z2_start_b
        self.z2_end_b = z2_end_b
        self.z_stim_start = z_stim_start
        self.z_stim_end = z_stim_end
        
        # Store other parameters
        self.ax_a = ax_a
        self.ax_b = ax_b
        self.image_a = image_a
        self.image_b = image_b
        self.stack_a = stack_a
        self.stack_b = stack_b
        self.trace_ax_a = trace_ax_a
        self.trace_ax_b = trace_ax_b
        self.sum_ax = sum_ax
        
        self.polygon = None
        self.vertices = []
        self.polygon_patch_a = None
        self.polygon_patch_b = None
        self.lines_a = []  # Store all line segments for ChanA
        self.lines_b = []  # Store all line segments for ChanB
        self.vertex_markers = []  # Store vertex markers
        self.dragged_vertex = None
        self.drag_index = None
        self.is_finished = False
        self.temp_line = None
        
        # Connect events
        self.cidpress = ax_a.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cidrelease = ax_a.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmotion = ax_a.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)
        
        # Add clear button
        self.clear_button_ax = plt.axes([0.81, 0.05, 0.1, 0.04])
        self.clear_button = Button(self.clear_button_ax, 'Clear')
        self.clear_button.on_clicked(self.clear_polygon)
        
        # Add finish button
        self.finish_button_ax = plt.axes([0.7, 0.05, 0.1, 0.04])
        self.finish_button = Button(self.finish_button_ax, 'Finish')
        self.finish_button.on_clicked(self.finish_polygon)
        
        self.is_drawing = False

    def add_vertex_marker(self, x, y):
        """Add a draggable vertex marker"""
        marker_a, = self.ax_a.plot(x, y, 'ro', markersize=8, picker=5)
        marker_b, = self.ax_b.plot(x, y, 'ro', markersize=8, picker=5)
        self.vertex_markers.append((marker_a, marker_b))
        return marker_a, marker_b

    def normalize_trace_excluding_stim(self, trace):
        """Normalize trace excluding the stimulation range"""
        # Create a mask for non-stimulation frames
        non_stim_mask = np.ones_like(trace, dtype=bool)
        non_stim_mask[self.z_stim_start:self.z_stim_end] = False
        
        # Get min and max from non-stimulation frames
        min_val = np.min(trace[non_stim_mask])
        max_val = np.max(trace[non_stim_mask])
        
        # Normalize using these values
        return (trace - min_val) / (max_val - min_val)

    def update_traces(self):
        """Update the trace plots with current polygon"""
        if len(self.vertices) < 3 or not self.is_finished:
            return
            
        # Clear previous traces
        self.trace_ax_a.clear()
        self.trace_ax_b.clear()
        self.sum_ax.clear()
        
        # Extract and plot traces for ChanA
        traces_a = extract_traces_in_polygon(self.stack_a, self.vertices)
        if len(traces_a) > 0:
            # Normalize traces
            traces_a = (traces_a - traces_a.mean(axis=1, keepdims=True)) / traces_a.std(axis=1, keepdims=True)
            
            # Plot trace matrix for ChanA
            self.trace_ax_a.imshow(traces_a, aspect='auto', cmap='viridis')
            self.trace_ax_a.set_title(f'ChanA Traces ({len(traces_a)} pixels)')
            self.trace_ax_a.set_xlabel('Frame')
            self.trace_ax_a.set_ylabel('Pixel')
            
            # Calculate sum for ChanA
            sum_trace_a = np.sum(traces_a, axis=0)
            # Normalize sum trace excluding stimulation range
            sum_trace_a = self.normalize_trace_excluding_stim(sum_trace_a)
            
            # Calculate ratio for ChanA within ROI
            z1_range_a = slice(self.z1_start_a, self.z1_end_a)
            z2_range_a = slice(self.z2_start_a, self.z2_end_a)
            baseline_a = np.mean(traces_a[:, z1_range_a], axis=1)
            response_a = np.mean(traces_a[:, z2_range_a], axis=1)
            ratio_a = np.mean(response_a / baseline_a)
        else:
            self.trace_ax_a.text(0.5, 0.5, 'No pixels in polygon', 
                               horizontalalignment='center',
                               verticalalignment='center',
                               transform=self.trace_ax_a.transAxes)
            sum_trace_a = None
            ratio_a = None
        
        # Extract and plot traces for ChanB
        traces_b = extract_traces_in_polygon(self.stack_b, self.vertices)
        if len(traces_b) > 0:
            # Normalize traces
            traces_b = (traces_b - traces_b.mean(axis=1, keepdims=True)) / traces_b.std(axis=1, keepdims=True)
            
            # Plot trace matrix for ChanB
            self.trace_ax_b.imshow(traces_b, aspect='auto', cmap='viridis')
            self.trace_ax_b.set_title(f'ChanB Traces ({len(traces_b)} pixels)')
            self.trace_ax_b.set_xlabel('Frame')
            self.trace_ax_b.set_ylabel('Pixel')
            
            # Calculate sum for ChanB
            sum_trace_b = np.sum(traces_b, axis=0)
            # Normalize sum trace excluding stimulation range
            sum_trace_b = self.normalize_trace_excluding_stim(sum_trace_b)
            
            # Calculate ratio for ChanB within ROI
            z1_range_b = slice(self.z1_start_b, self.z1_end_b)
            z2_range_b = slice(self.z2_start_b, self.z2_end_b)
            baseline_b = np.mean(traces_b[:, z1_range_b], axis=1)
            response_b = np.mean(traces_b[:, z2_range_b], axis=1)
            ratio_b = np.mean(response_b / baseline_b)
        else:
            self.trace_ax_b.text(0.5, 0.5, 'No pixels in polygon', 
                               horizontalalignment='center',
                               verticalalignment='center',
                               transform=self.trace_ax_b.transAxes)
            sum_trace_b = None
            ratio_b = None
        
        # Plot combined sum with ratio values in legend
        if sum_trace_a is not None:
            label_a = f'ChanA (ratio: {ratio_a:.2f})' if ratio_a is not None else 'ChanA'
            self.sum_ax.plot(sum_trace_a, 'r-', linewidth=1, label=label_a)
        if sum_trace_b is not None:
            label_b = f'ChanB (ratio: {ratio_b:.2f})' if ratio_b is not None else 'ChanB'
            self.sum_ax.plot(sum_trace_b, 'g-', linewidth=1, label=label_b)
        
        self.sum_ax.set_title('Normalized Sum of Traces')
        self.sum_ax.set_xlabel('Frame')
        self.sum_ax.set_ylabel('Normalized Intensity')
        self.sum_ax.grid(True)
        self.sum_ax.legend()
        
        # Set y-axis limits to [-0.25, 1.25] for normalized traces
        self.sum_ax.set_ylim(-0.25, 1.25)
        
        self.trace_ax_a.figure.canvas.draw_idle()
        self.trace_ax_b.figure.canvas.draw_idle()
        self.sum_ax.figure.canvas.draw_idle()
        
    def update_polygon(self):
        """Update the polygon and its lines based on current vertices"""
        # Clear existing lines
        for line_a, line_b in self.lines_a + self.lines_b:
            line_a.remove()
            line_b.remove()
        self.lines_a = []
        self.lines_b = []
        
        # Clear existing vertex markers
        for marker_a, marker_b in self.vertex_markers:
            marker_a.remove()
            marker_b.remove()
        self.vertex_markers = []
        
        # Clear existing polygon patches
        if self.polygon_patch_a:
            self.polygon_patch_a.remove()
            self.polygon_patch_a = None
        if self.polygon_patch_b:
            self.polygon_patch_b.remove()
            self.polygon_patch_b = None
        
        # Draw new lines and markers
        if len(self.vertices) > 1:
            # Create closed polygon vertices
            closed_vertices = self.vertices + [self.vertices[0]] if self.is_finished else self.vertices
            
            # Draw polygon patches if finished
            if self.is_finished:
                self.polygon_patch_a = Polygon(closed_vertices, fill=False, color='red')
                self.ax_a.add_patch(self.polygon_patch_a)
                self.polygon_patch_b = Polygon(closed_vertices, fill=False, color='red')
                self.ax_b.add_patch(self.polygon_patch_b)
            
            # Add vertex markers and line segments
            for i in range(len(self.vertices)):
                # Add vertex markers
                self.add_vertex_marker(self.vertices[i][0], self.vertices[i][1])
                
                # Add line segments
                if i < len(self.vertices) - 1:
                    line_a, = self.ax_a.plot([self.vertices[i][0], self.vertices[i+1][0]],
                                           [self.vertices[i][1], self.vertices[i+1][1]], 'r-')
                    line_b, = self.ax_b.plot([self.vertices[i][0], self.vertices[i+1][0]],
                                           [self.vertices[i][1], self.vertices[i+1][1]], 'r-')
                    self.lines_a.append((line_a, line_b))
            
            # Add closing line if finished
            if self.is_finished:
                line_a, = self.ax_a.plot([self.vertices[-1][0], self.vertices[0][0]],
                                       [self.vertices[-1][1], self.vertices[0][1]], 'r-')
                line_b, = self.ax_b.plot([self.vertices[-1][0], self.vertices[0][0]],
                                       [self.vertices[-1][1], self.vertices[0][1]], 'r-')
                self.lines_a.append((line_a, line_b))
        
        # Update traces if polygon is finished
        if self.is_finished:
            self.update_traces()
            
        self.ax_a.figure.canvas.draw_idle()
        self.ax_b.figure.canvas.draw_idle()
        
    def on_press(self, event):
        if event.inaxes != self.ax_a:
            return
            
        # Check if we're clicking on a vertex marker
        if event.button == 1:  # Left click
            for i, (marker_a, marker_b) in enumerate(self.vertex_markers):
                if marker_a.contains(event)[0] or marker_b.contains(event)[0]:
                    self.dragged_vertex = (marker_a, marker_b)
                    self.drag_index = i
                    return
                    
            # If not clicking on a vertex, add new vertex
            self.is_drawing = True
            self.vertices.append([event.xdata, event.ydata])
            self.update_polygon()
        
    def on_motion(self, event):
        if event.inaxes != self.ax_a:
            return
            
        if self.dragged_vertex is not None:
            # Update vertex position
            self.vertices[self.drag_index] = [event.xdata, event.ydata]
            self.update_polygon()
        elif self.is_drawing and len(self.vertices) > 0:
            # Show temporary line while drawing
            if self.temp_line:
                self.temp_line.remove()
            self.temp_line, = self.ax_a.plot([self.vertices[-1][0], event.xdata],
                                           [self.vertices[-1][1], event.ydata], 'r-')
            self.ax_a.figure.canvas.draw_idle()
            
    def on_release(self, event):
        if event.inaxes != self.ax_a:
            return
        self.is_drawing = False
        self.dragged_vertex = None
        self.drag_index = None
        
    def clear_polygon(self, event):
        self.vertices = []
        self.is_finished = False
        if self.polygon_patch_a:
            self.polygon_patch_a.remove()
            self.polygon_patch_a = None
        if self.polygon_patch_b:
            self.polygon_patch_b.remove()
            self.polygon_patch_b = None
        if self.temp_line:
            self.temp_line.remove()
            self.temp_line = None
        for line_a, line_b in self.lines_a + self.lines_b:
            line_a.remove()
            line_b.remove()
        self.lines_a = []
        self.lines_b = []
        for marker_a, marker_b in self.vertex_markers:
            marker_a.remove()
            marker_b.remove()
        self.vertex_markers = []
        self.ax_a.figure.canvas.draw_idle()
        self.ax_b.figure.canvas.draw_idle()
        
    def finish_polygon(self, event):
        if len(self.vertices) < 3:
            return
        self.is_finished = True
        self.update_polygon()

def create_dual_channel_interactive_window(stretched_average_a, stretched_average_b, y_coords_a, x_coords_a, y_coords_b, x_coords_b, stack_a, stack_b,
                                         z1_start_a, z1_end_a, z2_start_a, z2_end_a,
                                         z1_start_b, z1_end_b, z2_start_b, z2_end_b,
                                         z_stim_start, z_stim_end):
    """
    Create an interactive window for polygon drawing on both channels.
    
    Parameters:
    -----------
    stretched_average_a : numpy.ndarray
        The stretched average heatmap for ChanA
    stretched_average_b : numpy.ndarray
        The stretched average heatmap for ChanB
    y_coords_a : numpy.ndarray
        Y coordinates of top pixels for ChanA
    x_coords_a : numpy.ndarray
        X coordinates of top pixels for ChanA
    y_coords_b : numpy.ndarray
        Y coordinates of top pixels for ChanB
    x_coords_b : numpy.ndarray
        X coordinates of top pixels for ChanB
    stack_a : numpy.ndarray
        The image stack for ChanA
    stack_b : numpy.ndarray
        The image stack for ChanB
    z1_start_a, z1_end_a : int
        Start and end of first z-range for ChanA
    z2_start_a, z2_end_a : int
        Start and end of second z-range for ChanA
    z1_start_b, z1_end_b : int
        Start and end of first z-range for ChanB
    z2_start_b, z2_end_b : int
        Start and end of second z-range for ChanB
    z_stim_start, z_stim_end : int
        Start and end of stimulation range
    """
    fig = plt.figure(figsize=(20, 15))
    
    # Create subplots with further adjusted height ratios
    gs = plt.GridSpec(3, 2, figure=fig, height_ratios=[0.25, 1, 0.25])
    
    # ChanA subplots
    ax_a_image = fig.add_subplot(gs[1, 0])  # ChanA image (larger)
    ax_a_traces = fig.add_subplot(gs[0, 0])  # ChanA traces (smaller)
    
    # ChanB subplots
    ax_b_image = fig.add_subplot(gs[1, 1])  # ChanB image (larger)
    ax_b_traces = fig.add_subplot(gs[0, 1])  # ChanB traces (smaller)
    
    # Combined sum subplot
    ax_sum = fig.add_subplot(gs[2, :])  # Sum of traces (smaller)
    
    # Plot heatmaps and dots
    ax_a_image.imshow(stretched_average_a, cmap='viridis')
    ax_a_image.scatter(x_coords_a, y_coords_a, c='red', s=1, alpha=0.5)
    ax_a_image.set_title('ChanA - Draw polygon by clicking. Press "Finish" to close polygon.')
    
    ax_b_image.imshow(stretched_average_b, cmap='viridis')
    ax_b_image.scatter(x_coords_b, y_coords_b, c='red', s=1, alpha=0.5)
    ax_b_image.set_title('ChanB')
    
    # Initialize polygon drawer with z-range boundaries
    drawer = DualChannelPolygonDrawer(ax_a_image, ax_b_image, stretched_average_a, stretched_average_b, 
                                     stack_a, stack_b, ax_a_traces, ax_b_traces, ax_sum,
                                     z1_start_a, z1_end_a, z2_start_a, z2_end_a,
                                     z1_start_b, z1_end_b, z2_start_b, z2_end_b,
                                     z_stim_start, z_stim_end)
    plt.show()

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process tif stacks and create heatmaps')
    parser.add_argument('directory', type=str, help='Directory containing ChanA_stk.tif and ChanB_stk.tif')
    
    # ChanA z-ranges
    parser.add_argument('--z1_start_a', type=int, required=True, help='Start of first z-range for ChanA')
    parser.add_argument('--z1_end_a', type=int, required=True, help='End of first z-range for ChanA')
    parser.add_argument('--z2_start_a', type=int, required=True, help='Start of second z-range for ChanA')
    parser.add_argument('--z2_end_a', type=int, required=True, help='End of second z-range for ChanA')
    
    # ChanB z-ranges
    parser.add_argument('--z1_start_b', type=int, required=True, help='Start of first z-range for ChanB')
    parser.add_argument('--z1_end_b', type=int, required=True, help='End of first z-range for ChanB')
    parser.add_argument('--z2_start_b', type=int, required=True, help='Start of second z-range for ChanB')
    parser.add_argument('--z2_end_b', type=int, required=True, help='End of second z-range for ChanB')
    
    # Stimulation range
    parser.add_argument('--z_stim_start', type=int, required=True, help='Start of stimulation range')
    parser.add_argument('--z_stim_end', type=int, required=True, help='End of stimulation range')
    
    args = parser.parse_args()
    
    # Construct file paths
    chan_a_path = os.path.join(args.directory, 'ChanA_stk.tif')
    chan_b_path = os.path.join(args.directory, 'ChanB_stk.tif')
    
    # Load tif stacks
    stack_a = load_tif_stack(chan_a_path)
    stack_b = load_tif_stack(chan_b_path)
    
    # Validate z-ranges for ChanA
    if not (0 <= args.z1_start_a < args.z1_end_a <= stack_a.shape[0] and 
            0 <= args.z2_start_a < args.z2_end_a <= stack_a.shape[0]):
        raise ValueError("Invalid z-ranges for ChanA. Must be within stack dimensions.")
    
    # Validate z-ranges for ChanB
    if not (0 <= args.z1_start_b < args.z1_end_b <= stack_b.shape[0] and 
            0 <= args.z2_start_b < args.z2_end_b <= stack_b.shape[0]):
        raise ValueError("Invalid z-ranges for ChanB. Must be within stack dimensions.")
    
    # Validate stimulation range
    if not (0 <= args.z_stim_start < args.z_stim_end <= stack_a.shape[0]):
        raise ValueError("Invalid stimulation range. Must be within stack dimensions.")
    
    # Process ChanA with its specific z-ranges
    average_heatmap_a, ratio_heatmap_a = create_heatmaps(
        stack_a, 
        (args.z1_start_a, args.z1_end_a), 
        (args.z2_start_a, args.z2_end_a)
    )
    stretched_average_a, stretched_ratio_a = stretch_heatmaps(average_heatmap_a, ratio_heatmap_a)
    y_coords_a, x_coords_a = get_top_pixels(ratio_heatmap_a)  # Get top pixels from ratio heatmap
    normalized_traces_a = extract_and_normalize_traces(stack_a, y_coords_a, x_coords_a)
    mask_image_a = create_mask_image(ratio_heatmap_a.shape, y_coords_a, x_coords_a)
    
    # Process ChanB with its specific z-ranges
    average_heatmap_b, ratio_heatmap_b = create_heatmaps(
        stack_b, 
        (args.z1_start_b, args.z1_end_b), 
        (args.z2_start_b, args.z2_end_b)
    )
    stretched_average_b, stretched_ratio_b = stretch_heatmaps(average_heatmap_b, ratio_heatmap_b)
    y_coords_b, x_coords_b = get_top_pixels(ratio_heatmap_b)  # Get top pixels from ratio heatmap
    normalized_traces_b = extract_and_normalize_traces(stack_b, y_coords_b, x_coords_b)
    mask_image_b = create_mask_image(ratio_heatmap_b.shape, y_coords_b, x_coords_b)
    
    # Plot results for each channel
    plot_results(stretched_average_a, stretched_ratio_a, normalized_traces_a, mask_image_a, y_coords_a, x_coords_a, 'ChanA')
    plot_results(stretched_average_b, stretched_ratio_b, normalized_traces_b, mask_image_b, y_coords_b, x_coords_b, 'ChanB')
    
    # Create interactive window for dual channel polygon drawing
    create_dual_channel_interactive_window(stretched_average_a, stretched_average_b, y_coords_a, x_coords_a, y_coords_b, x_coords_b, stack_a, stack_b,
                                         args.z1_start_a, args.z1_end_a, args.z2_start_a, args.z2_end_a,
                                         args.z1_start_b, args.z1_end_b, args.z2_start_b, args.z2_end_b,
                                         args.z_stim_start, args.z_stim_end)

if __name__ == "__main__":
    main() 