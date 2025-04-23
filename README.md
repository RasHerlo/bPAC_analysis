# bPAC Analysis

This project contains tools for analyzing bPAC imaging data.

## Workflow

The typical workflow for analyzing bPAC data is as follows:

1. **Generate Stacks** (`generate_stks.py`)
   - Converts individual TIFF files into stack files
   - Creates `ChanA_stk.tif` and `ChanB_stk.tif` in the STKS directory

2. **Detect and Trace** (`bPAC_detect_and_trace.py`)
   - Processes tif stacks and creates heatmaps
   - Supports interactive polygon drawing for ROI selection
   - Saves ROIs in the ROIs directory

3. **Quantify ROIs** (`quantify_rois.py`)
   - Analyzes selected ROIs
   - Generates normalized and Z-scored traces
   - Provides interactive compensation adjustment

4. **ROI Quantification Overview** (`ROI_quant_overview.py`)
   - Creates summary statistics for all ROIs
   - Generates overview plots of ROI measurements

5. **PDF Overview** (`pdf_overview.py`)
   - Creates a PDF report with all analysis results
   - Includes heatmaps, traces, and statistics

6. **Final Figures** (`final_figs.py`)
   - Generates publication-quality figures
   - Combines data from multiple experiments

## Default Settings

**Note**: Default settings may vary depending on the experiment type and imaging conditions. The following are example settings that should be adjusted based on your specific experiment:

### bPAC_detect_and_trace.py

- **Channel A**:
  - z1 range: 10-30 (baseline)
  - z2 range: 205-245 (response)

- **Channel B**:
  - z1 range: 10-30 (baseline)
  - z2 range: 205-215 (response)

- **Stimulation range**: 202-205

### Usage

The scripts can be run with default settings:
```bash
python generate_stks.py "path/to/directory"
python bPAC_detect_and_trace.py "path/to/directory"
python quantify_rois.py "path/to/directory"
python ROI_quant_overview.py "path/to/directory"
python pdf_overview.py "path/to/directory"
python final_figs.py "path/to/directory"
```

Or with custom settings (example for bPAC_detect_and_trace.py):
```bash
python bPAC_detect_and_trace.py "path/to/directory" --z1_start_a 10 --z1_end_a 30 --z2_start_a 205 --z2_end_a 245 --z1_start_b 10 --z1_end_b 30 --z2_start_b 205 --z2_end_b 215 --z_stim_start 202 --z_stim_end 205
```

### Requirements

The scripts require the following Python packages:
- numpy
- matplotlib
- tifffile
- scikit-image
- Pillow
- tqdm
- reportlab (for PDF generation)

These can be installed using the requirements.txt file:
```bash
pip install -r requirements.txt
```

### Output

The analysis generates:
1. Stack files (ChanA_stk.tif, ChanB_stk.tif)
2. ROI files (.npy format)
3. Heatmaps and traces
4. Normalized and Z-scored data
5. PDF reports
6. Publication-quality figures
