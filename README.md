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

4. **Create Excel File** (Manual Step)
   - After running `quantify_rois.py`, create an Excel file containing:
     - `MOUSE` - The mouse identifier
     - `EXP` - The experiment identifier
     - `ROI#` - The ROI number
   - This file will be used as input for the next step

5. **ROI Quantification Overview** (`ROI_quant_overview.py`)
   - Creates summary statistics for all ROIs
   - Generates overview plots of ROI measurements
   - Uses the manually created Excel file as input

6. **PDF Overview** (`pdf_overview.py`)
   - Creates a PDF report with all analysis results
   - Includes heatmaps, traces, and statistics

7. **Final Figures** (`final_figs.py`)
   - Generates publication-quality figures
   - Combines data from multiple experiments

## Default Settings

**Note**: Default settings may vary depending on the experiment type and imaging conditions. The following are example settings that should be adjusted based on your specific experiment.

NB: This set of settings is called 'Setting Set 1' and is used for experiments with APs, bPAC, APs:

### bPAC_detect_and_trace.py

- **Channel A**:
  - z1 range: 10-30 (baseline)
  - z2 range: 205-245 (response)

- **Channel B**:
  - z1 range: 10-30 (baseline)
  - z2 range: 205-215 (response)

- **Stimulation range**: 202-205

Another set of settings, called 'Setting Set 2' is used for spontaneous bPAC-experiments:

- **Channel A**:
  - z1 range: 10-30 (baseline)
  - z2 range: 45-85 (response)

- **Channel B**:
  - z1 range: 10-30 (baseline)
  - z2 range: 45-55 (response)

- **Stimulation range**: 35-45


### Usage

The scripts can be run with default settings:
```bash
python generate_stks.py "path/to/directory"
python bPAC_detect_and_trace.py "path/to/directory"
python quantify_rois.py "path/to/directory"
# Create Excel file manually here
python ROI_quant_overview.py "path/to/directory"
python pdf_overview.py "path/to/directory"
python final_figs.py "path/to/directory"
```

Or with custom settings (here shown with 'Setting Set 1'):
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

### Excel File Requirements for ROI_quant_overview.py

The input Excel file for `ROI_quant_overview.py` should contain the following required columns:

1. `MOUSE` - The mouse identifier
2. `EXP` - The experiment identifier
3. `ROI#` - The ROI number

The script will automatically add the following columns during processing:
1. `ChanA_raw_trc` - Raw trace from Channel A
2. `ChanB_raw_trc` - Raw trace from Channel B
3. `ChanA_comp_trc` - Compensated trace for Channel A (raw - 0.15 * Channel B)
4. `Stim` - Stimulation point frame number
5. `ChanA_cut_trc` - Cut trace for Channel A (15 frames before to 85 frames after stimulation)
6. `ChanB_cut_trc` - Cut trace for Channel B (15 frames before to 85 frames after stimulation)
7. `ChanA_norm_trc` - Normalized trace for Channel A
8. `ChanB_norm_trc` - Normalized trace for Channel B
9. `ChanA_Z_trc` - Z-scored trace for Channel A
10. `ChanB_Z_trc` - Z-scored trace for Channel B

The script expects the Excel file to be in a directory structure where:
- The Excel file can be in any subdirectory of the parent directory
- The TIFF stacks should be in a `STKS` subdirectory
- The ROI files should be in a `ROIs` subdirectory within the `STKS` directory

Example directory structure:
```
parent_directory/
├── Excel_file.xlsx
└── MOUSE/
    └── EXP/
        └── STKS/
            ├── ChanA_stk.tif
            ├── ChanB_stk.tif
            └── ROIs/
                ├── ROI#1.npy
                ├── ROI#2.npy
                └── ...
```
