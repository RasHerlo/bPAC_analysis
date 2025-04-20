# bPAC Analysis

This project contains tools for analyzing bPAC imaging data.

## bPAC_detect_and_trace.py

This script processes tif stacks and creates heatmaps for bPAC analysis. It supports dual-channel analysis with interactive polygon drawing capabilities.

### Default Settings

The script uses the following default settings:

- **Channel A**:
  - z1 range: 40-90 (numerator in ratio calculation)
  - z2 range: 10-30 (denominator in ratio calculation)

- **Channel B**:
  - z1 range: 40-90 (numerator in ratio calculation)
  - z2 range: 10-30 (denominator in ratio calculation)

- **Stimulation range**: 35-40

### Usage

The script can be run in two ways:

1. Using default settings:
```bash
python bPAC_detect_and_trace.py "path/to/directory"
```

2. With custom settings:
```bash
python bPAC_detect_and_trace.py "path/to/directory" --z1_start_a 40 --z1_end_a 90 --z2_start_a 10 --z2_end_a 30 --z1_start_b 40 --z1_end_b 90 --z2_start_b 10 --z2_end_b 30 --z_stim_start 35 --z_stim_end 40
```

### Requirements

The script requires the following Python packages:
- numpy
- matplotlib
- tifffile
- scikit-image
- Pillow
- tqdm

These can be installed using the requirements.txt file:
```bash
pip install -r requirements.txt
```

### Output

The script generates:
1. Heatmaps for both channels
2. Interactive polygon drawing interface
3. Normalized traces for selected regions
4. Ratio calculations between z1 and z2 ranges
