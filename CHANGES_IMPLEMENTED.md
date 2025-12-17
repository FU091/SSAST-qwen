# SSAST Pretraining with Precomputed .pt Files - Changes Implemented

## Overview
This document details all changes made to enable SSAST pretraining using precomputed `.pt` spectrogram files instead of raw audio files. This approach allows for efficient pretraining on HPC systems where transferring large audio files is impractical.

## Files Modified/Added

### 1. `ssast-main\src\dataloader_pt_reader.py` (Updated)
- **Purpose**: Enhanced the `PrecomputedDataset` class to support additional configuration parameters
- **Changes**:
  - Modified the constructor to accept an optional `audio_conf` parameter
  - Added backward compatibility for when data_dir is passed via audio_conf
  - Added logic to infer data_dir from dataset_json_file if not explicitly provided
  - Maintained all original functionality while allowing more flexible initialization

### 2. `ssast-main\src\run.py` (Updated)
- **Purpose**: Modified the main training script to handle precomputed datasets
- **Changes**:
  - Added `--data_val_dir` command-line argument for specifying validation data directory
  - Added logic to detect `dataset=precomputed` and switch to using `PrecomputedDataset`
  - Updated data loading logic to route to appropriate dataset class based on dataset parameter
  - Added proper handling of audio_conf for both training and validation datasets
  - Added error handling for CUDA-related issues with fallback to CPU
  - Added proper data directory mapping for both training and validation sets

### 3. `ssast-main\src\pretrain\run_mask_patch_precomputed.sh` (New)
- **Purpose**: Shell script specifically designed for pretraining with precomputed data
- **Changes**:
  - Created new pretraining script tailored for precomputed .pt files
  - Configured paths for training and validation data directories
  - Set up proper parameters for precomputed dataset mode

### 4. `Dockerfile_updated` (New)
- **Purpose**: Updated Dockerfile to properly handle the modified SSAST project
- **Changes**:
  - Corrected the file copying structure to include all necessary components
  - Maintained all original dependencies
  - Ensured proper working directory setup

### 5. `docker_test.ps1` and `docker_test.bat` (New)
- **Purpose**: PowerShell and batch scripts for running Docker container
- **Changes**:
  - Created ready-to-use scripts for Docker execution
  - Included proper volume mounting for .pt files
  - Added comprehensive command-line arguments

## Key Features Implemented

### 1. Flexible Dataset Detection
- The system now automatically detects `--dataset precomputed` and uses the appropriate dataloader
- Maintains backward compatibility with original audio-based training

### 2. Efficient .pt File Handling
- Uses filename mapping from JSON files to corresponding .pt files
- Supports both training and validation datasets
- Handles missing files gracefully with warnings

### 3. Proper Data Flow
- Original audio filenames are mapped to .pt files with same naming convention
- Preserves all preprocessing steps and normalization
- Maintains compatibility with SSAST model architecture

### 4. Docker Integration
- Complete Docker setup for HPC deployment
- Proper volume mounting for large .pt files
- GPU support maintained

## Usage Instructions

### For Direct Python Execution:
```bash
python src/run.py --dataset precomputed \
  --data-train /path/to/training/json \
  --data-val /path/to/validation/json \
  --data_dir /path/to/training/pt/files \
  --data_val_dir /path/to/validation/pt/files \
  --label-csv /path/to/labels.csv \
  --task pretrain_joint \
  # ... other parameters
```

### For Docker Execution:
1. Build: `docker build -f Dockerfile_updated -t ssast-precomputed .`
2. Run: Use the provided `docker_test.ps1` or `docker_test.bat` scripts with appropriate paths

## Benefits of This Approach

1. **Efficiency**: Eliminates repeated audio processing during training
2. **HPC Compatibility**: Avoids transferring large audio files to cloud/HPC systems
3. **Preprocessing Consistency**: Ensures all spectrograms are preprocessed identically
4. **Scalability**: Enables faster iteration and experimentation on precomputed features

## Validation
- All original SSAST functionality maintained
- Precomputed dataset support added without breaking changes
- Memory-efficient float16 storage preserved in .pt files
- Full compatibility with existing model architectures maintained