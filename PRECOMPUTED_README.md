# SSAST with Precomputed Data

This repository has been modified to support pretraining SSAST models using precomputed `.pt` spectrogram files instead of raw audio files. This is especially useful when working with large datasets on HPC systems where transferring audio files is impractical.

## Prerequisites

- Your audio files have been preprocessed using the script: `src/新的轉pt有正規化_輸出頻譜跟label.py`
- Training and validation .pt files are stored in separate directories
- JSON files mapping original audio paths to labels are available

## Setup and Usage

### 1. Prepare Your Data
Make sure you have:
- Precomputed training spectrograms: `D:\spectrogram_pt_name\` (or your path)
- Precomputed validation spectrograms: `D:\val_spectrogram_pt_name\` (or your path)
- Training JSON: `combined_train_data.json`
- Validation/test JSON: `test.json`
- Labels CSV: `class_labels_indices.csv`

### 2. Using Docker (Recommended for HPC)

**PowerShell:**
```powershell
# Build the image
docker build -f Dockerfile_updated -t ssast-precomputed .

# Run with your data paths
./docker_test.ps1  # Edit paths in the script as needed
```

**Command Prompt:**
```batch
# Build the image
docker build -f Dockerfile_updated -t ssast-precomputed .

# Run with your data paths
docker_test.bat  # Edit paths in the batch file as needed
```

### 3. Direct Python Execution

```bash
python src/run.py --dataset precomputed \
  --data-train combined_train_data.json \
  --data-val test.json \
  --data_dir /path/to/training/pt/files \
  --data_val_dir /path/to/validation/pt/files \
  --label-csv class_labels_indices.csv \
  --dataset_mean -7.4482 \
  --dataset_std 2.4689 \
  --target_length 1024 \
  --num_mel_bins 128 \
  --fshape 16 \
  --tshape 16 \
  --fstride 16 \
  --tstride 16 \
  --model_size base \
  --task pretrain_joint \
  --mask_patch 400 \
  --batch_size 24 \
  --lr 0.0001 \
  --n-epochs 10 \
  --exp-dir exp \
  --save_model True
```

## Key Modifications

1. **Dataloader**: `PrecomputedDataset` class in `src/dataloader_pt_reader.py` handles .pt files
2. **Main Script**: `src/run.py` detects `--dataset precomputed` and uses appropriate dataset class
3. **Dockerfile**: Updated to properly copy all necessary files
4. **Scripts**: Added PowerShell and batch scripts for Docker execution

## Notes

- The original SSAST functionality is preserved - you can still use the original audio-based training
- Precomputed files should be in the format: `{"x": spectrogram_tensor, "y": label_tensor}`
- Spectrograms should be normalized using the same mean/std as used during preprocessing
- Both float32 and float16 tensors are supported in .pt files