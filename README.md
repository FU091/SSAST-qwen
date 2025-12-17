# SSAST Pretraining with Custom .pt Files

This repository contains modifications to enable SSAST self-supervised pretraining using precomputed .pt files, addressing path compatibility and library version issues.

## Problem Solved

1. **Path Compatibility Issues**: Fixed Windows/Linux path handling for .pt files
2. **CUDA Compatibility**: Added CPU fallback for CUDA errors
3. **Library Version Issues**: Updated for newer timm library compatibility
4. **Parameter Validation**: Fixed model parameter validation

## Key Modifications

### 1. Improved `dataloader_pt_reader.py`
- Added robust path handling with regex for both Windows and Unix style paths
- Enhanced error handling and file validation logic
- Added CPU fallback for CUDA errors

```python
# Use regex to handle both Windows and Unix paths
path_parts = re.split(r'[\\/]+', original_path)
filename = path_parts[-1]  # Get last part which should be the actual filename
filename_no_ext = os.path.splitext(filename)[0]
pt_path = os.path.join(self.data_dir, f"{filename_no_ext}.pt")
```

### 2. Updated `run.py`
- Added CUDA error handling and CPU fallback
- Fixed model parameter validation
- Improved path mapping for JSON files

### 3. Modified `ast_models.py`
- Added timm library version compatibility
- Added fallback model names for newer timm versions

```python
# Fallback for newer timm versions
try:
    self.v = timm.create_model('vit_deit_base_distilled_patch16_384', pretrained=False)
except:
    self.v = timm.create_model('deit_base_distilled_patch16_384', pretrained=False)
```

## Data Preparation

Your data should be organized as follows:
- Training .pt files: `D:\spectrogram_pt_name\`
- Validation .pt files: `D:\val_spectrogram_pt_name\`
- Training JSON: `combined_train_data.json` 
- Validation JSON: `fs_sound_list.json`

The .pt files should be named to match the audio filenames referenced in your JSON files.

## Training Commands

### Local CPU Training
```bash
python run.py --dataset precomputed \
--data-train "D:\spectrogram_pt_name" \
--data-val "D:\val_spectrogram_pt_name" \
--exp-dir "./exp/test_cpu" \
--label-csv "../class_labels_indices.csv" \
--lr 1e-4 --n-epochs 10 --batch-size 24 --save_model True \
--freqm 0 --timem 0 --mixup 0 --bal none \
--tstride 16 --fstride 16 --fshape 16 --tshape 16 \
--dataset_mean -7.4482 --dataset_std 2.4689 \
--target_length 1024 --num_mel_bins 128 \
--model_size base --mask_patch 400 --n-print-steps 100 \
--task pretrain_joint --lr_patience 2 --epoch_iter 4000
```

### Docker Training
```bash
docker run --gpus all -it --rm \
  --ipc=host \
  -v "C:\Users\Lin\Desktop\2_code\ssast_hub:/workspace/ssast_hub" \
  -v "D:\spectrogram_pt_name:/workspace/data/train" \
  -v "D:\val_spectrogram_pt_name:/workspace/data/val" \
  -e CUDA_LAUNCH_BLOCKING=1 \
  ssast_hub_image python -W ignore /workspace/ssast_hub/ssast-main/src/run.py --dataset precomputed \
--data-train "/workspace/data/train" \
--data-val "/workspace/data/val" \
--exp-dir "/workspace/ssast_hub/ssast-main/src/exp/test_docker_01" \
--label-csv "/workspace/ssast_hub/ssast-main/class_labels_indices.csv" \
--lr 1e-4 --n-epochs 10 --batch-size 24 --save_model False \
--freqm 0 --timem 0 --mixup 0 --bal none \
--tstride 16 --fstride 16 --fshape 16 --tshape 16 \
--dataset_mean -7.4482 --dataset_std 2.4689 \
--target_length 1024 --num_mel_bins 128 \
--model_size base --mask_patch 400 --n-print-steps 100 \
--task pretrain_joint --lr_patience 2 --epoch_iter 4000
```

## Prerequisites

- Anaconda environment with PyTorch
- Audio files converted to .pt format using the preprocessing scripts
- JSON files mapping to the .pt files

## Docker Setup

If using Docker:
1. Build your ssast_hub_image
2. Ensure volumes are properly mounted
3. Run with GPU support if available

## Testing

Run the test script to verify .pt file loading:
```bash
python test_pt_loading.py
```

## Authors

- [FU091](https://github.com/FU091) - Main contributor
- Original SSAST authors

## License

See original SSAST license for details.