# PowerShell script to build and run SSAST with precomputed .pt files using Docker

# Build the Docker image
Write-Host "Building Docker image for SSAST..." -ForegroundColor Green
docker build -f Dockerfile_updated -t ssast-precomputed .

# Check if build was successful
if ($LASTEXITCODE -ne 0) {
    Write-Host "Docker build failed!" -ForegroundColor Red
    exit 1
}

Write-Host "Docker image built successfully!" -ForegroundColor Green

# Example command to run the container with your precomputed data mounted
# Adjust the paths according to your system's directory structure

Write-Host "Running SSAST pretraining with precomputed .pt files..." -ForegroundColor Green

docker run -it --rm `
    --gpus all `  # Enable GPU access if available
    -v "D:\spectrogram_pt_name:/data/train" `  # Mount your training .pt files directory
    -v "D:\val_spectrogram_pt_name:/data/val" `  # Mount your validation .pt files directory
    -v "$(Get-Location):/ssast" `  # Mount current directory for JSON files
    ssast-precomputed `
    python src/run.py --dataset precomputed `
    --data-train /ssast/combined_train_data.json `
    --data-val /ssast/test.json `
    --data_dir /data/train `
    --data_val_dir /data/val `
    --label-csv /ssast/class_labels_indices.csv `
    --dataset_mean -7.4482 `
    --dataset_std 2.4689 `
    --target_length 1024 `
    --num_mel_bins 128 `
    --fshape 16 `
    --tshape 16 `
    --fstride 16 `
    --tstride 16 `
    --model_size base `
    --task pretrain_joint `
    --mask_patch 400 `
    --batch_size 24 `
    --lr 0.0001 `
    --n-epochs 10 `
    --exp-dir /ssast/exp `
    --save_model False `
    --freqm 0 `
    --timem 0 `
    --mixup 0 `
    --n-print-steps 100 `
    --lr_patience 2 `
    --epoch_iter 4000

Write-Host "Docker container execution completed." -ForegroundColor Green