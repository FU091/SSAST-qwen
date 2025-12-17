@echo off
echo Building Docker image for SSAST...
docker build -f Dockerfile_updated -t ssast-precomputed .

if %errorlevel% neq 0 (
    echo Docker build failed!
    exit /b 1
)

echo Docker image built successfully!

echo Running SSAST pretraining with precomputed .pt files...
docker run -it --rm ^
    --gpus all ^  REM Enable GPU access if available
    -v D:\spectrogram_pt_name:/data/train ^  REM Mount your training .pt files directory
    -v D:\val_spectrogram_pt_name:/data/val ^  REM Mount your validation .pt files directory
    -v %cd%:/ssast ^  REM Mount current directory for JSON files
    ssast-precomputed ^
    python src/run.py --dataset precomputed ^
    --data-train /ssast/combined_train_data.json ^
    --data-val /ssast/test.json ^
    --data_dir /data/train ^
    --data_val_dir /data/val ^
    --label-csv /ssast/class_labels_indices.csv ^
    --dataset_mean -7.4482 ^
    --dataset_std 2.4689 ^
    --target_length 1024 ^
    --num_mel_bins 128 ^
    --fshape 16 ^
    --tshape 16 ^
    --fstride 16 ^
    --tstride 16 ^
    --model_size base ^
    --task pretrain_joint ^
    --mask_patch 400 ^
    --batch_size 24 ^
    --lr 0.0001 ^
    --n-epochs 10 ^
    --exp-dir /ssast/exp ^
    --save_model False ^
    --freqm 0 ^
    --timem 0 ^
    --mixup 0 ^
    --n-print-steps 100 ^
    --lr_patience 2 ^
    --epoch_iter 4000

echo Docker container execution completed.