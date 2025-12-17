# 修復 run.py 檔案中的語法錯誤
$runPyPath = "C:\Users\Lin\Desktop\2_code\ssast_hub\ssast-main\src\run.py"

Write-Host "開始修復 run.py 檔案中的語法錯誤..." -ForegroundColor Yellow

# 讀取原始檔案
$content = Get-Content -Path $runPyPath -Encoding UTF8

# 修復第216行和第220行的重複參數問題
$fixedContent = @()
for ($i = 0; $i -lt $content.Length; $i++) {
    $line = $content[$i]

    # 修復包含重複參數的行
    if ($line -match "fstride=args\.fshape,\s*tshape=args\.tshape" -or $line -match "fstride=args\.fstride,\s*tshape=args\.tstride") {
        $line = $line -replace "fstride=args\.fshape,\s*tshape=args\.tshape", "fstride=args.fstride, tstride=args.tstride"
        $line = $line -replace "fstride=args\.fstride,\s*tshape=args\.tstride", "fstride=args.fstride, tstride=args.tstride"
    }

    $fixedContent += $line
}

# 寫回修復後的內容
$fixedContent | Set-Content -Path $runPyPath -Encoding UTF8

Write-Host "run.py 檔案已修復重複參數問題" -ForegroundColor Green

# 檢查語法是否正確
Write-Host "檢查 Python 語法..." -ForegroundColor Yellow
try {
    $result = python -m py_compile "C:\Users\Lin\Desktop\2_code\ssast_hub\ssast-main\src\run.py" 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "語法檢查通過！" -ForegroundColor Green
    } else {
        Write-Host "語法仍有問題: $result" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "Python 語法檢查出錯: $_" -ForegroundColor Red
    exit 1
}

# 設定 Docker 容器名稱
$DOCKER_IMAGE = "ssast_hub_image"

Write-Host "開始執行 SSAST 預訓練..." -ForegroundColor Yellow

# 執行 Docker 命令運行 SSAST 預訓練
docker run -it --rm `
    --gpus all `
    -v "D:\spectrogram_pt_name:/data/train" `
    -v "D:\val_spectrogram_pt_name:/data/val" `
    -v "C:\Users\Lin\Desktop\2_code\ssast_hub:/ssast" `
    $DOCKER_IMAGE `
    python /ssast/ssast-main/src/run.py --dataset precomputed `
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

Write-Host "SSAST 預訓練執行完成！" -ForegroundColor Green