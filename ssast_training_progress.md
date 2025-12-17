# SSAST 預訓練任務進度記錄

## 當前問題
在 Docker 環境中執行 SSAST 預訓練時遇到以下錯誤：
```
RuntimeError: CUDA error: no kernel image is available for execution on the device
CUDA kernel errors might be asynchronously reported at some other API call,so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1.
```

## 已確認的系統狀態
- 訓練集 `.pt` 檔案：`D:\spectrogram_pt_name` (160,782 個檔案)
- 驗證集 `.pt` 檔案：`D:\val_spectrogram_pt_name` (17,864 個檔案)  
- 訓練 JSON：`combined_train_data.json` (178,647 項目，160,783 匹配)
- 驗證 JSON：`fs_sound_list.json` (需確認匹配正確性)

## 已完成的修改
1. 修復了 `dataloader_pt_reader.py` 中的路徑處理邏輯
2. 確保 Windows 和 Linux 路徑格式正確處理
3. 創建了本地測試腳本 `test_pt_loading.py`
4. 創建了 CPU 兼容的運行腳本 `run_local_test.py`

## 待執行的行動

### 1. 本地 CPU 測試
```bash
# 啟動 Anaconda 環境
conda activate SSAST-NEW

# 運行測試腳本
cd C:\Users\Lin\Desktop\2_code\ssast_hub\ssast-main\src
python test_pt_loading.py
```

### 2. 本地訓練測試 (CPU)
```bash
# 使用較小的批次大小和訓練參數進行測試
python run_local_test.py --dataset precomputed \
--data-train "D:\spectrogram_pt_name" \
--data-val "D:\val_spectrogram_pt_name" \
--exp-dir "./exp/test_cpu_01" \
--label-csv "../class_labels_indices.csv" \
--lr 1e-4 --n-epochs 1 --batch-size 4 --save_model True \
--freqm 0 --timem 0 --mixup 0 --bal none \
--tstride 16 --fstride 16 --fshape 16 --tshape 16 \
--dataset_mean -7.4482 --dataset_std 2.4689 \
--target_length 1024 --num_mel_bins 128 \
--model_size base --mask_patch 400 --n-print-steps 10 \
--task pretrain_joint --lr_patience 2 --epoch_iter 100
```

### 3. Docker 修正版測試
```bash
# 在 Docker 中設置 CUDA_LAUNCH_BLOCKING 並使用較小的批次大小
docker run --gpus all -it --rm \
  --ipc=host \
  -v "C:\Users\Lin\Desktop\2_code\ssast_hub:/workspace/ssast_hub" \
  -v "D:\spectrogram_pt_name:/workspace/data/train" \
  -v "D:\val_spectrogram_pt_name:/workspace/data/val" \
  -e CUDA_LAUNCH_BLOCKING=1 \
  ssast_hub_image bash -c "
cd /workspace/ssast_hub/ssast-main/src
python -W ignore run.py --dataset precomputed \
--data-train '/workspace/data/train' \
--data-val '/workspace/data/val' \
--exp-dir '/workspace/ssast_hub/ssast-main/src/exp/test_docker_fixed' \
--label-csv '/workspace/ssast_hub/ssast-main/class_labels_indices.csv' \
--lr 1e-4 --n-epochs 1 --batch-size 4 --save_model True \
--freqm 0 --timem 0 --mixup 0 --bal none \
--tstride 16 --fstride 16 --fshape 16 --tshape 16 \
--dataset_mean -7.4482 --dataset_std 2.4689 \
--target_length 1024 --num_mel_bins 128 \
--model_size base --mask_patch 400 --n-print-steps 10 \
--task pretrain_joint --lr_patience 2 --epoch_iter 100
"
```

## 修正後的 dataloader_pt_reader.py 邏輯

確保路徑提取正確：

```python
# 使用正則表達式處理跨平台路徑
path_parts = re.split(r'[\\/]+', original_path)
filename = path_parts[-1]
filename_no_ext = os.path.splitext(filename)[0]
pt_path = os.path.join(self.data_dir, f"{filename_no_ext}.pt")
```

## 明天繼續的行動項目

- [ ] 執行本地 CPU 測試確認基本功能
- [ ] 修復可能的模型架構相容性問題
- [ ] 測試 Docker 中的 CUDA 相容性
- [ ] 根據測試結果調整訓練參數
- [ ] 執行完整的預訓練流程

## 預期結果
修復後的系統應該能：
1. ✅ 正確載入 .pt 檔案
2. ✅ 在 CPU 上成功運行基本訓練
3. ✅ 在 Docker 中解決 CUDA 相容性問題
4. ✅ 完成 SSAST 預訓練流程

---
**記錄時間**: 2025年12月16日  
**狀態**: 等待明天繼續執行測試  
**下次行動**: 運行本地 CPU 測試