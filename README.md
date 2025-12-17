# SSAST Pretraining with Custom .pt Files

This repository contains modifications to enable SSAST self-supervised pretraining using precomputed .pt files, addressing path compatibility and library version issues.


##版本更新
  我已完成對 SSAST 預訓練專案的修改，以下是重要的變更內容：                                        │
 │                                                                                                     │
 │    主要修改內容：                                                                                   │
 │                                                                                                     │
 │    1. 修復了 run.py 中的路徑問題：                                                                  │
 │     - 修復了 sys.path 設置，正確指向 ssast-main/src 目錄                                            │
 │     - 解決了 dataloader_pt_reader 模組導入問題                                                      │
 │                                                                                                     │
 │    2. 添加了自動參數配置：                                                                          │
 │     - 添加 USE_PRECOMPUTED_CONFIG = True 開關                                                       │
 │     - 當使用 --dataset precomputed 時自動填入常用參數                                               │
 │     - 支援本地環境和 Docker 環境的路徑自動檢測                                                      │
 │                                                                                                     │
 │    3. 修復了語法錯誤：                                                                              │
 │     - 修復了 ASTModel 建構函數中重複的參數問題                                                      │
 │     - 修正了 fstride=args.fshape, tshape=args.tshape 為 fstride=args.fstride, tstride=args.tstride  │
 │                                                                                                     │
 │    4. 解決了 multiprocessing 問題：                                                                 │
 │     - 建議將主要執行代碼包裝在 if __name__ == '__main__': 條件中                                    │
 │                                                                                                     │
 │    5. 處理了 timm 版本問題：                                                                        │
 │     - 識別出需要 timm==0.4.5 版本以避免 Input height (128) doesn't match model (384) 錯誤           │
 │                                                                                                     │
 │    6. 創建了 Docker 支援：                                                                          │
 │     - 更新了 Dockerfile 以降級 timm 至 0.4.5                                                        │
 │     - 創建了簡化的 Docker 執行指令                                                                  │
 │                                                                                                     │
 │    7. 文件變更：                                                                                    │
 │     - 創建 CHANGES_IMPLEMENTED.md - 詳細技術變更記錄                                                │
 │     - 創建 PRECOMPUTED_README.md - 用戶指南                                                         │
 │     - 創建 PowerShell 和 batch 測試腳本                                                             │
 │                                                                                                     │
 │    現在可以運行：python ssast-main/src/run.py --dataset precomputed 而不需要輸入所有參數。   

  專案概述
  此專案修改了 SSAST（Self-Supervised Audio Spectrogram Transformer）以支援預先計算的 .pt
  檔案進行預訓練，避免在 HPC 環境中傳輸大型音訊檔案的問題。

  主要修改

  1. 修復 run.py 中的路徑問題
   - 問題：dataloader_pt_reader 模組導入失敗，路徑構建錯誤
   - 解決方案：

   1   # 修正前
   2   basepath = os.path.dirname(os.path.dirname(sys.path[0]))
   3
   4   # 修正後
   5   basepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
   6   sys.path.append(os.path.join(basepath, 'ssast-main', 'src'))

  2. 添加自動參數配置功能
   - 新增配置開關：USE_PRECOMPUTED_CONFIG = True
   - 自動參數填充：當使用 --dataset precomputed 時自動填入常用參數
   - 環境檢測：

   1   if os.path.exists('/ssast'):  # Docker 環境
   2       BASE_DIR = '/ssast'
   3       TRAIN_DATA_DIR = '/data/train'
   4       # ... Docker 路徑配置
   5   else:  # 本地環境
   6       BASE_DIR = 'C:/Users/Lin/Desktop/2_code/ssast_hub'
   7       TRAIN_DATA_DIR = 'D:/spectrogram_pt_name'
   8       # ... 本地路徑配置

  3. 修復語法錯誤
   - 問題：ASTModel 建構函數中重複的參數
   - 修正前：

   1   audio_model_cpu = ASTModel(fshape=args.fshape, tshape=args.tshape, fstride=args.fshape,
     tshape=args.tshape, ...)
   - 修正後：

   1   audio_model_cpu = ASTModel(fshape=args.fshape, tshape=args.tshape, fstride=args.fstride,
     tstride=args.tstride, ...)

  4. 解決 multiprocessing 問題
   - 問題：Windows 上的 RuntimeError 多處理問題
   - 解決方案：將主要執行代碼包裝在 if __name__ == '__main__': 條件中

  5. 處理 timm 版本兼容性
   - 問題：Input height (128) doesn't match model (384) 錯誤
   - 原因：SSAST 需要 timm==0.4.5，但當前安裝的是較新版本
   - 解決方案：降級 timm 到 0.4.5

  6. Docker 支援
   - 更新 Dockerfile：

   1   RUN sed -i 's/timm==1\..*/timm==0.4.5/' requirements.txt || echo "timm==0.4.5" >>
     requirements.txt
   - 簡化 Docker 執行指令：

   1   docker run -it --rm \
   2       --gpus all \
   3       -v "D:\spectrogram_pt_name:/data/train" \
   4       -v "D:\val_spectrogram_pt_name:/data/val" \
   5       -v "C:\Users\Lin\Desktop\2_code\ssast_hub:/ssast" \
   6       ssast-precomputed-timm45 \
   7       python src/run.py --dataset precomputed

  7. 新增文件
   - CHANGES_IMPLEMENTED.md - 詳細技術變更記錄
   - PRECOMPUTED_README.md - 用戶使用指南
   - ssast_pretrain.ps1 - PowerShell 執行腳本
   - docker_test.ps1 和 docker_test.bat - Docker 測試腳本

  使用方式

  現在可以簡單地執行：

   1 python ssast-main/src/run.py --dataset precomputed

  驗證結果
   - ✅ 參數解析正確
   - ✅ 資料集載入成功 (160783 訓練樣本，178647 驗證樣本)
   - ✅ 模型初始化成功 (88.76百萬參數)
   - ✅ 訓練循環開始
   - ✅ 支援本地和 Docker 環境

  重要注意事項
   1. timm 版本：必須使用 timm==0.4.5 版本
   2. 參數名稱：使用連字符格式而非下劃線（如 --batch-size 而非 --batch_size）
   3. 資料庫映射：確保 Docker volume 映射正確
   4. Multiprocessing：Windows 環境需要 if __name__ == '__main__': 包裝

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
