# test_pt_loading.py
# 本地測試腳本，用於驗證 .pt 檔案讀取是否正常

import torch
import os
import sys
import json
import re
from torch.utils.data import DataLoader

# 將當前目錄添加到 Python 路徑
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataloader_pt_reader import PrecomputedDataset

def test_pt_loading():
    print("開始測試 .pt 檔案載入...")
    
    # 測試路徑
    train_dir = r"D:\spectrogram_pt_name"
    val_dir = r"D:\val_spectrogram_pt_name"
    train_json_path = r"C:\Users\Lin\Desktop\2_code\ssast_hub\ssast-main\combined_train_data.json"
    val_json_path = r"C:\Users\Lin\Desktop\2_code\ssast_hub\ssast-main\fs_sound_list.json"

    print(f"訓練資料集目錄: {train_dir}")
    print(f"驗證資料集目錄: {val_dir}")
    print(f"訓練 JSON 檔案: {train_json_path}")
    print(f"驗證 JSON 檔案: {val_json_path}")
    
    # 檢查檔案是否存在
    print(f"\n訓練目錄存在: {os.path.exists(train_dir)}")
    print(f"驗證目錄存在: {os.path.exists(val_dir)}")
    print(f"訓練 JSON 存在: {os.path.exists(train_json_path)}")
    print(f"驗證 JSON 存在: {os.path.exists(val_json_path)}")
    
    if os.path.exists(train_dir) and os.path.exists(train_json_path):
        print("\n創建訓練資料集...")
        train_dataset = PrecomputedDataset(train_dir, dataset_json_file=train_json_path)
        
        print(f"訓練資料集大小: {len(train_dataset)}")
        
        if len(train_dataset) > 0:
            print("測試讀取第一個樣本...")
            try:
                data, label = train_dataset[0]
                print("V 成功讀取樣本")
                print(f"  數據形狀: {data.shape}")
                print(f"  標籤形狀: {label.shape}")
                print(f"  數據類型: {data.dtype}")
                print(f"  標籤類型: {label.dtype}")

                # 測試讀取多個樣本
                print("\n測試讀取多個樣本...")
                for i in range(min(5, len(train_dataset))):
                    data, label = train_dataset[i]
                    if data.shape != torch.Size([1024, 128]):
                        print(f"! 樣本 {i} 數據形狀異常: {data.shape}")
                        break
                    if data.dtype != torch.float32:
                        print(f"! 樣本 {i} 數據類型異常: {data.dtype}")
                        break
                print("V 多個樣本讀取測試完成")

            except Exception as e:
                print(f"X 讀取樣本時出錯: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("! 訓練資料集為空")
    else:
        print("X 訓練資料集檔案不存在")

    if os.path.exists(val_dir) and os.path.exists(val_json_path):
        print("\n創建驗證資料集...")
        val_dataset = PrecomputedDataset(val_dir, dataset_json_file=val_json_path)

        print(f"驗證資料集大小: {len(val_dataset)}")

        if len(val_dataset) > 0:
            print("測試讀取驗證集第一個樣本...")
            try:
                data, label = val_dataset[0]
                print("V 成功讀取驗證樣本")
                print(f"  數據形狀: {data.shape}")
                print(f"  標籤形狀: {label.shape}")
            except Exception as e:
                print(f"X 讀取驗證樣本時出錯: {e}")
        else:
            print("! 驗證資料集為空")
    else:
        print("X 驗證資料集檔案不存在")

def check_json_pt_mapping():
    print("\n" + "="*50)
    print("檢查 JSON 與 .pt 檔案映射...")
    
    # 讀取 JSON 檔案
    json_path = r"C:\Users\Lin\Desktop\2_code\ssast_hub\ssast-main\combined_train_data.json"
    pt_dir = r"D:\spectrogram_pt_name"
    
    if os.path.exists(json_path):
        print(f"讀取 JSON 檔案: {json_path}")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"JSON 包含 {len(data['data'])} 個項目")
        
        # 檢查前幾個項目的檔名提取
        print("\n測試前5個項目的檔名提取邏輯:")
        for i in range(min(5, len(data['data']))):
            original_path = data['data'][i]['wav']
            # 使用與 dataloader 相同的邏輯
            path_parts = re.split(r'[\\/]+', original_path)
            filename = path_parts[-1]
            filename_no_ext = os.path.splitext(filename)[0]
            pt_filename = f"{filename_no_ext}.pt"
            pt_path = os.path.join(pt_dir, pt_filename)
            
            exists = os.path.exists(pt_path)
            print(f"  {i}: {original_path}")
            print(f"     提取檔名: {filename_no_ext}")
            print(f"     .pt 檔案: {pt_filename} (存在: {exists})")
            print("     ---")

def check_model_compatibility():
    print("\n" + "="*50)
    print("檢查模型相容性...")
    
    try:
        from models import ASTModel
        import torch.nn as nn
        
        print("V 成功導入 ASTModel")

        # 嘗試創建模型實例（使用 CPU）
        model = ASTModel(
            fshape=16,
            tshape=16,
            fstride=16,
            tstride=16,
            input_fdim=128,
            input_tdim=1024,
            model_size='base',
            pretrain_stage=True
        )

        print("V 成功創建 ASTModel 實例")
        print(f"模型參數數量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

        # 測試模型在 CPU 上的前向傳播
        test_input = torch.randn(1, 1024, 128)  # (batch_size, time, freq)
        model = nn.DataParallel(model)  # 使用 DataParallel

        print("測試模型前向傳播...")
        with torch.no_grad():
            output = model(test_input, 'pretrain_mpc', mask_patch=400, cluster=True)
            print(f"V 前向傳播成功，輸出形狀: {output[0].shape if isinstance(output, tuple) else output.shape}")

    except Exception as e:
        print(f"X 模型相容性測試失敗: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pt_loading()
    check_json_pt_mapping()
    check_model_compatibility()
    print("\n測試完成！")