# test_dataloader_path.py
import torch
import os
import sys
import importlib.util
import traceback

def test_dataloader_path_handling():
    print("=== 測試 dataloader_pt_reader.py 路徑處理 ===")

    # 1. 處理 import 問題 (ssast-main 包含 "-" 無法直接用 import 關鍵字)
    module_path = os.path.abspath(r"ssast-main/src/dataloader_pt_reader.py")
    if not os.path.exists(module_path):
        print(f"❌ 找不到 dataloader 腳本: {module_path}")
        return False
        
    try:
        spec = importlib.util.spec_from_file_location("dataloader_pt_reader", module_path)
        dataloader_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dataloader_module)
        PrecomputedDataset = dataloader_module.PrecomputedDataset
        print("✅ 成功載入 PrecomputedDataset 類別")
    except Exception as e:
        print(f"❌ 載入模組失敗: {e}")
        return False

    # 測試訓練資料集路徑
    train_path = r"D:\spectrogram_pt_name"
    train_json_path = r"C:\Users\Lin\Desktop\2_code\ssast_hub\combined_train_data.json"

    print(f"測試路徑 - 資料目錄: {train_path}")
    print(f"測試路徑 - JSON: {train_json_path}")

    if os.path.exists(train_path):
        print(f"✅ 訓練資料目錄存在")
    else:
        print(f"❌ 訓練資料目錄不存在: {train_path}")
        return False

    if os.path.exists(train_json_path):
        print(f"✅ 訓練 JSON 文件存在")
    else:
        print(f"❌ 訓練 JSON 文件不存在: {train_json_path}")
        return False

    # 測試創建 dataset
    try:
        dataset = PrecomputedDataset(
            data_dir=train_path,
            dataset_json_file=train_json_path
        )
        print(f"✅ PrecomputedDataset 創建成功，長度: {len(dataset)}")

        # 測試讀取第一個樣本
        if len(dataset) > 0:
            fbank, label = dataset[0]
            print(f"✅ 樣本讀取成功 - fbank shape: {fbank.shape}, label shape: {label.shape}")
            print(f"✅ 樣本類型 - fbank: {fbank.dtype}, label: {label.dtype}")
            return True
        else:
            print("⚠️ 資料集為空")
            return False

    except Exception as e:
        print(f"❌ 執行錯誤: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_dataloader_path_handling()
    if success:
        print("\n🎉 dataloader 路徑處理測試成功！")
    else:
        print("\n💥 dataloader 路徑處理測試失敗！")