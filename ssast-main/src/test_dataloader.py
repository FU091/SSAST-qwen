# test_dataloader.py
import torch
import sys
import os
# 添加當前目錄到 Python 路徑
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataloader_pt_reader import PrecomputedDataset

def test_dataloader():
    print("Testing PrecomputedDataset...")
    
    # 測試訓練集路徑 (請替換為您本地實際的路徑)
    train_data_dir = r"D:\spectrogram_pt_name"
    train_json_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "combined_train_data.json")  # 相對於 src 目錄
    
    print(f"Data directory: {train_data_dir}")
    print(f"JSON file: {train_json_path}")
    
    if os.path.exists(train_json_path) and os.path.exists(train_data_dir):
        print("Files exist, creating dataset...")
        dataset = PrecomputedDataset(
            data_dir=train_data_dir,
            dataset_json_file=train_json_path
        )
        
        print(f"Dataset length: {len(dataset)}")
        print(f"Available files in directory: {dataset.get_available_files_count()}")
        
        if len(dataset) > 0:
            print("Testing first sample...")
            try:
                sample_data, sample_label = dataset[0]
                print(f"Sample data shape: {sample_data.shape}")
                print(f"Sample label shape: {sample_label.shape}")
                print("✓ First sample loaded successfully")
                
                # 測試隨機樣本
                if len(dataset) > 5:
                    sample_data, sample_label = dataset[5]
                    print(f"6th sample data shape: {sample_data.shape}")
                    print(f"6th sample label shape: {sample_label.shape}")
                    print("✓ Random sample loaded successfully")
                    
            except Exception as e:
                print(f"✗ Error loading sample: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("✗ Dataset is empty or no matching files found")
    else:
        print(f"✗ Files not found:")
        print(f"  JSON file exists: {os.path.exists(train_json_path)}")
        print(f"  Data directory exists: {os.path.exists(train_data_dir)}")
        if os.path.exists(train_data_dir):
            files = [f for f in os.listdir(train_data_dir) if f.endswith('.pt') and f != 'dataset_config.pt']
            print(f"  .pt files in directory: {len(files)}")

if __name__ == "__main__":
    test_dataloader()