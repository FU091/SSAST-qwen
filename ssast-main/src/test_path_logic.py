import os
import sys
import re

def test_path_logic():
    """
    測試路徑處理邏輯是否正確
    """
    print("測試路徑處理邏輯...")
    
    # 測試從 JSON 中提取檔案名稱的邏輯
    test_paths = [
        "D:\\FS_SOUND\\Soundscape_Audio-20250428T180906Z-1-001\\Soundscape_Audio\\FS01_20250114_062700.wav",
        "D:\\GW_audio\\GWB01\\Data_20250204-20250303\\GWB01_20250217_091500.wav",
        "D:\\GW_audio\\GW03\\Data_20250402-20250416\\GW03_20250406_121500.wav",
        "/unix/path/to/file.wav",
        "mixed\\path/with/both.wav"
    ]
    
    print("測試路徑提取邏輯:")
    for path in test_paths:
        # 使用與 dataloader 相同的邏輯
        path_parts = re.split(r'[\\/]+', path)
        filename = path_parts[-1]  # 獲取最後一部分
        filename_no_ext = os.path.splitext(filename)[0]  # 去掉副檔名
        print(f"  原路徑: {path}")
        print(f"  提取檔名: {filename_no_ext}")
        print(f"  構造 .pt 檔案名稱: {filename_no_ext}.pt")
        print("  ---")
    
    print("\n✓ 路徑提取邏輯測試完成")

def check_files():
    """
    檢查您的檔案是否存在
    """
    print("\n檢查檔案是否存在...")
    
    # 檢查主要檔案
    base_path = r"C:\Users\Lin\Desktop\2_code\ssast_hub\ssast-main"
    json_path = os.path.join(base_path, "combined_train_data.json")
    dataloader_path = os.path.join(base_path, "src", "dataloader_pt_reader.py")
    
    print(f"JSON file exists: {os.path.exists(json_path)}")
    print(f"Dataloader file exists: {os.path.exists(dataloader_path)}")
    
    # 檢查 .pt 資料夾
    train_pt_path = r"D:\spectrogram_pt_name"
    val_pt_path = r"D:\val_spectrogram_pt_name"
    
    print(f"Training .pt directory exists: {os.path.exists(train_pt_path)}")
    if os.path.exists(train_pt_path):
        pt_files = [f for f in os.listdir(train_pt_path) if f.endswith('.pt') and f != 'dataset_config.pt']
        print(f"  Found {len(pt_files)} .pt files in training directory")
        print(f"  First few files: {pt_files[:5] if len(pt_files) > 0 else 'None'}")
    
    print(f"Validation .pt directory exists: {os.path.exists(val_pt_path)}")
    if os.path.exists(val_pt_path):
        val_pt_files = [f for f in os.listdir(val_pt_path) if f.endswith('.pt') and f != 'dataset_config.pt']
        print(f"  Found {len(val_pt_files)} .pt files in validation directory")
        print(f"  First few files: {val_pt_files[:5] if len(val_pt_files) > 0 else 'None'}")

if __name__ == "__main__":
    test_path_logic()
    check_files()
    print("\n測試完成！")