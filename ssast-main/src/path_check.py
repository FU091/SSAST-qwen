import os
import re
import json

def check_path_extraction():
    print("Testing path extraction logic...")
    
    # 模擬 JSON 中的路徑
    test_paths = [
        "D:\\FS_SOUND\\Soundscape_Audio-20250428T180906Z-1-001\\Soundscape_Audio\\FS01_20250114_062700.wav",
        "D:\\GW_audio\\GWB01\\Data_20250204-20250303\\GWB01_20250217_091500.wav",
        "/unix/path/to/file.wav",
        "mixed\\path/with/both.wav"
    ]
    
    for path in test_paths:
        # 使用我們在 dataloader 中的邏輯
        path_parts = re.split(r'[\\/]+', path)
        filename = path_parts[-1]
        filename_no_ext = os.path.splitext(filename)[0]
        print(f"Original: {path}")
        print(f"Extracted filename: {filename_no_ext}")
        print("---")

def check_files_exist():
    print("Checking if required files exist...")
    
    # 檢查基本檔案是否存在
    json_path = r"../combined_train_data.json"
    abs_json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "combined_train_data.json")
    
    print(f"Looking for JSON file at: {abs_json_path}")
    print(f"JSON file exists: {os.path.exists(abs_json_path)}")
    
    if os.path.exists(abs_json_path):
        print("Checking first few entries in JSON...")
        try:
            with open(abs_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"Total entries: {len(data['data'])}")
            print("First 3 entries:")
            for i in range(min(3, len(data['data']))):
                entry = data['data'][i]
                print(f"  {i}: {entry}")
        except Exception as e:
            print(f"Error reading JSON: {e}")
    
    # 檢查 dataloader_pt_reader.py
    dataloader_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataloader_pt_reader.py")
    print(f"Dataloader file exists: {os.path.exists(dataloader_path)}")
    
    if os.path.exists(dataloader_path):
        with open(dataloader_path, 'r', encoding='utf-8') as f:
            content = f.read()
            print(f"Dataloader file size: {len(content)} characters")
            # 檢查是否包含我們的修改
            if 're.split(r\'[\\\\/]+\',' in content:
                print("✓ Contains our path fix")
            else:
                print("✗ Path fix not found in file")

if __name__ == "__main__":
    check_path_extraction()
    print("\n" + "="*50 + "\n")
    check_files_exist()