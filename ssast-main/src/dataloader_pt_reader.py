import torch
from torch.utils.data import Dataset, DataLoader
import os
import json
import re

class PrecomputedDataset(Dataset):
    def __init__(self, data_dir, dataset_json_file=None):
        """
        讀取預先計算好的 .pt 檔案 (包含 fbank 和 label)
        :param data_dir: 存放 .pt 檔案的資料夾路徑 (例如 "D:\\AS")
        :param dataset_json_file: JSON file path that contains the mapping (optional)
        """
        self.data_dir = data_dir
        self.config_path = os.path.join(data_dir, "dataset_config.pt")

        # If a JSON file is provided, create a mapping from index to filename
        if dataset_json_file and os.path.exists(dataset_json_file):
            with open(dataset_json_file, 'r') as fp:
                data_json = json.load(fp)
            self.data_list = data_json['data']
            self.length = len(self.data_list)
            print(f"Found {self.length} samples in JSON configuration.")

            # Verify which files actually exist and create a valid index mapping
            self.valid_indices = []
            for i in range(len(self.data_list)):
                original_path = self.data_list[i]['wav']
                # Extract filename properly by splitting on both \ and /
                path_parts = re.split(r'[\\/]+', original_path)
                filename = path_parts[-1]  # Get last part which should be the actual filename
                filename_no_ext = os.path.splitext(filename)[0]
                pt_path = os.path.join(self.data_dir, f"{filename_no_ext}.pt")
                if os.path.exists(pt_path):
                    self.valid_indices.append(i)

            print(f"Out of {self.length} JSON entries, {len(self.valid_indices)} .pt files exist.")

        elif os.path.exists(self.config_path):
            print(f"Loading config from {self.config_path}...")
            config = torch.load(self.config_path)
            self.length = config.get('total_samples', 0)
            self.audio_conf = config.get('audio_conf', {})
            print(f"Found {self.length} samples in configuration.")
            self.valid_indices = list(range(self.length))  # Assume all indices are valid if using config
        else:
            # 如果找不到設定檔，則掃描資料夾計算 .pt 檔案數量 (排除 config 檔)
            print("Config file not found, scanning directory...")
            files = [f for f in os.listdir(data_dir) if f.endswith('.pt') and f != "dataset_config.pt"]
            self.length = len(files)
            print(f"Scanned {self.length} .pt files.")
            self.valid_indices = list(range(self.length))  # All indices are valid when using directory scan

    def __getitem__(self, index):
        """
        讀取單個 .pt 檔案
        return:
            fbank: Float32 Tensor [T, F]
            label: Float32 Tensor [num_classes]
        """
        # Check if we have JSON mapping (filename-based naming)
        if hasattr(self, 'data_list'):
            if index >= len(self.valid_indices):
                raise IndexError(f"Index {index} is out of range for valid dataset with {len(self.valid_indices)} items")

            # Get the actual index in the original JSON list
            actual_index = self.valid_indices[index]

            # Use the JSON mapping to get the original filename
            # Handle both Windows and Unix style paths in the JSON
            original_path = self.data_list[actual_index]['wav']

            # Extract filename properly by splitting on both \ and /
            # Split by both forward slash and backslash to handle both Windows and Unix paths
            path_parts = re.split(r'[\\/]+', original_path)
            filename = path_parts[-1]  # Get last part which should be the actual filename
            filename_no_ext = os.path.splitext(filename)[0]
            pt_path = os.path.join(self.data_dir, f"{filename_no_ext}.pt")
        else:
            # Use index-based naming (fallback)
            pt_path = os.path.join(self.data_dir, f"{index}.pt")

        try:
            # 讀取字典
            data_dict = torch.load(pt_path)

            # 1. 取出 fbank 並轉回 float32 (模型訓練通常需要 float32)
            # 原始儲存為 float16 (Half) 以節省空間
            fbank = data_dict['x'].to(torch.float32)

            # 2. 取出 Label
            label = data_dict['y']

            # 確保 label 也是 float32 (若計算 Loss 需要)
            if label.dtype == torch.float16:
                label = label.to(torch.float32)

            return fbank, label

        except FileNotFoundError:
            # Handle error differently based on whether we have JSON mapping
            if hasattr(self, 'data_list'):
                # JSON case - we already checked that file exists, so this is unexpected
                actual_index = self.valid_indices[index] if index < len(self.valid_indices) else index
                # Get the original path and extract filename for error message
                original_path = self.data_list[actual_index]['wav']
                path_parts = re.split(r'[\\/]+', original_path)
                filename = path_parts[-1]
                filename_no_ext = os.path.splitext(filename)[0]
                constructed_path = os.path.join(self.data_dir, f"{filename_no_ext}.pt")

                print(f"Warning: File {constructed_path} not found for index {index} (actual_index {actual_index}). Skipping this sample.")
                print(f"  JSON entry: {self.data_list[actual_index]}")
            else:
                # Fallback case - index-based naming
                print(f"Warning: File {pt_path} not found for index {index}. Skipping this sample.")
            # This shouldn't happen since we already verified the file exists, but just in case
            # Use more generic tensor creation - we'll adjust dimensions based on model requirements if needed
            # Using common SSAST dimensions: [time=1024, freq=128] for spectrogram and appropriate label size
            # The label dimension should match the number of classes in your problem
            # If you're using audioset, it's typically 527 classes
            # But we should make this configurable or read from label CSV if possible
            return torch.zeros(1024, 128, dtype=torch.float32), torch.zeros(527, dtype=torch.float32)  # Default to 527 classes for AudioSet

    def get_available_files_count(self):
        """返回資料夾中可用的 .pt 檔案數量"""
        return len([f for f in os.listdir(self.data_dir) if f.endswith('.pt') and f != 'dataset_config.pt'])

    def __len__(self):
        # Return the number of valid samples (files that actually exist)
        if hasattr(self, 'valid_indices'):
            return len(self.valid_indices)
        return self.length

# ==========================================
#  使用範例 (Main)
# ==========================================
if __name__ == "__main__":
    # 設定你的資料夾路徑
    DATA_DIR = r"D:\spectrogram_pt"
    
    # 1. 實例化 Dataset
    dataset = PrecomputedDataset(DATA_DIR)
    
    # 2. 建立 DataLoader (模擬訓練時的批次讀取)
    # num_workers > 0 對於讀取大量小檔案很重要，可以加速
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    
    print("\nStarting check...")
    
    # 讀取一個 batch 來測試
    for i, (fbank_batch, label_batch) in enumerate(dataloader):
        print(f"Batch {i}:")
        print(f"  - fbank shape: {fbank_batch.shape}") # 預期: [Batch, T, F] -> [4, 1024, 128]
        print(f"  - label shape: {label_batch.shape}") # 預期: [Batch, Class_Num]
        print(f"  - fbank dtype: {fbank_batch.dtype}") # 預期: torch.float32
        
        # 只要測試第一筆就好，跳出迴圈
        break

    print("✅ 讀取測試完成")