import torch
from torch.utils.data import Dataset, DataLoader
import os
import json
import re

class PrecomputedDataset(Dataset):
    def __init__(self, data_dir=None, dataset_json_file=None, audio_conf=None):
        """
        讀取預先計算好的 .pt 檔案 (包含 fbank 和 label)
        :param data_dir: 存放 .pt 檔案的資料夾路徑 (例如 "D:\\AS")
        :param dataset_json_file: JSON file path that contains the mapping (optional)
        :param audio_conf: audio configuration dictionary
        """
        # If data_dir is None but passed in audio_conf, use it from audio_conf (backward compatibility)
        if data_dir is None and audio_conf is not None and 'data_dir' in audio_conf:
            self.data_dir = audio_conf['data_dir']
        else:
            self.data_dir = data_dir

        if self.data_dir is None:
            # If no data_dir provided, try to infer from the JSON file directory
            if dataset_json_file and os.path.exists(dataset_json_file):
                self.data_dir = os.path.dirname(dataset_json_file)
            else:
                raise ValueError("Either data_dir must be provided or dataset_json_file must exist for directory inference")

        self.config_path = os.path.join(self.data_dir, "dataset_config.pt")

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
            files = [f for f in os.listdir(self.data_dir) if f.endswith('.pt') and f != "dataset_config.pt"]
            self.length = len(files)
            print(f"Scanned {self.length} .pt files.")
            self.valid_indices = list(range(self.length))  # All indices are valid when using directory scan
            
    def __getitem__(self, index):
        """
        [最終版] 讀取單個 .pt 檔案 (包含壞檔自動替補機制)
        """
        # --- 1. 計算檔案路徑 ---
        if hasattr(self, 'data_list'):
            # 邊界檢查
            if index >= len(self.valid_indices):
                new_index = torch.randint(0, len(self.valid_indices), (1,)).item()
                return self.__getitem__(new_index)
                
            actual_index = self.valid_indices[index]
            original_path = self.data_list[actual_index]['wav']
            path_parts = re.split(r'[\\/]+', original_path)
            filename = path_parts[-1]
            filename_no_ext = os.path.splitext(filename)[0]
            pt_path = os.path.join(self.data_dir, f"{filename_no_ext}.pt")
        else:
            pt_path = os.path.join(self.data_dir, f"{index}.pt")

        # --- 2. 讀取與自動替補 ---
        try:
            data_dict = torch.load(pt_path)
            
            # 檢查是否包含關鍵 Key 'x'
            if 'x' not in data_dict:
                # 發現壞檔！印出警告並紀錄
                print(f"[Warning] 發現壞檔 (Missing 'x')，正在跳過: {pt_path}")
                print(f"   -> 檔案內現有的 Keys: {list(data_dict.keys())}")
                
                # 自動重抽：隨機選另一個 index 遞迴呼叫
                new_index = torch.randint(0, len(self), (1,)).item()
                return self.__getitem__(new_index)

            # 正常讀取
            fbank = data_dict['x'].to(torch.float32)

            # Label 處理 (容錯)
            if 'y' in data_dict:
                label = data_dict['y']
            elif 'label' in data_dict:
                label = data_dict['label']
            else:
                # 如果沒有 label，跳過或給 dummy (這裡選擇跳過比較保險)
                print(f"[Warning] 發現壞檔 (Missing label)，正在跳過: {pt_path}")
                new_index = torch.randint(0, len(self), (1,)).item()
                return self.__getitem__(new_index)

            # 格式轉換
            if hasattr(label, 'dtype') and label.dtype == torch.float16:
                label = label.to(torch.float32)

            return fbank, label

        except Exception as e:
            # 捕捉檔案損壞、讀取權限等所有錯誤
            print(f"[Read Error] 無法讀取檔案，正在替補: {pt_path}")
            print(f"   -> 原因: {e}")
            
            # 隨機重抽
            new_index = torch.randint(0, len(self), (1,)).item()
            return self.__getitem__(new_index)
        

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
    DATA_DIR = r"D:\spectrogram_pt_name"
    
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