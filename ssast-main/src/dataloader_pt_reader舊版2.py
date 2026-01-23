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
            # ---------------------------------------------------------
            # 1. 定義讀取單個 .pt 的內部函數 (保留你的容錯機制)
            # ---------------------------------------------------------
            def load_sample(idx):
                # A. 計算路徑 (保留你的原始邏輯)
                if hasattr(self, 'data_list'):
                    # 邊界檢查與重抽
                    if idx >= len(self.valid_indices):
                        new_idx = torch.randint(0, len(self.valid_indices), (1,)).item()
                        return load_sample(new_idx)
                    
                    actual_index = self.valid_indices[idx]
                    datum = self.data_list[actual_index] # 取得 JSON 中的該筆資料
                    
                    # 解析路徑
                    original_path = datum['wav']
                    path_parts = re.split(r'[\\/]+', original_path)
                    filename = path_parts[-1]
                    filename_no_ext = os.path.splitext(filename)[0]
                    pt_path = os.path.join(self.data_dir, f"{filename_no_ext}.pt")
                else:
                    # Fallback
                    pt_path = os.path.join(self.data_dir, f"{idx}.pt")
                    datum = {} # 如果沒有 data_list，可能無法做 Soft Label，需注意

                # B. 讀取 .pt
                try:
                    data_dict = torch.load(pt_path)
                    
                    # 檢查壞檔
                    if 'x' not in data_dict:
                        print(f"[Warning] 發現壞檔 (Missing 'x')，跳過: {pt_path}")
                        new_idx = torch.randint(0, len(self.valid_indices), (1,)).item()
                        return load_sample(new_idx)

                    fbank = data_dict['x'].to(torch.float32)
                    
                    # [重要修改] 這裡回傳 datum (包含 JSON 裡的標籤資訊) 而不是 .pt 裡的 'y'
                    return fbank, datum 

                except Exception as e:
                    print(f"[Error] 讀取失敗 {pt_path}: {e}")
                    new_idx = torch.randint(0, len(self.valid_indices), (1,)).item()
                    return load_sample(new_idx)

            # ---------------------------------------------------------
            # 2. 主邏輯：執行 Mixup 與 標籤生成
            # ---------------------------------------------------------
            
            # 載入主要樣本
            fbank1, datum1 = load_sample(index)
            
            # 決定是否 Mixup
            do_mixup = (self.mixup > 0 and random.random() < self.mixup)
            
            if do_mixup:
                # 隨機抽取第二個樣本
                mix_idx = random.randint(0, len(self.valid_indices)-1)
                fbank2, datum2 = load_sample(mix_idx)
                
                # 混合頻譜 (Mixup Lambda 通常服從 Beta 分布，這裡簡化沿用官方邏輯或 Random)
                # SSAST 官方通常用 random.random() 來決定 lambda 或是由 Beta 分布產生
                # 這裡假設你的 _wav2fbank 裡有處理 lambda，或是我們直接在這裡算
                mix_lambda = np.random.beta(10, 10) # 或是 random.random()
                
                # 簡單的線性混合 (如果是 log-mel, 有時會在 raw wave 混合, 但 feature level 混合也是常見做法)
                fbank = mix_lambda * fbank1 + (1 - mix_lambda) * fbank2
            else:
                fbank = fbank1
                mix_lambda = 1.0
                datum2 = None # 不需要第二個樣本

            # ---------------------------------------------------------
            # 3. 處理標籤 (將 JSON 資訊轉為 Tensor)
            # ---------------------------------------------------------
            label_indices = np.zeros(self.label_num)

            # 定義填入權重的 Helper function
            def add_labels(datum_info, lam):
                if datum_info is None: return
                
                # 狀況 A: List 分開格式 (你最新的 JSON)
                if 'weights' in datum_info:
                    for label_str, weight in zip(datum_info['labels'], datum_info['weights']):
                        if label_str in self.label_map:
                            label_indices[self.label_map[label_str]] += float(weight) * lam
                
                # 狀況 B: Dict 格式
                elif isinstance(datum_info['labels'], dict):
                    for label_key, weight in datum_info['labels'].items():
                        if label_key in self.label_map:
                            label_indices[self.label_map[label_key]] += float(weight) * lam
                
                # 狀況 C: 舊 String 格式
                elif isinstance(datum_info['labels'], str):
                    for label_str in datum_info['labels'].split(','):
                        if label_str in self.label_map:
                            label_indices[self.label_map[label_str]] += 1.0 * lam

            # 填入樣本 1 的標籤
            add_labels(datum1, mix_lambda)

            # 如果有 Mixup，填入樣本 2 的標籤
            if do_mixup and datum2 is not None:
                add_labels(datum2, 1.0 - mix_lambda)

            label_indices = torch.FloatTensor(label_indices)

            return fbank, label_indices
        

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