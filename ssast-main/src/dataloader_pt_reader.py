import torch
from torch.utils.data import Dataset, DataLoader
import os

class PrecomputedDataset(Dataset):
    def __init__(self, data_dir):
        """
        讀取預先計算好的 .pt 檔案 (包含 fbank 和 label)
        :param data_dir: 存放 .pt 檔案的資料夾路徑 (例如 "D:\\AS")
        """
        self.data_dir = data_dir
        self.config_path = os.path.join(data_dir, "dataset_config.pt")
        
        # 1. 嘗試讀取設定檔以獲取總樣本數
        if os.path.exists(self.config_path):
            print(f"Loading config from {self.config_path}...")
            config = torch.load(self.config_path)
            self.length = config.get('total_samples', 0)
            self.audio_conf = config.get('audio_conf', {})
            print(f"Found {self.length} samples in configuration.")
        else:
            # 如果找不到設定檔，則掃描資料夾計算 .pt 檔案數量 (排除 config 檔)
            print("Config file not found, scanning directory...")
            files = [f for f in os.listdir(data_dir) if f.endswith('.pt') and f != "dataset_config.pt"]
            self.length = len(files)
            print(f"Scanned {self.length} .pt files.")

    def __getitem__(self, index):
        """
        讀取單個 .pt 檔案
        return: 
            fbank: Float32 Tensor [T, F]
            label: Float32 Tensor [num_classes]
        """
        # 組合路徑，例如 D:\AS\0.pt
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
            print(f"Warning: File {pt_path} not found. Returning zeros.")
            # 避免程式崩潰，回傳全零 (依需求可改為 raise Error)
            # 假設 shape 為 [1024, 128] 與 [527] (需視你的實際資料而定)
            # 怕傳0會洗掉先註解 #return torch.zeros(1024, 128), torch.zeros(1)

    def __len__(self):
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