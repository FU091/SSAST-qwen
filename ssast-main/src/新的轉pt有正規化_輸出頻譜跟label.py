import torch
from tqdm import tqdm
from dataloader import AudioDataset
import os
import gc

# --- 1. 設定參數 ---
audio_conf = {
    "mode": "ssl",
    "num_mel_bins": 128,
    "target_length": 1024,
    "freqm": 0,
    "timem": 0,
    "mixup": 0,
    "noise": False,
    "dataset": "audioset",
    "mean": 7.4482,
    "std": 2.4689,
    "skip_norm": False  #(不跳過表示有執行正規化)
}

PROJECT_ROOT = os.getcwd()
print(f"當前目錄為: {PROJECT_ROOT}")

# 設定輸出資料夾 (請確保此路徑所在的硬碟有足夠空間)
# 預計大小：17萬筆 x 250KB ≈ 40~50 GB
OUTPUT_DIR = r"D:\spectrogram_pt_name"
os.makedirs(OUTPUT_DIR, exist_ok=True)

label_csv = os.path.join(PROJECT_ROOT, "class_labels_indices.csv")
dataset_json_file = os.path.join(PROJECT_ROOT, "combined_train_data.json")

# --- 2. 初始化 Dataset ---
# 這裡使用原始的 AudioDataset 來負責讀取音檔、正規化、Masking
print("正在初始化 Dataset...")
dataset = AudioDataset(dataset_json_file, audio_conf, label_csv)
print(f"Dataset size: {len(dataset)}")

print(f"開始處理... 檔案將儲存於: {OUTPUT_DIR}")
print("策略: 逐筆處理 -> 轉 float16 -> 個別存檔 -> 釋放記憶體")

# --- 3. 執行迴圈 (Process -> Save -> Release) ---
success_count = 0
error_count = 0

# 使用 tqdm 顯示進度
for i in tqdm(range(len(dataset))):
    try:
        # 1. 取得資料 (呼叫 dataloader 的 __getitem__)
        # fbank shape: [1024, 128] (Float32)
        # label shape: [Class_Num] (通常是 Float32 或 Long)
        fbank, label = dataset[i]

        # 2. 取得原始檔名 !!! 關鍵修改 !!!
        # dataset.data 是一個 list，裡面存著 json 的內容
        record = dataset.data[i]
        
        # 假設 json 裡存路徑的 key 叫 'wav' (這是 SSAST 的預設)
        # 如果你的 json key 是 'file_path'，請改成 record['file_path']
        original_path = record['wav'] 
        
        # 提取檔名: "C:/Data/dog.wav" -> "dog.wav" -> "dog"
        filename_with_ext = os.path.basename(original_path)
        filename_no_ext = os.path.splitext(filename_with_ext)[0]


        
        # 2. 轉換為 Float16 (節省一半空間)
        fbank_half = fbank.to(torch.float16)
        
        # 如果 label 也是 tensor 且是 float 類型，也可以轉 float16 (視需求而定，通常 label 佔用空間很小，不轉也沒關係)
        # label = label.to(torch.float16) 
        
        # 3. 建立儲存內容
        # 我們將 X 和 Y 存再一起，這樣訓練時讀取一個檔案就有 data 和 label
        save_dict = {
            "x": fbank_half, 
            "y": label
        }
        
        # 4. 決定檔名 (使用索引 i 命名，例如 0.pt, 1.pt)
        #save_path = os.path.join(OUTPUT_DIR, f"{i}.pt")
        save_path = os.path.join(OUTPUT_DIR, f"{filename_no_ext}.pt")
        
        # 5. 存檔
        torch.save(save_dict, save_path)
        
        success_count += 1

        # 6. 釋放記憶體 (雖然 Python 會自動回收，但在大量迴圈顯式刪除更保險)
        del fbank
        del fbank_half
        del label
        del save_dict

    except Exception as e:
        error_count += 1
        print(f"\nError processing index {i}: {e}")
        continue

# 7. 儲存設定檔 (方便之後訓練時讀取 mean/std)
config_save_path = os.path.join(OUTPUT_DIR, "dataset_config.pt")
torch.save({
    "audio_conf": audio_conf,
    "total_samples": len(dataset),    
    "success_count": success_count
}, config_save_path)

print("\n" + "="*30)
print(f"✅ 處理完成！")
print(f"成功: {success_count}")
print(f"失敗: {error_count}")
print(f"資料存放於: {OUTPUT_DIR}")
print(f"設定檔存放於: {config_save_path}")
print("="*30)