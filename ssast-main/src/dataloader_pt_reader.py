import torch
from torch.utils.data import Dataset, DataLoader
import os
import json
import re
import numpy as np
import pandas as pd
import random

class PrecomputedDataset(Dataset):
    def __init__(self, dataset_json_file, audio_conf=None, data_dir=None, label_csv=None, mixup=0.0):
        """
        :param dataset_json_file: 訓練/驗證/測試用的 JSON 路徑
        :param audio_conf: 音訊設定 (包含 target_length 等)
        :param data_dir: 資料根目錄 (來自 run.py 的 --dataset_root)
        :param label_csv: 類別對照表 CSV (必須包含 index, display_name)
        :param mixup: Mixup 機率 (0.0 ~ 1.0)
        """
        self.data_dir = data_dir
        self.mixup = mixup
        self.audio_conf = audio_conf or {}
        
        # -------------------------------------------------------
        # 1. 讀取 Label Map (CSV)
        # -------------------------------------------------------
        # 尝試在多個可能的位置查找 CSV 文件
        csv_found = False
        possible_csv_paths = [
            label_csv,
            label_csv.replace('/finetune_stratified_final/', '/'),
            '/work/t113618009/ssast_hub/class_labels_indices.csv',
            '/ssast_hub/class_labels_indices.csv',
            '/work/t113618009/ssast_hub/class_labels_indices.csv',
            # 【新增】添加 librispeech 的標籤文件路徑
            '/work/t113618009/ssast_hub/ssast-main/src/prep_data/librispeech/class_labels_indices.csv'
        ]

        df = None
        for csv_path in possible_csv_paths:
            if csv_path and os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                    print(f"Found label CSV at: {csv_path}")
                    csv_found = True
                    break
                except Exception as e:
                    print(f"Could not read CSV at {csv_path}: {e}")
                    continue

        if csv_found and df is not None:
            # 建立 name -> index 的對照表
            # 注意：這裡需配合你的 CSV 欄位名稱，通常是 mid, display_name, 或 index
            # 這裡假設你的 CSV 至少有 'display_name' 欄位，且列的順序即為 index
            if 'display_name' in df.columns and 'mid' in df.columns:
                # 同時支持 mid 和 display_name 映射
                # 優先使用 mid 作為鍵，因為這是 AudioSet 標準
                self.label_map = {}
                for idx, row in df.iterrows():
                    # 添加 mid 映射
                    if pd.notna(row['mid']):
                        self.label_map[str(row['mid'])] = int(row['index'])
                    # 添加 display_name 映射
                    if pd.notna(row['display_name']):
                        self.label_map[str(row['display_name'])] = int(row['index'])
            elif 'display_name' in df.columns:
                self.label_map = {str(name): i for i, name in enumerate(df['display_name'])}
            elif 'mid' in df.columns: # Audioset 格式
                self.label_map = {str(name): i for i, name in enumerate(df['mid'])}
            else:
                # Fallback: 如果沒有 header，假設第一欄是名稱
                self.label_map = {str(row[0]): i for i, row in df.iterrows()}

            self.label_num = len(set(self.label_map.values()))  # 使用集合去重後計算實際類別數
            print(f"Loaded Label Map: {self.label_num} classes with {len(self.label_map)} mappings.")
            # 保存 DataFrame 以便後續使用
            self.df = df
        else:
            raise ValueError(f"Label CSV not found in any of these locations: {possible_csv_paths}")

        # -------------------------------------------------------
        # 2. 讀取 Dataset JSON
        # -------------------------------------------------------
        json_found = False
        possible_json_paths = [
            dataset_json_file,
            dataset_json_file.replace('/finetune_stratified_final/', '/'),
            dataset_json_file.replace('/work/t113618009/ssast_hub/', '/ssast_hub/'),
            dataset_json_file.replace('/work/t113618009/', '/')
        ]

        data_json = None
        for json_path in possible_json_paths:
            if json_path and os.path.exists(json_path):
                try:
                    with open(json_path, 'r', encoding='utf-8') as fp:
                        data_json = json.load(fp)
                    print(f"Found JSON at: {json_path}")
                    json_found = True
                    break
                except Exception as e:
                    print(f"Could not read JSON at {json_path}: {e}")
                    continue

        if json_found and data_json is not None:
            self.data_list = data_json['data']
            print(f"Found {len(self.data_list)} samples in JSON: {json_path}")
        else:
            raise ValueError(f"JSON file not found in any of these locations: {possible_json_paths}")

        # -------------------------------------------------------
        # 3. 驗證檔案存在性 (Valid Indices)
        # -------------------------------------------------------
        self.valid_indices = []
        missing_count = 0
        
        for i, item in enumerate(self.data_list):
            # 決定檔案路徑：
            # 策略 A: 如果有給 data_dir，則組合 data_dir + 檔名 (忽略 JSON 裡的路徑部分)
            # 策略 B: 如果沒給 data_dir，則直接用 JSON 裡的完整路徑
            
            json_path = item['wav']
            filename = os.path.basename(json_path) # 只取檔名 xxx.pt
            
            if self.data_dir:
                final_path = os.path.join(self.data_dir, filename)
            else:
                final_path = json_path
            
            # 檢查是否存在
            if os.path.exists(final_path):
                self.valid_indices.append(i)
                # 將計算出的最終路徑存回記憶體中的 list，加速 __getitem__
                self.data_list[i]['_final_path'] = final_path
            else:
                missing_count += 1
                if missing_count < 5:
                    print(f"[Warning] File missing: {final_path}")

        print(f"Final valid samples: {len(self.valid_indices)} (Missing: {missing_count})")
        if len(self.valid_indices) == 0:
            raise RuntimeError("No valid files found! Check your paths.")

    def __len__(self):
        return len(self.valid_indices)

    def _get_single_item(self, index):
        """ 讀取單個檔案的 Helper function """
        real_index = self.valid_indices[index]
        datum = self.data_list[real_index]
        pt_path = datum['_final_path'] # 使用在 __init__ 預算好的路徑

        try:
            # 讀取 .pt
            data_dict = torch.load(pt_path)
            
            # 檢查 'x' 是否存在 (spectrogram)
            if 'x' not in data_dict:
                # 若壞檔，隨機重抽一個
                return self._get_single_item(random.randint(0, len(self.valid_indices)-1))
            
            fbank = data_dict['x'] # Shape: [Time, Freq]
            
            # 確保是 float32
            if isinstance(fbank, np.ndarray):
                fbank = torch.from_numpy(fbank).float()
            else:
                fbank = fbank.float()

            # 處理長度 (截斷或補零) - 根據 audio_conf
            target_length = self.audio_conf.get('target_length', 1024) 
            # 如果你的 .pt 已經是固定長度，這段可視情況保留或移除
            n_frames = fbank.shape[0]
            if n_frames > target_length:
                fbank = fbank[:target_length, :] # Crop
            elif n_frames < target_length:
                # Padding (用 0 補)
                pad_len = target_length - n_frames
                fbank = torch.cat((fbank, torch.zeros(pad_len, fbank.shape[1])), dim=0)
            
            return fbank, datum
            
        except Exception as e:
            print(f"Error loading {pt_path}: {e}")
            return self._get_single_item(random.randint(0, len(self.valid_indices)-1))

    def __getitem__(self, index):
        # 1. 讀取主要樣本
        fbank1, datum1 = self._get_single_item(index)

        # 2. 準備標籤容器
        label_indices = np.zeros(self.label_num)

        # 3. 處理 Mixup
        mix_lambda = 1.0
        datum2 = None

        # 使用 self.audio_conf 中的模式信息來判斷是否為訓練模式
        # 如果 audio_conf 中沒有 'mode'，默認為訓練模式
        is_training = self.audio_conf.get('mode', 'train') in ['train', 'training']

        if self.mixup > 0 and is_training: # 只在訓練模式且設定 mixup 時啟用
             # 改進：如果 mixup > 0，我們就執行 mixup (訓練時傳入 >0，測試時傳入 0)
             if random.random() < self.mixup:
                 mix_idx = random.randint(0, len(self.valid_indices)-1)
                 fbank2, datum2 = self._get_single_item(mix_idx)

                 mix_lambda = np.random.beta(10, 10)
                 fbank1 = mix_lambda * fbank1 + (1 - mix_lambda) * fbank2
        
        # 4. 填入標籤 (Helper Function)
        def add_labels(info, lam):
            if not info: return

            # 【修改】支援軟標籤格式，帶權重的標籤
            # 格式1: "labels": "/m/09x0r" (單一標籤字符串)
            # 格式2: "labels": ["/m/09x0r"] (標籤列表)
            # 格式3: {'labels': ["/m/09x0r", "/m/0dgw0r"], 'weights': [0.8, 0.2]} (帶權重的軟標籤格式)

            labels = info.get('labels', [])
            weights = info.get('weights', [])

            # 處理不同格式的 labels
            if isinstance(labels, str):
                # 如果是單一標籤字符串，轉換為列表
                labels = [labels]
            elif not isinstance(labels, list):
                # 如果不是字符串也不是列表，跳過
                return

            # 如果沒有 weights，預設全為 1.0
            if not weights:
                weights = [1.0] * len(labels)

            # 確保 labels 和 weights 長度相同
            if len(labels) != len(weights):
                print(f"[Warning] Mismatch between labels and weights length: {len(labels)} vs {len(weights)}")
                min_len = min(len(labels), len(weights))
                labels = labels[:min_len]
                weights = weights[:min_len]

            for lbl, w in zip(labels, weights):
                # 尝試匹配標籤到 label_map
                if lbl in self.label_map:
                    idx = self.label_map[lbl]
                    label_indices[idx] += w * lam
                else:
                    # 如果找不到完全匹配，嘗試通過 display_name 或 mid 匹配
                    found_match = False
                    for csv_key, csv_idx in self.label_map.items():
                        # 檢查是否與 CSV 中的某個鍵匹配
                        if lbl == csv_key:
                            idx = csv_idx
                            label_indices[idx] += w * lam
                            found_match = True
                            break

                    if not found_match:
                        print(f"[Warning] Label '{lbl}' not found in label map, skipping.")

        add_labels(datum1, mix_lambda)
        if datum2:
            add_labels(datum2, 1.0 - mix_lambda)

        # 轉為 Tensor
        label_indices = torch.FloatTensor(label_indices)

        # 【新增】對於多標籤分類，將標籤二值化以適應 sklearn 的要求
        # 如果標籤中有非0/1的值，將其轉換為二進制格式（用於驗證階段）
        # 但保留原始軟標籤用於訓練損失計算
        if not is_training:  # 驗證模式下，將軟標籤轉為二進制
            label_indices = (label_indices > 0).float()

        return fbank1, label_indices

# ==========================================
#  本地測試區塊 (模擬 HPC 環境)
# ==========================================
if __name__ == "__main__":
    # 模擬你的 run.py 傳入參數
    # 注意：在 Windows 本機測試時，路徑要改成你本機的
    # 在 HPC 上跑這段 __main__ 可能會報錯，因為它是用來 Debug 的
    
    # 假設我們在 Windows 本機測試，請手動建立一個 dummy csv 和 json
    print("--- Testing Dataloader ---")
    
    # 這裡僅供你在本機 Run 測試，實際 HPC 會由 run.py 呼叫 Class
    pass