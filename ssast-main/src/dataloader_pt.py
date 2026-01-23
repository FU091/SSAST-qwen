# -*- coding: utf-8 -*-
# @Time    : 6/19/21 12:23 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : dataloader_pt.py

# Modified version to support precomputed .pt files and JSON Soft Labels
import csv
import json
import numpy as np
import torch
import torch.nn.functional
from torch.utils.data import Dataset
import random
import os
import re

def make_index_dict(label_csv):
    """
    讀取 class_labels_indices.csv
    回傳: { 'display_name': index, 'mid': index }
    這樣無論 JSON 裡寫的是 '蛙' (display_name) 還是 '/m/09x0r' (mid) 都能找到 index
    """
    index_lookup = {}
    with open(label_csv, 'r', encoding='utf-8') as f:
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            # 確保 index 是整數
            idx = int(row['index'])
            # 支援透過 display_name 查詢 (例如: "蛙" -> 0)
            if 'display_name' in row:
                index_lookup[row['display_name']] = idx
            # 支援透過 mid 查詢
            if 'mid' in row:
                index_lookup[row['mid']] = idx
    return index_lookup

class AudioDataset(Dataset):
    def __init__(self, dataset_json_file, audio_conf, label_csv=None):
        """
        Dataset that loads precomputed .pt files containing fbank and labels
        :param dataset_json_file: Path to json file with audio paths and labels
        :param audio_conf: Dictionary containing the audio loading and preprocessing settings
        :param label_csv: Path to csv file with class labels
        """
        self.datapath = dataset_json_file
        self.audio_conf = audio_conf
        self.data_dir = None
        
        # 1. 設定資料根目錄 (優先權: audio_conf > JSON路徑推斷)
        if 'data_dir' in audio_conf:
            self.data_dir = audio_conf['data_dir']
        elif 'data_path' in audio_conf:
            self.data_dir = audio_conf['data_path']
        else:
            self.data_dir = os.path.dirname(dataset_json_file)
            print(f"Warning: data_dir not specified, inferring from JSON path: {self.data_dir}")
        
        print('---------------the {:s} dataloader---------------'.format(self.audio_conf.get('mode')))
        
        # 2. 讀取 JSON
        with open(dataset_json_file, 'r', encoding='utf-8') as fp:
            data_json = json.load(fp)
        self.data = data_json['data']
        
        # 3. 驗證檔案是否存在 (使用 Smart Path Logic)
        self.valid_indices = []
        self.pt_files = []
        
        # 為了加速，只印出前幾個找不到的警告
        miss_count = 0
        
        for i, item in enumerate(self.data):
            original_path = item['wav']
            # [Smart Logic] 無論 JSON 是絕對還是相對路徑，都強制只取檔名
            path_parts = re.split(r'[\\/]+', original_path)
            filename = path_parts[-1]
            filename_no_ext = os.path.splitext(filename)[0]
            
            # 組合出正確的路徑
            pt_path = os.path.join(self.data_dir, f"{filename_no_ext}.pt")
            
            if os.path.exists(pt_path):
                self.valid_indices.append(i)
                self.pt_files.append(pt_path)
            else:
                miss_count += 1
                if miss_count <= 5:
                    print(f"Warning: .pt file not found: {pt_path}")
        
        print(f"Out of {len(self.data)} JSON entries, {len(self.valid_indices)} .pt files exist.")
        if miss_count > 5:
            print(f"... and {miss_count - 5} more files missing.")
        
        if len(self.valid_indices) == 0:
            raise ValueError("No .pt files found! Please check data_dir and file names.")

        # 4. 讀取標籤設定 (Label CSV)
        if label_csv is not None:
            self.index_dict = make_index_dict(label_csv)
            self.label_num = len(set(self.index_dict.values())) # 計算唯一類別數
            print('number of classes is {:d}'.format(self.label_num))
        else:
            self.index_dict = {}
            self.label_num = 0

        # 5. 其他設定 (保留原樣)
        self.melbins = self.audio_conf.get('num_mel_bins', 128)
        self.target_length = self.audio_conf.get('target_length', 1024)
        self.freqm = self.audio_conf.get('freqm', 0)
        self.timem = self.audio_conf.get('timem', 0)
        print('now using following mask: {:d} freq, {:d} time'.format(self.freqm, self.timem))
        self.mixup = self.audio_conf.get('mixup', 0.0)
        print('now using mix-up with rate {:f}'.format(self.mixup))
        self.dataset = self.audio_conf.get('dataset', 'audioset')
        
        self.norm_mean = self.audio_conf.get('mean')
        self.norm_std = self.audio_conf.get('std')
        self.skip_norm = self.audio_conf.get('skip_norm', False)
        
        if self.skip_norm:
            print('now skip normalization.')
        else:
            print('use dataset mean {:.3f} and std {:.3f} to normalize.'.format(self.norm_mean, self.norm_std))
        
        self.noise = self.audio_conf.get('noise', False)
        if self.noise == True:
            print('now use noise augmentation')

    def __getitem__(self, index):
        """
        returns: fbank, label
        """
        # 取得實際的 index
        actual_index = self.valid_indices[index]
        pt_path = self.pt_files[index]
        datum = self.data[actual_index] # 取得 JSON 裡的該筆資料 (包含 labels, weights)
        
        try:
            # 1. 讀取 .pt (只取 x / fbank)
            data_dict = torch.load(pt_path)
            fbank = data_dict['x']
            
            # 格式轉換
            if fbank.dtype == torch.float16:
                fbank = fbank.to(torch.float32)

            # 2. 建構 Label Tensor (從 JSON 讀取，而非從 .pt 讀取)
            label = torch.zeros(self.label_num, dtype=torch.float32)
            
            if 'labels' in datum:
                # 處理 labels (List) 和 weights (List)
                lbls = datum['labels']
                wgts = datum.get('weights', [1.0] * len(lbls)) # 如果沒有 weights，預設全為 1.0

                # 確保是列表
                if isinstance(lbls, str): lbls = [lbls]
                if isinstance(wgts, (int, float)): wgts = [wgts]

                for lbl, w in zip(lbls, wgts):
                    # 查找 Index (支援中文名稱或 MID)
                    if lbl in self.index_dict:
                        idx = self.index_dict[lbl]
                        label[idx] = float(w)
            
            # 3. Masking (保留原樣)
            if self.audio_conf.get('mode') == 'train' and (self.freqm > 0 or self.timem > 0):
                fbank = self.apply_masking(fbank)
            
            # 4. Noise Augmentation (保留原樣)
            if self.noise == True:
                fbank = fbank + torch.rand(fbank.shape[0], fbank.shape[1]) * np.random.rand() / 10
                fbank = torch.roll(fbank, np.random.randint(-10, 10), 0)
                
        except Exception as e:
            print(f"Error loading {pt_path}: {e}")
            fbank = torch.zeros(self.target_length, self.melbins, dtype=torch.float32)
            label = torch.zeros(self.label_num, dtype=torch.float32)
        
        return fbank, label

    def apply_masking(self, fbank):
        """
        Apply frequency and time masking
        """
        fbank = torch.transpose(fbank, 0, 1)
        fbank = fbank.unsqueeze(0)
        
        if self.freqm > 0:
            freq_len = fbank.shape[2]
            if self.freqm < freq_len:
                freq_start = torch.randint(0, freq_len - self.freqm, (1,))
                fbank[0, :, freq_start:freq_start + self.freqm] = 0
        
        if self.timem > 0:
            time_len = fbank.shape[3]
            if self.timem < time_len:
                time_start = torch.randint(0, time_len - self.timem, (1,))
                fbank[0, :, :, time_start:time_start + self.timem] = 0
        
        fbank = fbank.squeeze(0)
        fbank = torch.transpose(fbank, 0, 1)
        return fbank

    def __len__(self):
        return len(self.valid_indices)