# get normalization stats for input spectrogram
# -*- coding: utf-8 -*-
# Corrected for global calculation accuracy

import torch
import numpy as np
import dataloader
from tqdm import tqdm  # 如果沒有這個庫，可以 pip install tqdm，或刪除相關代碼

# 設定配置
# 注意：計算 mean/std 時，必須把 mixup, freqm, timem 都設為 0，
# 這樣才能算出「原始數據」的統計值，而不是增強後數據的統計值。
audio_conf = {
    'num_mel_bins': 128, 
    'target_length': 1024, 
    'freqm': 0,       # 修正：計算統計值時不應遮罩
    'timem': 0,       # 修正：計算統計值時不應遮罩
    'mixup': 0,       # 修正：計算統計值時不應 Mixup
    'skip_norm': True, # 這是必須的，確保讀入的數據是原始數值
    'mode': 'train', 
    'dataset': 'audioset'
}

def main():
    # 請確認以下路徑在你的電腦上是正確的
    json_path = r'C:\Users\Lin\Desktop\2_code\ssast_hub\combined_train_data.json'
    csv_path = r'C:\Users\Lin\Desktop\2_code\ssast_hub\class_labels_indices.csv'

    print("Initializing Dataloader...")
    dataset = dataloader.AudiosetDataset(
        json_path,
        label_csv=csv_path,
        audio_conf=audio_conf
    )

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=100,  # 根據你的顯存大小調整，純計算不需要反向傳播，可以設大一點
        shuffle=False, 
        num_workers=8, 
        pin_memory=True
    )

    # 初始化累加器 (使用 float64 防止溢出)
    total_sum = 0.0
    total_sq_sum = 0.0
    total_num = 0.0

    print("Start calculating global stats...")
    
    # 使用 tqdm 顯示進度條
    for i, (audio_input, labels) in enumerate(tqdm(train_loader)):
        # audio_input shape: [batch_size, time_frames, freq_bins]
        
        # 轉到 GPU 加速計算 (如果有 GPU)
        if torch.cuda.is_available():
            audio_input = audio_input.cuda()

        # 展平成一維或者保持原樣皆可，重點是求全體 sum
        # 這裡我們計算整個 Dataset 所有像素點的全域 Mean/Std
        
        audio_input = audio_input.double() # 轉成雙精度以防數值誤差
        
        total_sum += torch.sum(audio_input)
        total_sq_sum += torch.sum(audio_input ** 2)
        total_num += audio_input.numel() # 累加總元素個數 (Batch * Time * Freq)

    # 計算最終結果
    # E[X]
    global_mean = total_sum / total_num
    
    # Var(X) = E[X^2] - (E[X])^2
    global_variance = (total_sq_sum / total_num) - (global_mean ** 2)
    global_std = torch.sqrt(global_variance)

    print('\nCalculation Finished!')
    print('-------------------------------------------')
    print(f'Total elements processed: {int(total_num)}')
    print(f'Global Mean: {global_mean.item():.4f}')
    print(f'Global Std : {global_std.item():.4f}')
    print('-------------------------------------------')
    
    # 方便複製的格式
    print(f"target_mean = {global_mean.item():.4f}")
    print(f"target_std = {global_std.item():.4f}")

if __name__ == "__main__":
    main()