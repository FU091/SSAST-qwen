import torch
import matplotlib.pyplot as plt
import numpy as np
import os

# =================設定檔名=================
# 請修改成您電腦上實際的檔案路徑
file_path = r"D:\spectrogram_6s_pt_name\FS01_20230107_201500_5.pt"
# ==========================================

if not os.path.exists(file_path):
    print(f"錯誤: 找不到檔案 {file_path}")
else:
    try:
        print(f"正在讀取 {file_path} ...")
        
        # 1. 讀取 .pt 檔案 (讀取到 CPU)
        loaded_dict = torch.load(file_path, map_location=torch.device('cpu'))
        
        # 2. 確認這是不是一個字典，並提取 "x" (fbank) 和 "y" (label)
        if isinstance(loaded_dict, dict) and "x" in loaded_dict:
            fbank_data = loaded_dict["x"] # 這是 float16, shape [1024, 128]
            label_data = loaded_dict["y"]
            
            print("讀取成功！")
            print(f"字典 Keys: {loaded_dict.keys()}")
            print(f"Fbank shape: {fbank_data.shape}, Type: {fbank_data.dtype}")
            print(f"Label: {label_data}")

            # 3. 轉回 Float32 (為了安全繪圖與顯示)
            fbank_data = fbank_data.float() 
            
            # 轉為 numpy
            np_data = fbank_data.numpy()

            # --- 部分一：印出完整數值 (寫入 txt) ---
            output_txt = 'fbank_values.txt'
            print(f"\n正在將 1024x128 的數值寫入 {output_txt} ...")
            
            with open(output_txt, 'w', encoding='utf-8') as f:
                rows, cols = np_data.shape
                f.write(f"Fbank Data (Shape: {rows}x{cols})\n")
                f.write(f"Label: {label_data}\n")
                f.write("-" * 30 + "\n")
                
                # 遍歷所有數值
                for t in range(rows):      # 0~1023
                    for m in range(cols):  # 0~127
                        # 格式: [Time=t, Mel=m]: 數值
                        f.write(f"[{t}, {m}]: {np_data[t, m]:.6f}\n")
                    f.write("\n") # 每個 frame 結束換行
                    
            print(f"數值儲存完成，請查看 {output_txt}")

            # --- 部分二：繪製梅爾頻譜圖 ---
            print("正在繪製頻譜圖...")
            
            plt.figure(figsize=(10, 6))
            
            # 轉置: 將 [1024, 128] 轉為 [128, 1024] 以符合 (Freq, Time) 慣例
            plot_data = np_data.T
            
            # 繪圖 (origin='lower' 確保低頻在下方)
            plt.imshow(plot_data, origin='lower', aspect='auto', cmap='inferno')
            
            plt.colorbar(label='Magnitude (Float16 raw value)')
            plt.title(f"Mel Spectrogram (Label: {label_data})")
            plt.xlabel("Time Frames (1024)")
            plt.ylabel("Mel Bins (128)")
            plt.tight_layout()
            
            plt.show()
            
        else:
            print("錯誤: 檔案內容格式不符，找不到 key 'x'。")
            print(f"檔案內容類型: {type(loaded_dict)}")

    except Exception as e:
        print(f"讀取或處理時發生錯誤: {e}")