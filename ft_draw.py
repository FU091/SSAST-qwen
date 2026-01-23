import re
import matplotlib.pyplot as plt
import os
import sys

def parse_slurm_log(log_path):
    train_losses = []
    valid_losses = []
    maps = []
    epochs = []
    
    current_epoch = 0
    temp_train_loss = 0.0 # 初始化以防報錯
    
    # 讀取 Log 檔案
    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    for line in lines:
        # 抓取 Train Loss
        # 格式範例: Epoch: [1][100/133] ... Train Loss 0.3613
        if "Train Loss" in line:
            epoch_match = re.search(r'Epoch: \[(\d+)\]', line)
            loss_match = re.search(r'Train Loss (\d+\.\d+)', line)
            
            if epoch_match and loss_match:
                ep = int(epoch_match.group(1))
                loss = float(loss_match.group(1))
                
                if ep > current_epoch:
                    current_epoch = ep
                    temp_train_loss = loss
                else:
                    temp_train_loss = loss

        # 抓取 Validation 數據
        # 格式: valid_loss: 0.719642
        if "valid_loss:" in line:
            parts = line.split('valid_loss:')
            if len(parts) > 1:
                v_loss = float(parts[1].split()[0].strip()) # 確保只抓數值
                valid_losses.append(v_loss)
                train_losses.append(temp_train_loss)
                epochs.append(current_epoch)

        # 抓取 mAP
        # 格式: mAP: 0.381193
        if "mAP:" in line:
            parts = line.split('mAP:')
            if len(parts) > 1:
                m_map = float(parts[1].split()[0].strip())
                maps.append(m_map)

    return epochs, train_losses, valid_losses, maps

# ==========================================
# 使用設定：請把您的單一檔名填在這裡
# ==========================================
target_log_file = r"C:\Users\Lin\Desktop\2_code\ssast_hub\soft_labels_865573.out"

# 檢查檔案是否存在
if not os.path.exists(target_log_file):
    print(f"錯誤: 找不到檔案 '{target_log_file}'")
    sys.exit(1)

# 取得檔名 (不含副檔名)，用於存檔和標題
# 例如: "slurm-865180.out" -> "slurm-865180"
base_name = os.path.splitext(os.path.basename(target_log_file))[0]

# 開始解析
print(f"正在解析: {target_log_file} ...")
epochs, t_losses, v_losses, maps = parse_slurm_log(target_log_file)

if not epochs:
    print("警告: 未解析到任何數據，請確認 Log 格式是否正確。")
    sys.exit(1)

# 設定畫布
plt.figure(figsize=(14, 6))

# --- 左圖: mAP ---
plt.subplot(1, 2, 1)
plt.plot(epochs, maps, marker='o', color='b', label='mAP')
plt.title(f"mAP Curve ({base_name})")
plt.xlabel("Epochs")
plt.ylabel("mAP")
plt.grid(True)
plt.legend()

# --- 右圖: Loss ---
plt.subplot(1, 2, 2)
plt.plot(epochs, t_losses, label='Train Loss', linestyle='--', color='orange')
plt.plot(epochs, v_losses, label='Valid Loss', color='red')
plt.title(f"Loss Curve ({base_name})")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()

plt.tight_layout()

# 儲存圖片 (自動使用 .out 的檔名，改為 .png)
output_filename = f"{base_name}.png"
plt.savefig(output_filename)
print(f"圖表已儲存為: {output_filename}")

# 顯示圖片
plt.show()