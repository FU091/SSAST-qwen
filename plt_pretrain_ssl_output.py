"""
Docstring for plt_pretrain_ssl_output
我們在畫圖時，必須將「Epoch 單位」換算成圖表 X 軸的「Global Step 單位」：
記得換算完要去改 STEPS_PER_EPOCH = 
Epoch: [1][100/13398] ... Train Loss 76.1002
表示13398 step 是1epoch

"""

import re
import pandas as pd
import matplotlib.pyplot as plt

# 設定檔案路徑
file_path = "slurm-866749.out"

# 讀取檔案
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# --------------------------
# 1. 數據解析 (Data Parsing)
# --------------------------
train_loss_data = []
eval_data = []
lr_data = [] # 儲存所有 LR 點位

STEPS_PER_EPOCH = 133 # 根據日誌 Epoch: [x][y/133] 修正

# 正則表達式
loss_pattern = re.compile(r'Epoch:\s*\[(\d+)\]\[(\d+)/(\d+)\]\s*.*Train Loss\s*([\d\.]+)')
eval_step_pattern = re.compile(r'step (\d+) evaluation')
masked_acc_train_pattern = re.compile(r'masked acc train:\s*([\d\.]+)')
masked_acc_eval_pattern = re.compile(r'masked acc eval:\s*([\d\.]+)')
lr_report_pattern = re.compile(r'lr:\s*([0-9\.e\-]+)')
warmup_lr_pattern = re.compile(r'warm-up learning rate is ([\d\.]+)')

lines = content.split('\n')
current_epoch = 1
current_eval_step = None

for line in lines:
    # --- 解析 Warm-up LR ---
    w_match = warmup_lr_pattern.search(line)
    if w_match:
        lr_val = float(w_match.group(1))
        lr_data.append({'epoch': float(current_epoch), 'lr': lr_val})
        continue

    # --- 解析 Training Loss ---
    loss_match = loss_pattern.search(line)
    if loss_match:
        epoch_idx = int(loss_match.group(1))
        step_idx = int(loss_match.group(2))
        total_steps = int(loss_match.group(3))
        loss_val = float(loss_match.group(4))
        current_epoch = epoch_idx
        # 計算精確 Epoch 進度 (從 0 開始)
        frac_epoch = (epoch_idx - 1) + (step_idx / total_steps)
        train_loss_data.append({'epoch': frac_epoch, 'loss': loss_val})
        continue

    # --- 解析 Evaluation Step ---
    eval_step_match = eval_step_pattern.search(line)
    if eval_step_match:
        current_eval_step = int(eval_step_match.group(1))
        eval_epoch = current_eval_step / STEPS_PER_EPOCH
        eval_data.append({'step': current_eval_step, 'epoch': eval_epoch})
        continue

    # --- 解析 Evaluation 內的指標與 LR ---
    if current_eval_step is not None:
        entry = eval_data[-1]
        mat = masked_acc_train_pattern.search(line)
        if mat: entry['masked_acc_train'] = float(mat.group(1))
        mae = masked_acc_eval_pattern.search(line)
        if mae: entry['masked_acc_eval'] = float(mae.group(1))
        lr_m = lr_report_pattern.search(line)
        if lr_m:
            lr_val = float(lr_m.group(1).rstrip('.'))
            entry['lr'] = lr_val
            lr_data.append({'epoch': entry['epoch'], 'lr': lr_val})
            current_eval_step = None

# 轉換 DataFrame
df_loss = pd.DataFrame(train_loss_data)
df_eval = pd.DataFrame(eval_data)
df_lr = pd.DataFrame(lr_data).sort_values('epoch')

if not df_loss.empty:
    df_loss['moving_avg'] = df_loss['loss'].rolling(window=100, min_periods=1).mean()

# --------------------------
# 2. 繪圖 (Visualization)
# --------------------------
plt.figure(figsize=(16, 18))
x_max = max(df_loss['epoch'].max(), df_eval['epoch'].max(), df_lr['epoch'].max())

# 圖 1: Training Loss
plt.subplot(3, 1, 1)
plt.plot(df_loss['epoch'], df_loss['loss'], color='blue', alpha=0.3, label='Raw Loss')
plt.plot(df_loss['epoch'], df_loss['moving_avg'], color='red', linewidth=2, label='Moving Avg')
plt.title('Training Loss (Corrected Scale)', fontsize=14, fontweight='bold')
plt.ylabel('Loss')
plt.xlim(0, x_max); plt.grid(True, alpha=0.3); plt.legend()

# 圖 2: Masked Accuracy
plt.subplot(3, 1, 2)
plt.plot(df_eval['epoch'], df_eval['masked_acc_train'], 'o-', color='green', label='Train Accuracy')
plt.plot(df_eval['epoch'], df_eval['masked_acc_eval'], 's-', color='blue', label='Eval Accuracy')
plt.title('Masked Accuracy', fontsize=14, fontweight='bold')
plt.ylabel('Accuracy')
plt.xlim(0, x_max); plt.grid(True, alpha=0.3); plt.legend()

# 圖 3: Learning Rate (含 Warm-up)
plt.subplot(3, 1, 3)
plt.plot(df_lr['epoch'], df_lr['lr'], 'p-', color='purple', linewidth=2, markersize=4, label='Learning Rate')
plt.title('Learning Rate Schedule (with Warm-up)', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.xlim(0, x_max); plt.grid(True, alpha=0.3); plt.legend()

plt.tight_layout()
plt.savefig('SSL_fixed_analysis.png')