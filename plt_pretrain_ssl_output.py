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
file_path = r"C:\Users\Lin\slurm-854730.out"

# 讀取檔案 (使用 utf-8)
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# --------------------------
# 1. 數據解析 (Data Parsing)
# --------------------------

train_loss_data = []
eval_data = []
lr_changes = []

# 常數：一個 Dataset Epoch 有多少步
STEPS_PER_EPOCH = 13398

# 正則表達式
loss_pattern = re.compile(r'Epoch:\s*\[(\d+)\]\[(\d+)/(\d+)\]\s*.*Train Loss\s*([\d\.]+)')
eval_step_pattern = re.compile(r'step (\d+) evaluation')
masked_acc_train_pattern = re.compile(r'masked acc train:\s*([\d\.]+)')
masked_acc_eval_pattern = re.compile(r'masked acc eval:\s*([\d\.]+)')
lr_report_pattern = re.compile(r'lr:\s*([0-9\.e\-]+)')
lr_reduce_pattern = re.compile(r'Epoch\s+(\d+):\s+reducing learning rate')

lines = content.split('\n')
current_eval_step = 0 # 追蹤當前的 Validation Step

for line in lines:
    # --- 1. 解析 Training Loss ---
    loss_match = loss_pattern.search(line)
    if loss_match:
        epoch_idx = int(loss_match.group(1))
        step_idx = int(loss_match.group(2))
        total_steps = int(loss_match.group(3))
        loss_val = float(loss_match.group(4))
        
        # 計算精確的 Epoch (例如 1.5)
        # 1-based index: Epoch 1 的第 0 步是 1.0
        frac_epoch = epoch_idx + (step_idx / total_steps) - 1 + 1 
        
        train_loss_data.append({
            'epoch': frac_epoch,
            'loss': loss_val
        })
        continue

    # --- 2. 解析 Evaluation Context (更新當前 Step) ---
    eval_step_match = eval_step_pattern.search(line)
    if eval_step_match:
        current_eval_step = int(eval_step_match.group(1))
        
        # 初始化這個 Step 的資料字典 (如果還沒建立)
        if not any(d['step'] == current_eval_step for d in eval_data):
            eval_data.append({
                'step': current_eval_step,
                # 換算成 Real Dataset Epoch
                'epoch': 1 + (current_eval_step / STEPS_PER_EPOCH)
            })
        continue

    # --- 3. 解析 Evaluation Metrics & LR ---
    # 這些數據通常緊跟在 'step X evaluation' 之後
    if current_eval_step > 0:
        # 找到對應當前 Step 的字典
        entry = next((d for d in eval_data if d['step'] == current_eval_step), None)
        if entry:
            mat = masked_acc_train_pattern.search(line)
            if mat: entry['masked_acc_train'] = float(mat.group(1))
            
            mae = masked_acc_eval_pattern.search(line)
            if mae: entry['masked_acc_eval'] = float(mae.group(1))
            
            lr_m = lr_report_pattern.search(line)
            # 確保只抓一次 LR
            if lr_m and 'lr' not in entry:
                try:
                    lr_val = float(lr_m.group(1).rstrip('.'))
                    entry['lr'] = lr_val
                except:
                    pass

    # --- 4. 解析 ReduceLROnPlateau Trigger ---
    # Log 中的 "Epoch X reducing" 其實是發生在當前的 Validation Step
    lr_reduce_match = lr_reduce_pattern.search(line)
    if lr_reduce_match:
        # 記錄發生時的真實 Epoch 時間點
        real_epoch = 1 + (current_eval_step / STEPS_PER_EPOCH)
        lr_changes.append(real_epoch)

# 轉換 DataFrame 並排序
df_loss = pd.DataFrame(train_loss_data)
if not df_loss.empty:
    df_loss = df_loss.sort_values('epoch')
    # 修正：min_periods=1 確保第一筆資料就開始畫線
    df_loss['moving_avg'] = df_loss['loss'].rolling(window=100, min_periods=1).mean()

df_eval = pd.DataFrame(eval_data)
if not df_eval.empty:
    df_eval = df_eval.sort_values('epoch')

# --------------------------
# 3. 繪圖 (Visualization)
# --------------------------
plt.figure(figsize=(16, 15))
x_min = 1
x_max = max(df_loss['epoch'].max() if not df_loss.empty else 1, df_eval['epoch'].max() if not df_eval.empty else 1)

# --- 圖 1: Training Loss ---
plt.subplot(3, 1, 1)
if not df_loss.empty:
    plt.plot(df_loss['epoch'], df_loss['loss'], color='blue', alpha=0.4, linewidth=0.8, label='Raw Loss')
    plt.plot(df_loss['epoch'], df_loss['moving_avg'], color='red', linewidth=2, label='Moving Avg (100 steps)')
    
plt.title('Training Loss', fontsize=14, fontweight='bold')
plt.ylabel('Loss', fontsize=12)
plt.xlim(x_min, x_max)
plt.grid(True, alpha=0.3)
plt.legend(loc='upper right')

# --- 圖 2: Masked Accuracy ---
plt.subplot(3, 1, 2)
if not df_eval.empty:
    plt.plot(df_eval['epoch'], df_eval['masked_acc_train'], 'o-', color='green', markersize=5, linewidth=1.5, label='Train Accuracy')
    plt.plot(df_eval['epoch'], df_eval['masked_acc_eval'], 's-', color='blue', markersize=5, linewidth=1.5, label='Eval Accuracy')

plt.title('Masked Accuracy', fontsize=14, fontweight='bold')
plt.ylabel('Accuracy', fontsize=12)
plt.xlim(x_min, x_max)
plt.grid(True, alpha=0.3)
plt.legend(loc='lower right')

# --- 圖 3: Learning Rate ---
plt.subplot(3, 1, 3)
if not df_eval.empty:
    plt.plot(df_eval['epoch'], df_eval['lr'], drawstyle='steps-post', color='purple', linewidth=2.5, label='Learning Rate')
    
    # 畫出修正後的 Trigger 線
    for trigger_epoch in lr_changes:
        plt.axvline(x=trigger_epoch, color='red', linestyle='--', linewidth=2, label='ReduceLR Trigger' if trigger_epoch == lr_changes[0] else "")
        plt.text(trigger_epoch, df_eval['lr'].min(), f' Trigger @ Epoch {trigger_epoch:.1f}', color='black', rotation=90, verticalalignment='bottom')

plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
plt.xlabel('Real Dataset Epoch', fontsize=12)
plt.ylabel('Learning Rate (Log Scale)', fontsize=12)
plt.yscale('log')
plt.xlim(x_min, x_max)
plt.grid(True, alpha=0.3, which="both")
plt.legend(loc='lower left')

plt.tight_layout()
plt.savefig('SSL854730_analysis.png')
