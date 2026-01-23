import os
import json
import pandas as pd
import random
from collections import defaultdict, Counter

def generate_stratified_json_v2(
    txt_file_path, 
    remote_target_path, 
    csv_path, 
    output_dir, 
    split_ratio=(0.8, 0.1, 0.1) # Train / Val / Test
):
    """
    優化版：
    1. 支援「稀有物種」優先分層。
    2. 自動忽略「當前不存在的物種」(Count=0)，防止報錯。
    3. 保持 Grouping (同一錄音檔不拆散)。
    """
    
    random.seed(42) 

    # ==========================================
    # 1. 讀取 CSV 與 定義所有可能的類別
    # ==========================================
    print(f"1. 讀取 CSV: {csv_path} ...")
    if not os.path.exists(csv_path):
        print("錯誤: 找不到 CSV 檔案")
        return
    
    # --- [修正點] 嘗試不同編碼讀取 ---
    try:
        # 先嘗試標準 UTF-8
        df = pd.read_csv(csv_path, encoding='utf-8')
    except UnicodeDecodeError:
        print("   偵測到編碼錯誤，切換為 CP950 (Big5) 模式讀取...")
        try:
            # 嘗試台灣 Windows 常用的 CP950 (Big5)
            df = pd.read_csv(csv_path, encoding='cp950')
        except UnicodeDecodeError:
            # 若還是失敗，嘗試 utf-16 (少見但有可能)
            df = pd.read_csv(csv_path, encoding='utf-16')
            

    # 定義所有「理論上可能出現」的來源標籤
    source_classes = ['蟬', '蟲', '鳥', '鴞', '蛙', '哺乳類', '人', '風', '雨', '流水', '安靜', '航空器']
    
    # 定義名稱映射
    name_mapping = {'人': '人聲', '鳥類': '鳥', '鳥': '鳥'}
    
    # 建立一個 Set 來存放所有轉換後的目標標籤名稱 (例如: '人聲', '鳥', '蟬'...)
    # 這樣我們才能知道哪些標籤是「定義了但沒出現」
    all_defined_labels = set()
    for s in source_classes:
        all_defined_labels.add(name_mapping.get(s, s))

    df = pd.read_csv(csv_path)
    csv_label_map = {} 

    for idx, row in df.iterrows():
        fname = str(row['FileName']).strip()
        fname_stem = os.path.splitext(fname)[0]

        final_class_weights = {}
        for cls in source_classes:
            w = 0.0
            # 容錯處理：確認欄位是否存在
            main_col = f"MainTag_{cls}"
            sub_col = f"SubTag_{cls}"
            
            if main_col in df.columns and row[main_col] == 1: w = 1.0
            elif sub_col in df.columns and row[sub_col] == 1: w = 0.6
            
            if w > 0:
                target_name = name_mapping.get(cls, cls)
                final_class_weights[target_name] = max(final_class_weights.get(target_name, 0), w)
        
        if final_class_weights:
            csv_label_map[fname_stem] = {
                "labels": list(final_class_weights.keys()),
                "weights": list(final_class_weights.values())
            }

    print(f"   CSV 載入完成。")

    # ==========================================
    # 2. 讀取 HPC 列表並建立 Group 資料
    # ==========================================
    print(f"2. 讀取檔案列表: {txt_file_path} ...")
    if not os.path.exists(txt_file_path):
        print("錯誤: 找不到 txt 列表")
        return

    with open(txt_file_path, 'r', encoding='utf-8') as f:
        hpc_files = [line.strip() for line in f if line.strip().endswith('.pt')]

    groups = {} 
    
    matched_count = 0
    missing_count = 0

    for pt_file in hpc_files:
        file_stem = os.path.splitext(pt_file)[0]
        
        # 反推 Base Name
        base_name = None
        if file_stem in csv_label_map:
            base_name = file_stem
        else:
            parts = file_stem.split('_')
            if len(parts) > 1:
                potential_base = "_".join(parts[:-1])
                if potential_base in csv_label_map:
                    base_name = potential_base

        if base_name:
            # 處理路徑：如果 txt 裡已經有路徑，就不加前綴；否則加上
            if "/" in pt_file or "\\" in pt_file:
                 remote_full_path = pt_file
            else:
                 remote_full_path = f"{remote_target_path}/{pt_file}"

            entry = {
                "wav": remote_full_path,
                "labels": csv_label_map[base_name]['labels'],
                "weights": csv_label_map[base_name]['weights']
            }
            
            if base_name not in groups:
                groups[base_name] = {
                    'files': [], 
                    'tags': set(csv_label_map[base_name]['labels']) 
                }
            groups[base_name]['files'].append(entry)
            matched_count += 1
        else:
            missing_count += 1

    unique_ids = list(groups.keys())
    print(f"   匹配完成: {matched_count} 個檔案，歸類為 {len(unique_ids)} 個錄音事件(Groups)。")

    # ==========================================
    # 3. 區分「存在物種」與「不存在物種」
    # ==========================================
    label_counts = Counter()
    for uid in unique_ids:
        for tag in groups[uid]['tags']:
            label_counts[tag] += 1
            
    # 找出目前資料集中實際存在的標籤
    present_labels = [tag for tag in label_counts if label_counts[tag] > 0]
    
    # 找出定義了但沒出現的標籤
    absent_labels = [tag for tag in all_defined_labels if tag not in label_counts]

    print("\n" + "="*30)
    print("   【物種統計報告】")
    print(f"   目前存在的物種 ({len(present_labels)}): {present_labels}")
    print(f"   目前不存在的物種 ({len(absent_labels)}): {absent_labels} (將跳過處理)")
    print("="*30 + "\n")

    # ==========================================
    # 4. 貪婪分層劃分 (Greedy Stratified Split)
    # ==========================================
    print("3. 正在執行分層劃分 (保護稀有物種)...")
    
    # 只對「存在的物種」依照稀有度排序 (越稀有越前面)
    sorted_labels = sorted(present_labels, key=lambda x: label_counts[x])
    
    split_names = ['train', 'val', 'test']
    split_target_ratios = {'train': split_ratio[0], 'val': split_ratio[1], 'test': split_ratio[2]}
    split_buckets = {'train': [], 'val': [], 'test': []}
    split_counts = {'train': 0, 'val': 0, 'test': 0} # 記錄檔案總數
    
    assigned_ids = set()

    for label in sorted_labels:
        # 找出包含此類別且未分配的 Group
        relevant_groups = [uid for uid in unique_ids if label in groups[uid]['tags'] and uid not in assigned_ids]
        random.shuffle(relevant_groups)
        
        for uid in relevant_groups:
            total_assigned = sum(split_counts.values()) + 1e-9
            current_ratios = {k: split_counts[k] / total_assigned for k in split_names}
            
            # 找最缺的箱子
            target_split = max(split_names, key=lambda k: split_target_ratios[k] - current_ratios[k])
            
            split_buckets[target_split].append(uid)
            split_counts[target_split] += len(groups[uid]['files'])
            assigned_ids.add(uid)

    # 處理剩餘的 Groups (通常是包含的標籤都已經被處理過了，或是無標籤資料)
    remaining_ids = [uid for uid in unique_ids if uid not in assigned_ids]
    random.shuffle(remaining_ids)
    for uid in remaining_ids:
        total_assigned = sum(split_counts.values()) + 1e-9
        current_ratios = {k: split_counts[k] / total_assigned for k in split_names}
        target_split = max(split_names, key=lambda k: split_target_ratios[k] - current_ratios[k])
        
        split_buckets[target_split].append(uid)
        split_counts[target_split] += len(groups[uid]['files'])

    print(f"   劃分完成。檔案分佈: Train={split_counts['train']}, Val={split_counts['val']}, Test={split_counts['test']}")

    # ==========================================
    # 5. 寫入 JSON
    # ==========================================
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    def save_split(split_name):
        ids = split_buckets[split_name]
        out_list = []
        for uid in ids:
            out_list.extend(groups[uid]['files'])
            
        path = os.path.join(output_dir, f"{split_name}.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'data': out_list}, f, indent=1, ensure_ascii=False)
        
        # 驗證
        split_tags = Counter()
        for uid in ids:
            for tag in groups[uid]['tags']:
                split_tags[tag] += 1
        print(f"   [{split_name}] 類別統計: {dict(split_tags)}")

    save_split('train')
    save_split('val')
    save_split('test')
    
    # 額外產出: class_vocab.json (方便檢查或未來對應用)
    vocab_path = os.path.join(output_dir, "class_vocab.json")
    with open(vocab_path, 'w', encoding='utf-8') as f:
        # 將所有定義的標籤(含不存在的)都寫進去，按照名稱排序
        vocab_list = sorted(list(all_defined_labels))
        json.dump({'all_classes': vocab_list, 'present_classes': sorted(present_labels)}, f, indent=1, ensure_ascii=False)
    print(f"   已建立類別清單: {vocab_path}")

    print("-" * 30)
    print(f"處理完成！輸出資料夾: {output_dir}")

# ==========================================
# 參數設定
# ==========================================
if __name__ == "__main__":
    # 1. CSV 檔案
    csv_file = r"D:\FS_SOUND\Soundscape_Checkdata_20250425_T.csv"
    
    # 2. 合併後的檔案列表 (請在 HPC 上先合併成一個 txt)
    txt_list_file = r"C:\Users\Lin\hpc_file_val_list.txt"
    
    # 3. HPC 路徑 (如果 txt 裡沒有路徑才需要填，否則留空或填入前綴)
    # 假設您的 txt 只有檔名，且所有檔都在這個資料夾：
    remote_path = '/work/t113618009/val_spectrogram_pt_name'
    
    # 4. 輸出位置
    out_dir = './finetune_stratified_v2'
    
    generate_stratified_json_v2(
        txt_file_path=txt_list_file,
        remote_target_path=remote_path,
        csv_path=csv_file,
        output_dir=out_dir
    )