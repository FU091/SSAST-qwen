import os
import json
import pandas as pd
import random
from collections import defaultdict, Counter

def read_csv_smart(filepath):
    """
    智慧讀取 CSV：
    1. 輪詢多種編碼 (UTF-8, CP950/Big5, GBK)。
    2. 確保關鍵欄位存在。
    """
    encodings_to_try = ['utf-8', 'cp950', 'utf-8-sig', 'gbk']
    check_cols = ['MainTag_鳥', 'MainTag_蟬', 'MainTag_蟲', 'MainTag_蛙']
    
    print(f"   [讀取測試] 正在偵測 CSV 編碼: {filepath}")
    
    for enc in encodings_to_try:
        try:
            df = pd.read_csv(filepath, encoding=enc, encoding_errors='replace')
            if any(col in df.columns for col in check_cols):
                print(f"[成功] 使用編碼 '{enc}' 讀取成功，欄位驗證正確。")
                return df
            else:
                continue
        except Exception as e:
            continue
    raise ValueError(f"無法讀取 CSV 檔，請確認檔案未損毀且包含 {check_cols} 其中之一的欄位。")

def generate_stratified_json_final(
    txt_file_path, 
    csv_path, 
    output_dir, 
    split_ratio=(0.8, 0.1, 0.1)
):
    random.seed(42) 

    # ==========================================
    # 1. 讀取 CSV
    # ==========================================
    print(f"1. 讀取 CSV: {csv_path} ...")
    if not os.path.exists(csv_path):
        print("錯誤: 找不到 CSV 檔案")
        return

    try:
        df = read_csv_smart(csv_path)
    except ValueError as e:
        print(f"錯誤: {e}")
        return

    # 定義欄位與映射
    source_classes = ['蟬', '蟲', '鳥', '鴞', '蛙', '哺乳類', '人', '風', '雨', '流水', '安靜', '航空器']
    name_mapping = {
        '蟬': 'Cicada', '蟲': 'Insect', '鳥': 'Bird', '鴞': 'Owl', '蛙': 'Frog',
        '哺乳類': 'Mammal', '人': 'Speech', '風': 'Wind', '雨': 'Rain', 
        '流水': 'Stream', '安靜': 'Silence', '航空器': 'Aircraft'
    }
    
    # [修正] 補回 all_defined_labels 的定義
    all_defined_labels = set()
    for s in source_classes:
        all_defined_labels.add(name_mapping.get(s, s))

    # 建立 CSV 查詢表 (Key = 檔名 Stem, Value = Labels)
    csv_label_map = {} 

    for idx, row in df.iterrows():
        # 強制只取 Stem
        raw_fname = str(row['FileName']).strip()
        fname_stem = os.path.splitext(os.path.basename(raw_fname))[0] 

        final_class_weights = {}
        for cls in source_classes:
            w = 0.0
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

    print(f"   CSV 處理完成，建立 {len(csv_label_map)} 筆標籤索引 (Example Key: {list(csv_label_map.keys())[0]})")

    # ==========================================
    # 2. 讀取 HPC 檔案列表並匹配
    # ==========================================
    print(f"2. 讀取檔案列表: {txt_file_path} ...")
    if not os.path.exists(txt_file_path):
        print("錯誤: 找不到 txt 列表")
        return

    with open(txt_file_path, 'r', encoding='utf-8') as f:
        hpc_files = [line.strip() for line in f if line.strip()]

    groups = {} 
    matched_count = 0
    missing_count = 0
    debug_miss_examples = []

    for full_path in hpc_files:
        if not full_path.endswith('.pt'):
            continue

        # 從完整路徑中提取純檔名 (Stem)
        filename_only = os.path.basename(full_path)
        file_stem = os.path.splitext(filename_only)[0]
        
        base_name = None
        
        # 1. 直接匹配
        if file_stem in csv_label_map:
            base_name = file_stem
        else:
            # 2. 嘗試切除尾綴匹配
            parts = file_stem.split('_')
            for i in range(len(parts)-1, 0, -1):
                potential_base = "_".join(parts[:i])
                if potential_base in csv_label_map:
                    base_name = potential_base
                    break

        '''
        if base_name:
            # 匹配成功：使用 TXT 裡原本的完整路徑，並轉為 Linux 斜線
            clean_full_path = full_path.replace("\\", "/")

            entry = {
                "wav": clean_full_path, 
                "labels": csv_label_map[base_name]['labels'],
                "weights": csv_label_map[base_name]['weights']
            }
            '''

        if base_name:
            # -----------------------------------------------------------
            # [修改] 強制指定 HPC 上的目標資料夾，忽略 txt 裡的舊路徑
            # -----------------------------------------------------------
            target_hpc_folder = "/work/t113618009/spectrogram_pt_name/"
            
            # 組合新路徑： "目標資料夾" + "檔名"
            # filename_only 變數你在第 114 行已經取得過了，直接拿來用即可
            final_path = target_hpc_folder + filename_only
            
            entry = {
                "wav": final_path,  # 這裡寫入的一定會是新路徑
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
            if len(debug_miss_examples) < 5:
                debug_miss_examples.append(file_stem)

    unique_ids = list(groups.keys())
    print(f"   匹配完成: {matched_count} 個檔案，歸類為 {len(unique_ids)} 個錄音事件(Groups)。")
    
    if missing_count > 0:
        print(f"   警告: 有 {missing_count} 個檔案在 CSV 中找不到對應標籤。")
        print(f"   [Debug] 前 5 個匹配失敗的檔名 (請檢查 CSV 是否有這些檔名):")
        for miss in debug_miss_examples:
            print(f"      - TXT提取出: {miss}")

    # ==========================================
    # 3. 統計與分層 (Stratified Split)
    # ==========================================
    label_counts = Counter()
    for uid in unique_ids:
        for tag in groups[uid]['tags']:
            label_counts[tag] += 1
            
    present_labels = [tag for tag in label_counts if label_counts[tag] > 0]
    
    print("\n" + "="*30)
    print("   【物種統計報告】")
    print(f"   目前存在的物種 ({len(present_labels)}): {present_labels}")
    print("="*30 + "\n")

    # ==========================================
    # 4. 貪婪分層劃分
    # ==========================================
    print("3. 正在執行分層劃分...")
    
    sorted_labels = sorted(present_labels, key=lambda x: label_counts[x])
    
    split_names = ['train', 'val', 'test']
    split_target_ratios = {'train': split_ratio[0], 'val': split_ratio[1], 'test': split_ratio[2]}
    split_buckets = {'train': [], 'val': [], 'test': []}
    split_counts = {'train': 0, 'val': 0, 'test': 0}
    
    assigned_ids = set()

    for label in sorted_labels:
        relevant_groups = [uid for uid in unique_ids if label in groups[uid]['tags'] and uid not in assigned_ids]
        random.shuffle(relevant_groups)
        
        for uid in relevant_groups:
            total_assigned = sum(split_counts.values()) + 1e-9
            current_ratios = {k: split_counts[k] / total_assigned for k in split_names}
            target_split = max(split_names, key=lambda k: split_target_ratios[k] - current_ratios[k])
            
            split_buckets[target_split].append(uid)
            split_counts[target_split] += len(groups[uid]['files'])
            assigned_ids.add(uid)

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
        
        split_tags = Counter()
        for uid in ids:
            for tag in groups[uid]['tags']:
                split_tags[tag] += 1
        print(f"   [{split_name}] 類別統計: {dict(split_tags)}")

    save_split('train')
    save_split('val')
    save_split('test')
    
    vocab_path = os.path.join(output_dir, "class_vocab.json")
    with open(vocab_path, 'w', encoding='utf-8') as f:
        vocab_list = sorted(list(all_defined_labels))
        json.dump({'all_classes': vocab_list, 'present_classes': sorted(present_labels)}, f, indent=1, ensure_ascii=False)
    
    print("-" * 30)
    print(f"處理完成！輸出資料夾: {output_dir}")

# ==========================================
# 參數設定
# ==========================================
if __name__ == "__main__":
    # 請依據您的實際路徑填寫
    csv_file = 'D:\\FS_SOUND\\Soundscape_Checkdata_20250425_T.csv' 
    txt_list_file = 'C:\\Users\\Lin\\hpc_file_val_list.txt'
    
    out_dir = './finetune_stratified_final'
    
    generate_stratified_json_final(
        txt_file_path=txt_list_file,
        csv_path=csv_file,
        output_dir=out_dir
    )