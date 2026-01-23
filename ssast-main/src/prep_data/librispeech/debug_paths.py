import json
import os
import sys

# 你的設定檔路徑
json_file = "/work/t113618009/ssast_hub/finetune_stratified_final/train.json"
# 你的資料根目錄
data_root = "/work/t113618009/spectrogram_pt_name"

print("="*30)
print("=== Singularity 環境診斷 ===")
print("="*30)

# 1. 檢查資料根目錄是否成功掛載
print(f"[Check 1] 檢查根目錄: {data_root}")
if os.path.exists(data_root):
    print(f"   -> [OK] 資料夾存在。")
    try:
        files = os.listdir(data_root)
        print(f"   -> 資料夾內檔案數量: {len(files)}")
        print(f"   -> 前 5 個檔案: {files[:5]}")
        if len(files) == 0:
            print("   -> [警告] 資料夾是空的！請檢查宿主機路徑是否正確。")
    except Exception as e:
        print(f"   -> [Error] 無法讀取目錄內容: {e}")
else:
    print(f"   -> [CRITICAL] 資料夾不存在！")
    # 嘗試列出上一層，看看是否拼字錯誤
    parent_dir = os.path.dirname(data_root)
    if os.path.exists(parent_dir):
        print(f"   -> 上一層目錄 ({parent_dir}) 內容: {os.listdir(parent_dir)}")
    else:
        print(f"   -> 上一層目錄也不存在。")

print("-" * 20)

# 2. 檢查 JSON 內容與路徑組合
print(f"[Check 2] 讀取 JSON: {json_file}")
try:
    with open(json_file, 'r', encoding='utf-8') as f:
        content = json.load(f)
        data_list = content.get('data', [])
        print(f"   -> 成功讀取，共有 {len(data_list)} 筆資料。")
except Exception as e:
    print(f"   -> [Error] JSON 讀取失敗: {e}")
    sys.exit(1)

if len(data_list) == 0:
    print("   -> JSON 內無資料，結束。")
    sys.exit(1)

# 3. 模擬 dataloader 抓取第一筆資料
sample = data_list[0]
json_path = sample['wav'] # JSON 裡的路徑
print(f"[Check 3] 測試第一筆資料")
print(f"   -> JSON 內的路徑 (A): {json_path}")

# 測試 A: 直接讀取 (如果 JSON 是絕對路徑)
exists_a = os.path.exists(json_path)
print(f"   -> [測試 A] 直接存取結果: {'EXIST (成功)' if exists_a else 'NOT FOUND (失敗)'}")

# 測試 B: 組合路徑 (模擬 os.path.join)
# 注意：如果 json_path 是絕對路徑，os.path.join 標準行為會忽略 data_root
# 但如果你的 run.py 用的是字串相加，或者是把 json_path 當相對路徑處理，就會出錯
join_path = os.path.join(data_root, json_path)
print(f"   -> [測試 B] os.path.join(root, path) 結果: {join_path}")
print(f"   -> 存取結果: {'EXIST' if os.path.exists(join_path) else 'NOT FOUND'}")

# 測試 C: 強制相對路徑組合 (除去開頭的 /)
relative_path = json_path.lstrip('/')
force_join_path = os.path.join(data_root, relative_path)
print(f"   -> [測試 C] 強制串接 (Root + Relative): {force_join_path}")
print(f"   -> 存取結果: {'EXIST (如果是這個成功，代表 run.py 預期 JSON 裡要是相對路徑)' if os.path.exists(force_join_path) else 'NOT FOUND'}")

print("="*30)