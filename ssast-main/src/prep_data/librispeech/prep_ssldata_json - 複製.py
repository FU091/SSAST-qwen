import os
import json

def generate_json(local_scan_path, remote_target_path, output_filename):
    """
    local_scan_path: 你現在執行這隻程式的電腦上，檔案存放的資料夾 (例如 D:\spectrogram_pt_name)
    remote_target_path: 你希望 JSON 裡面寫的路徑 (例如 /work/t113618009/spectrogram_pt_name)
    output_filename: 輸出的 json 檔名
    """
    
    data_list = []
    
    # 檢查本機路徑是否存在
    if not os.path.exists(local_scan_path):
        print(f"錯誤: 找不到本機資料夾 -> {local_scan_path}")
        print("請修改程式碼中的 'local_scan_path' 為你電腦上的真實路徑。")
        return

    print(f"正在掃描: {local_scan_path} ...")
    
    file_count = 0
    # 遍歷資料夾
    for root, dirs, files in os.walk(local_scan_path):
        for file in files:
            # 確保只抓取 .pt 檔
            if file.lower().endswith('.pt'):
                file_count += 1
                
                # 組合出在 HPC 上的絕對路徑
                # 注意：這裡強制使用 "/" (Linux 斜線)，不論你在 Windows 還是 Linux 執行
                remote_full_path = f"{remote_target_path}/{file}"
                
                # 建立字典 entry
                # key 依然保留 'wav' 以相容原本的 dataloader
                cur_dict = {
                    "wav": remote_full_path,
                    "labels": "/m/09x0r"
                }
                data_list.append(cur_dict)

    # 寫入 JSON
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump({'data': data_list}, f, indent=1)
        
    print(f"成功產出: {output_filename}")
    print(f"共包含 {file_count} 筆資料。")
    print("-" * 30)

# ==========================================
# 請在這裡設定你的路徑
# ==========================================

# 1. 處理 驗證集 (Validation Set) -> 產出 test.json
# 如果你在 Windows 跑這支程式，請確認 local_scan_path 是 D:\...
# 如果你在 HPC 上跑，請把 local_scan_path 改成跟 remote_target_path 一樣
generate_json(
    local_scan_path=r'D:\val_spectrogram_pt_name',        # 程式去哪裡找檔案 (依你實際情況修改)
    remote_target_path='/work/t113618009/val_spectrogram_pt_name', # JSON 裡面要寫什麼路徑
    output_filename='test.json'
)

# 2. 處理 訓練集 (Training Set) -> 產出 combined_train_data.json
generate_json(
    local_scan_path=r'D:\spectrogram_pt_name',            # 程式去哪裡找檔案 (依你實際情況修改)
    remote_target_path='/work/t113618009/spectrogram_6s_pt_name',     # JSON 裡面要寫什麼路徑
    output_filename='combined_train_data.json'
)