import os
import json
import random
from pathlib import Path

def collect_audio_files(audio_scan_path, filter_keywords=None):
    """
    收集指定關鍵詞的音頻文件路徑

    audio_scan_path: 本地音頻文件掃描路徑
    filter_keywords: 關鍵詞列表，用於篩選特定類型的文件(如['FS', 'TML'])
    """

    audio_files = []

    # 檢查本機路徑是否存在
    if not os.path.exists(audio_scan_path):
        print(f"錯誤: 找不到本機資料夾 -> {audio_scan_path}")
        print("請修改程式碼中的 'audio_scan_path' 為你電腦上的真實路徑。")
        return []

    print(f"正在掃描: {audio_scan_path} ...")

    file_count = 0
    # 遍歷資料夾
    for root, dirs, files in os.walk(audio_scan_path):
        for file in files:
            # 確保只處理 .wav 檔
            if file.lower().endswith('.wav'):
                # 如果指定了篩選關鍵詞，則檢查文件名是否包含這些關鍵詞
                if filter_keywords:
                    if not any(keyword.lower() in file.lower() for keyword in filter_keywords):
                        continue

                file_count += 1
                full_path = os.path.join(root, file)
                audio_files.append(full_path)

    print(f"共找到 {file_count} 個符合條件的音頻文件。")
    return audio_files


def split_data(audio_files, train_ratio=0.8):
    """
    將數據按比例分割為訓練集和驗證集

    audio_files: 音頻文件路徑列表
    train_ratio: 訓練集佔比
    """
    # 隨機打亂數據
    shuffled_files = audio_files.copy()
    random.shuffle(shuffled_files)

    # 計算分割點
    split_point = int(len(shuffled_files) * train_ratio)

    train_files = shuffled_files[:split_point]
    val_files = shuffled_files[split_point:]

    return train_files, val_files


def generate_ssl_json_from_file_list(file_list, pt_remote_path, output_filename):
    """
    從文件列表生成SSL JSON文件

    file_list: 音頻文件完整路徑列表
    pt_remote_path: 遠端.pt文件路徑前綴
    output_filename: 輸出JSON文件名
    """

    data_list = []

    print(f"正在處理 {len(file_list)} 個文件到 {output_filename} ...")

    for full_path in file_list:
        filename = os.path.basename(full_path)

        # 根據音頻文件名生成對應的.pt文件名
        pt_filename = filename.replace('.wav', '.pt')

        # 組合出在 HPC 上的絕對路徑
        remote_full_path = f"{pt_remote_path}/{pt_filename}"

        # 建立字典 entry
        cur_dict = {
            "wav": remote_full_path,
            "labels": "/m/09x0r"
        }
        data_list.append(cur_dict)

    # 寫入 JSON
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump({'data': data_list}, f, indent=1)

    print(f"成功產出: {output_filename}")
    print(f"共包含 {len(data_list)} 筆資料。")
    print("-" * 30)


def generate_mixed_fs_tml_ssl_json(audio_scan_path, pt_remote_path_base, train_output_filename, val_output_filename, train_ratio=0.8):
    """
    生成混合FS和TML的SSL JSON文件，分為訓練集和驗證集
    """

    # 收集FS和TML文件
    print("正在收集FS和TML音頻文件...")
    audio_files = collect_audio_files(
        audio_scan_path=audio_scan_path,
        filter_keywords=['FS', 'TML']  # 同時篩選FS和TML文件
    )

    if not audio_files:
        print("未找到任何FS或TML音頻文件！")
        return

    # 分割數據
    print(f"開始分割數據，訓練集比例: {train_ratio}")
    train_files, val_files = split_data(audio_files, train_ratio)

    # 生成訓練集JSON - 使用相同的目錄路徑
    print("正在生成訓練集SSL JSON文件...")
    generate_ssl_json_from_file_list(
        file_list=train_files,
        pt_remote_path=pt_remote_path_base,  # 使用相同的基礎路徑
        output_filename=train_output_filename
    )

    # 生成驗證集JSON - 使用相同的目錄路徑
    print("正在生成驗證集SSL JSON文件...")
    generate_ssl_json_from_file_list(
        file_list=val_files,
        pt_remote_path=pt_remote_path_base,  # 使用相同的基礎路徑
        output_filename=val_output_filename
    )


# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":
    # 設定路徑參數
    AUDIO_SCAN_PATH = r'D:\FS_SOUND'  # 音頻文件掃描路徑
    PT_REMOTE_PATH_BASE = '/work/t113618009/spectrogram_pt_name'  # 遠端.pt文件路徑基礎

    # 生成混合FS和TML的SSL JSON文件（分為訓練集和驗證集）
    generate_mixed_fs_tml_ssl_json(
        audio_scan_path=AUDIO_SCAN_PATH,
        pt_remote_path_base=PT_REMOTE_PATH_BASE,
        train_output_filename='ssl_fs_tml_train.json',
        val_output_filename='ssl_fs_tml_val.json',
        train_ratio=0.8  # 80% 用於訓練，20% 用於驗證
    )

    print("FS和TML混合SSL JSON文件已生成完成！")