import os
from pydub import AudioSegment

# ================= 設定區 =================

# 1. 原始資料夾路徑
TARGET_FOLDERS = [
    r"D:\GW_audio",
    r"D:\FS_SOUND"
]

# 2. 裁切後檔案的「輸出根目錄」 (請自行修改為您想要存放的地方)
# 程式會在這個資料夾下自動建立對應的 GW_audio 和 FS_SOUND 資料夾
OUTPUT_ROOT = r"D:\Audio_Output"

# 3. 裁切長度 (6000 毫秒 = 6 秒)
CHUNK_LENGTH_MS = 6000 

# =========================================

def process_audio_files():
    print(f"開始處理... 輸出目標資料夾: {OUTPUT_ROOT}")
    
    for source_root in TARGET_FOLDERS:
        if not os.path.exists(source_root):
            print(f"找不到原始資料夾: {source_root}，跳過。")
            continue

        # 取得原始資料夾的名稱 (例如 "GW_audio")，用於在輸出目錄建立第一層分類
        folder_name = os.path.basename(os.path.normpath(source_root))

        # 遍歷資料夾 (包含子資料夾)
        for dirpath, dirnames, filenames in os.walk(source_root):
            for filename in filenames:
                if filename.lower().endswith('.wav'):
                    
                    full_source_path = os.path.join(dirpath, filename)
                    
                    # --- 路徑計算核心邏輯 ---
                    # 1. 計算該檔案相對於原始根目錄的路徑 (例如: "2024\SubFolder")
                    rel_path = os.path.relpath(dirpath, source_root)
                    
                    # 2. 組合新的輸出路徑: 輸出根目錄 + 原始資料夾名 + 相對路徑
                    target_dir = os.path.join(OUTPUT_ROOT, folder_name, rel_path)
                    
                    # 3. 如果目標資料夾不存在，就建立它
                    if not os.path.exists(target_dir):
                        os.makedirs(target_dir)
                    # ----------------------

                    try:
                        print(f"處理: {filename}")
                        
                        audio = AudioSegment.from_file(full_source_path)
                        duration_ms = len(audio)
                        
                        part_count = 1
                        
                        # 迴圈裁切
                        for i in range(0, duration_ms, CHUNK_LENGTH_MS):
                            chunk = audio[i : i + CHUNK_LENGTH_MS]
                            
                            # 只儲存大於 0 秒的片段
                            if len(chunk) > 0:
                                # 組合新檔名
                                file_name_no_ext = os.path.splitext(filename)[0]
                                extension = os.path.splitext(filename)[1]
                                new_filename = f"{file_name_no_ext}_{part_count}{extension}"
                                
                                # 組合完整的輸出路徑
                                output_file_path = os.path.join(target_dir, new_filename)
                                
                                chunk.export(output_file_path, format="wav")
                                part_count += 1
                        
                    except Exception as e:
                        print(f"  -> 錯誤: {filename} - {e}")

    print("所有作業已完成。")

if __name__ == "__main__":
    process_audio_files()