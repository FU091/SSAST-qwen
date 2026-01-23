import os
import shutil
import random

def split_dataset():
    # --- 1. 設定參數 ---
    source_dir = r"D:\spectrogram_6s_pt_name"           # 原始資料夾 (訓練集)
    target_dir = r"D:\val_spectrogram_6s_pt_name"      # 新資料夾 (驗證集)
    config_filename = "dataset_config.pt"
    val_ratio = 0.10                # 10% 驗證集

    # --- 2. 建立新資料夾 ---
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"✅ 已建立資料夾: {target_dir}")
    else:
        print(f"ℹ️ 資料夾已存在: {target_dir}")

    # --- 3. 掃描檔案 ---
    print("正在掃描檔案，請稍候...")
    all_files = os.listdir(source_dir)
    
    # 篩選出 .pt 檔，並排除 config 檔
    pt_files = [f for f in all_files if f.endswith('.pt') and f != config_filename]
    
    total_files = len(pt_files)
    move_count = int(total_files * val_ratio)
    
    print(f"📊 總共找到 {total_files} 個數據檔案。")
    print(f"🔄 預計移動 {move_count} 個檔案 (約 {val_ratio*100}%) 到 {target_dir}...")

    # --- 4. 隨機挑選檔案 (分散挑選) ---
    files_to_move = random.sample(pt_files, move_count)

    # --- 5. 移動檔案 (剪下 -> 貼上) ---
    print("🚀 開始移動檔案...")
    count = 0
    for file_name in files_to_move:
        src_path = os.path.join(source_dir, file_name)
        dst_path = os.path.join(target_dir, file_name)
        
        try:
            shutil.move(src_path, dst_path)
            count += 1
            # 每移動 1000 個檔案顯示一次進度
            if count % 1000 == 0:
                print(f"   已移動 {count} / {move_count} 筆...")
        except Exception as e:
            print(f"⚠️ 移動失敗: {file_name}, 錯誤: {e}")

    print(f"✅ 完成！共移動了 {count} 筆資料。")

    # --- 6. 複製 Config 檔 (複製 -> 貼上) ---
    config_src = os.path.join(source_dir, config_filename)
    config_dst = os.path.join(target_dir, config_filename)

    if os.path.exists(config_src):
        shutil.copy2(config_src, config_dst)
        print(f"✅ 已複製 Config 檔 ({config_filename}) 到驗證集資料夾。")
    else:
        print(f"⚠️ 警告: 在來源資料夾找不到 {config_filename}，請手動檢查。")

    print("\n🎉 所有步驟執行完畢。")
    print(f"訓練集剩餘: {len(os.listdir(source_dir)) - 1} (扣除 config)") # 簡單估算
    print(f"驗證集數量: {len(os.listdir(target_dir)) - 1} (扣除 config)")

if __name__ == "__main__":
    split_dataset()