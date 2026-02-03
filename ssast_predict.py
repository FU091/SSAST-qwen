import torch
import torchaudio
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from ast_predict_models import ASTModel

# === 使用者設定 ===
MODEL_PATH = r"C:\Users\Lin\Desktop\2_code\ssast_hub\best_audio_model.pth"
LABEL_CSV_PATH = r"C:\Users\Lin\Desktop\2_code\ssast_hub\ssast-main\src\prep_data\librispeech\class_labels_indices.csv"
TARGET_FILE_LIST_CSV = r"D:\FS_SOUND\Soundscape_Checkdata_20250425_T.csv"
AUDIO_ROOT_DIR = r"D:\FS_SOUND"
OUTPUT_CSV_PATH = r"C:\Users\Lin\Desktop\2_code\ssast_hub\prediction_results_pth.csv"


# 參數對齊
NUM_CLASSES = 12
FSTRIDE = 10 
TSTRIDE = 10
MODEL_SIZE = 'base' 
NORM_MEAN = -7.4482
NORM_STD = 2.4689

def load_label_map(csv_path):
    """讀取標籤對照表"""
    try:
        df = pd.read_csv(csv_path)
        return df.set_index('index')['display_name'].to_dict()
    except Exception as e:
        print(f"無法讀取標籤表，將使用索引代號: {e}")
        return {i: f"class_{i}" for i in range(NUM_CLASSES)}

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_map = load_label_map(LABEL_CSV_PATH)
    class_names = [label_map.get(i, f"class_{i}") for i in range(NUM_CLASSES)]
    
    # 1. 初始化模型
    model = ASTModel(label_dim=NUM_CLASSES, fstride=FSTRIDE, tstride=TSTRIDE, model_size=MODEL_SIZE)
    
    # 2. 載入權重
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items() 
                      if 'v.head' not in k and 'cpred' not in k and 'gpred' not in k}
    
    model.load_state_dict(new_state_dict, strict=False)
    model.to(device).eval()

    # 3. 讀取檔案清單 (增加編碼容錯)
    try:
        # 先嘗試 utf-8
        df_files = pd.read_csv(TARGET_FILE_LIST_CSV, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            # 失敗則嘗試 big5 (台灣 Excel 常見格式)
            df_files = pd.read_csv(TARGET_FILE_LIST_CSV, encoding='big5')
        except UnicodeDecodeError:
            # 仍失敗則嘗試 gbk 或 cp950
            df_files = pd.read_csv(TARGET_FILE_LIST_CSV, encoding='cp950')
    
    # 修正：確保我們拿到正確的檔名列
    file_list = df_files['FileName'].dropna().tolist()
    all_results = []
    print(f"開始預測 {len(file_list)} 個檔案...")
    for fname in tqdm(file_list):
        # 強化路徑搜尋：遞迴搜尋 AUDIO_ROOT_DIR
        audio_path = None
        target_name = str(fname) if str(fname).lower().endswith('.wav') else f"{fname}.wav"
        
        # 這裡改用遍歷搜尋，確保能找到嵌套資料夾裡的檔案
        for root, dirs, files in os.walk(AUDIO_ROOT_DIR):
            if target_name in files:
                audio_path = os.path.join(root, target_name)
                break
        
        if not audio_path:
            continue # 如果找不到檔案則跳過

        try:
            waveform, sr = torchaudio.load(audio_path)
            if sr != 16000:
                waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
            
            fbank = torchaudio.compliance.kaldi.fbank(
                waveform, htk_compat=True, sample_frequency=16000, use_energy=False,
                window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10
            )

            if fbank.shape[0] < 1024:
                fbank = torch.nn.functional.pad(fbank, (0, 0, 0, 1024 - fbank.shape[0]))
            else:
                fbank = fbank[:1024, :]

            fbank = (fbank - NORM_MEAN) / (NORM_STD * 2)
            fbank = fbank.unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(fbank)
                probs = torch.sigmoid(output).cpu().numpy()[0]
            
            # 建立結果字典
            res = {
                'FileName': fname,
                'Full_Path': audio_path,
                'Label_0.5': ','.join([class_names[i] for i, p in enumerate(probs) if p >= 0.5]) or "None",
                'Label_0.8': ','.join([class_names[i] for i, p in enumerate(probs) if p >= 0.8]) or "None"
            }
            # 增加各別類別機率欄位
            for i, p in enumerate(probs):
                res[f'{class_names[i]}_prob'] = round(float(p), 4)
                
            all_results.append(res)

        except Exception as e:
            print(f"\n跳過損壞檔案 {fname}: {e}")

    # 4. 存檔
    if all_results:
        out_df = pd.DataFrame(all_results)
        out_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8-sig')
        print(f"\n預測完成，成功處理 {len(all_results)} 個檔案，結果存於: {OUTPUT_CSV_PATH}")
    else:
        print("\n錯誤：未成功處理任何檔案，請檢查 AUDIO_ROOT_DIR 路徑是否正確。")

if __name__ == '__main__':
    main()