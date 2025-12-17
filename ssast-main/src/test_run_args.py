# test_run_args.py
import argparse
import sys
import os
import traceback

# 修復路徑問題 - 正確設置 Python 路徑
current_dir = os.path.dirname(os.path.abspath(__file__))  
print(f"當前目錄: {current_dir}")

ssast_main_dir = os.path.join(current_dir, 'ssast-main')
print(f"ssast-main 目錄: {ssast_main_dir}")

src_dir = os.path.join(ssast_main_dir, 'src')
print(f"src 目錄: {src_dir}")

# 將 src 目錄添加到 Python 路徑的最前面，確保導入優先權
sys.path.insert(0, src_dir)

# 測試導入
try:
    from dataloader_pt_reader import PrecomputedDataset
    print("✅ 成功導入 PrecomputedDataset")
except ImportError as e:
    print(f"❌ 導入失敗: {e}")
    # 檢查 src 目錄下有哪些文件，方便排錯
    if os.path.exists(src_dir):
        files = os.listdir(src_dir)
        print(f"src 目錄中的文件: {files}")
    else:
        print(f"❌ src 目錄不存在: {src_dir}")

def test_run_to_dataloader_params():
    print("\n=== 測試 run.py 參數傳遞到 dataloader_pt_reader.py ===")

    # 模擬 run.py 中的參數設置
    class Args:
        def __init__(self):
            self.dataset = 'precomputed'
            self.data_train = os.path.join(current_dir, 'combined_train_data.json')
            self.data_val = os.path.join(current_dir, 'test.json')
            self.data_dir = 'D:/spectrogram_pt_name'
            self.data_val_dir = 'D:/val_spectrogram_pt_name'

    args = Args()

    # 模擬 run.py 中的 audio_conf 設置
    audio_conf = {
        'num_mel_bins': 128,
        'target_length': 1024,
        'freqm': 0,
        'timem': 0,
        'mixup': 0,
        'dataset': 'audioset',
        'mode': 'train',
        'mean': -7.4482,
        'std': 2.4689,
        'noise': False,
    }

    val_audio_conf = {
        'num_mel_bins': 128,
        'target_length': 1024,
        'freqm': 0,
        'timem': 0,
        'mixup': 0,
        'dataset': 'audioset',
        'mode': 'evaluation',
        'mean': -7.4482,
        'std': 2.4689,
        'noise': False,
    }

    # 正確設置 data_dir 邏輯
    audio_conf['data_dir'] = args.data_dir if args.data_dir else args.data_train
    val_audio_conf['data_dir'] = args.data_val_dir if args.data_val_dir else (
        args.data_val if args.data_val != args.data_train else args.data_dir
    )

    print(f"訓練 audio_conf['data_dir']: {audio_conf['data_dir']}")
    print(f"驗證 val_audio_conf['data_dir']: {val_audio_conf['data_dir']}")

    # 測試是否能創建 PrecomputedDataset
    try:
        train_dataset = PrecomputedDataset(
            data_dir=audio_conf['data_dir'],
            dataset_json_file=args.data_train,
            audio_conf=audio_conf
        )
        print(f"✅ 訓練資料集創建成功，長度: {len(train_dataset)}")

        val_dataset = PrecomputedDataset(
            data_dir=val_audio_conf['data_dir'],
            dataset_json_file=args.data_val,
            audio_conf=val_audio_conf
        )
        print(f"✅ 驗證資料集創建成功，長度: {len(val_dataset)}")

        # 測試讀取一個樣本
        if len(train_dataset) > 0:
            sample_data, sample_label = train_dataset[0]
            print(f"✅ 訓練樣本讀取成功: fbank shape = {sample_data.shape}, label shape = {sample_label.shape}")

        if len(val_dataset) > 0:
            val_sample_data, val_sample_label = val_dataset[0]
            print(f"✅ 驗證樣本讀取成功: fbank shape = {val_sample_data.shape}, label shape = {val_sample_label.shape}")

        return True

    except Exception as e:
        print(f"❌ 錯誤: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_run_to_dataloader_params()
    if success:
        print("\n🎉 參數傳遞測試成功！")
    else:
        print("\n💥 參數傳遞測試失敗！")