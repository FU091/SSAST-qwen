import subprocess
import sys
import os

def run_ssast_pretraining():
    """
    使用 Docker 運行 SSAST 預訓練的 Python 腳本
    已修正參數錯誤：移除 data_dir, data_val_dir，並將 batch_size 改為 -b
    """
    print("開始運行 SSAST 預訓練...")

    # 構建 Docker 命令
    docker_cmd = [
        "docker", "run", "-it", "--rm",
        "--gpus", "all",
        "-v", "D:\\spectrogram_pt_name:/data/train",
        "-v", "D:\\val_spectrogram_pt_name:/data/val",
        "-v", "C:\\Users\\Lin\\Desktop\\2_code\\ssast_hub:/ssast",
        "ssast_hub_image",
        "python", "/ssast/ssast-main/src/run.py",
        "--dataset", "precomputed",
        "--data-train", "/ssast/combined_train_data.json",
        "--data-val", "/ssast/test.json",
        "--label-csv", "/ssast/class_labels_indices.csv",
        "--dataset_mean", "-7.4482",
        "--dataset_std", "2.4689",
        "--target_length", "1024",
        "--num_mel_bins", "128",
        "--fshape", "16",
        "--tshape", "16",
        "--fstride", "16",
        "--tstride", "16",
        "--model_size", "base",
        "--task", "pretrain_joint",
        "--mask_patch", "400",
        "-b", "24",
        "--lr", "0.0001",
        "--n-epochs", "10",
        "--exp-dir", "/ssast/exp",
        "--save_model", "False",
        "--freqm", "0",
        "--timem", "0",
        "--mixup", "0",
        "--n-print-steps", "100",
        "--lr_patience", "2",
        "--epoch_iter", "4000"
    ]

    try:
        print("執行命令: " + " ".join(docker_cmd))
        # 這裡改為直接使用 call，這樣可以看到即時輸出，而不是跑完才顯示
        # 如果發生錯誤，subprocess.call 會返回非 0 的數字
        return_code = subprocess.call(docker_cmd)
        
        if return_code == 0:
            print("預訓練完成!")
            return True
        else:
            print(f"預訓練執行失敗，錯誤代碼: {return_code}")
            return False

    except FileNotFoundError:
        print("錯誤: 找不到 Docker。請確認 Docker 已安裝並在 PATH 中。")
        return False

if __name__ == "__main__":
    success = run_ssast_pretraining()
    if not success:
        sys.exit(1)