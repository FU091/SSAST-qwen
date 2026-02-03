#!/bin/bash
#SBATCH --job-name=ssast_ssl    ## job name
#SBATCH --gres=gpu:1             ## 每個節點索取GPU數量
#SBATCH --nodes=1                ## 索取 幾節點
#SBATCH --ntasks-per-node=1      ## 新增這行，解決 INFO 警告
#SBATCH --cpus-per-task=4        ## 每個 gpu 索取 4 CPUs
#SBATCH --account="ACD114152"    ## PROJECT_ID
#SBATCH --partition=gp4d                 ## 【關鍵修改】改為 gp4d (支援最長 4 天)
#SBATCH --time=96:00:00                  ## 【關鍵修改】設定時間 (需小於 96 小時)
#SBATCH --output=slurm-%j.out            ## 標準輸出日誌 (%j 會變成 Job ID)
#SBATCH --error=slurm-%j.err             ## 錯誤輸出日誌
model purge
module load singularity

# 定義變數 (Level 3 整理)
SIF_PATH="/work/t113618009/ssast_hub/tune_hub.sif"
# 這是你的 JSON 列表
TRAIN_JSON="/work/t113618009/ssast_hub/ssl_fs_tml_train.json"
TEST_JSON="/work/t113618009/ssast_hub/ssl_fs_tml_val.json"
# 這是你的 .pt 檔所在的根目錄 (雖然 JSON 是絕對路徑，但設對比較保險)
DATA_ROOT="/work/t113618009/spectrogram_pt_name"
DATA_VAL_ROOT="/work/t113618009/spectrogram_pt_name"
LABEL_CSV="/work/t113618009/ssast_hub/ssl_labels_indices.csv"

# 執行指令
# 執行指令
singularity exec --nv \
--bind /work/t113618009:/work/t113618009 \
"$SIF_PATH" \
python /work/t113618009/ssast_hub/ssast-main/src/run.py \
  --data_train "$TRAIN_JSON" \
  --data_val "$TEST_JSON" \
  --data_dir "$DATA_ROOT" \
  --data_val_dir "$DATA_VAL_ROOT" \
  --label-csv "$LABEL_CSV" \
  --dataset precomputed \
  --num-workers 8 \
  --dataset_mean -7.4482 \
  --dataset_std 2.4689 \
  --target_length 1024 \
  --num_mel_bins 128 \
  --fshape 16 \
  --tshape 16 \
  --fstride 16 \
  --tstride 16 \
  --model_size small \
  --task pretrain_joint \
  --mask_patch 400 \
  --cluster_factor 3 \
  --n-epochs 400 \
  --batch-size 128 \
  --lr 0.0001 \
  --weight_decay 0.05 \
  --warmup_steps 10000 \
  --exp-dir "/work/t113618009/ssast_hub/exp/pretrain_fs_tml_small" \