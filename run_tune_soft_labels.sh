#!/bin/bash
#SBATCH --job-name=ssast_finetune    ## job name
#SBATCH --nodes=1                ## 索取 1 節點
#SBATCH --ntasks-per-node=1      ## 每個節點 1 個任務
#SBATCH --cpus-per-task=8        ## 每個 gpu 最多可以索取 4 CPUs
#SBATCH --gres=gpu:2             ## 每個節點索取 1 GPU
#SBATCH --account="ACD114152"    ## PROJECT_ID
#SBATCH --partition=gp2d         ## 使用 gp4d partition (最長跑 天)
#SBATCH --time=48:00:00          ## 設定時間 (需小於 96 小時)
#SBATCH -o /work/t113618009/ssast_hub/logs/soft_labels_%j.out
#SBATCH -e /work/t113618009/ssast_hub/logs/soft_labels_%j.err

module purge
# 檢查並加載 Singularity 模塊
if module avail singularity 2>/dev/null; then
    module load singularity
elif module avail singularityce 2>/dev/null; then
    module load singularityce
elif module avail apptainer 2>/dev/null; then
    module load apptainer
else
    echo "No container runtime module found, attempting direct execution"
fi

SIF_PATH="/work/t113618009/ssast_hub/tune_hub.sif"
# 這是你的 JSON 列表 (從 finetune_stratified_final 目錄)
TRAIN_JSON="/work/t113618009/ssast_hub/finetune_stratified_final/train.json"
VAL_JSON="/work/t113618009/ssast_hub/finetune_stratified_final/val.json"
TEST_JSON="/work/t113618009/ssast_hub/finetune_stratified_final/test.json"
# 這是你的 .pt 檔所在的根目錄
DATA_ROOT="/work/t113618009/spectrogram_pt_name"
# 使用根目錄的標籤文件 (有12個類別)
LABEL_CSV="/work/t113618009/ssast_hub/class_labels_indices.csv"

# 執行指令 - 使用正確的12類標籤文件和支援軟標籤的格式
singularity exec --nv \
--bind /work/t113618009:/work/t113618009 \
--bind /work/t113618009/spectrogram_pt_name:/work/t113618009/spectrogram_pt_name \
--bind /work/t113618009/ssast_hub/finetune_stratified_final:/work/t113618009/ssast_hub/finetune_stratified_final \
--bind /work/t113618009/ssast_hub/class_labels_indices.csv:/work/t113618009/ssast_hub/class_labels_indices.csv \
"$SIF_PATH" \
python /work/t113618009/ssast_hub/ssast-main/src/run.py \
  --dataset precomputed \
  --data_train "$TRAIN_JSON" \
  --data_val "$VAL_JSON" \
  --data_eval "$TEST_JSON" \
  --label-csv "$LABEL_CSV" \
  --num-workers 8 \
  --n_class 12 \
  --dataset_mean -7.4482 \
  --dataset_std 2.4689 \
  --target_length 1024 \
  --num_mel_bins 128 \
  --exp-dir "/work/t113618009/ssast_hub/exp/finetune/epoch30/ssl" \
  --lr 0.00005 \
  --n-epochs 40 \
  --batch-size 12 \
  --save_model True \
  --freqm 24 \
  --timem 24 \
  --mixup 0.0 \
  --bal none \
  --fstride 10 \
  --tstride 10 \
  --fshape 16 \
  --tshape 16 \
  --warmup False \
  --task ft_cls \
  --model_size base \
  --adaptschedule False \
  --pretrained_mdl_path "/work/t113618009/ssast_hub/exp/pretrain_test/models/audio_model.134.pth" \
  --dataset_root "$DATA_ROOT" \
  --head_lr 10 \
  --noise False \
  --lrscheduler_start 10 \
  --lrscheduler_step 5 \
  --lrscheduler_decay 0.5 \
  --wa False \
  --loss BCE \
  --metrics mAP