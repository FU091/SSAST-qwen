#!/bin/bash
#SBATCH --job-name=ssast_pretrain        ## 作業名稱
#SBATCH --gres=gpu:4                     ## 要求 4 張 GPU
#SBATCH --nodes=1                        ## 要求 1 個節點
#SBATCH --cpus-per-task=4                ## 每個 task 配 4 CPUs
#SBATCH --account="ACD114152"            ## 計畫 ID，需填入正確的專案代碼
#SBATCH --time=72:00:00                  ## 最長執行時間 (這裡設 72 小時)
#SBATCH --partition=normal               ## 使用 normal queue

##- normal queue：一般正式使用的佇列，適合長時間訓練或正式任務。
##- gtest queue：測試用，通常限制時間很短（例如 10 分鐘），方便快速檢查腳本是否能跑。
##- 其他 queue（例如 gp1d, gp2d, gp4d）：代表最長可跑 1 天、2 天、4 天的 GPU 分區。


# Debug 模式
set -x

# 使用 Singularity 容器執行程式
singularity exec --nv \
  --bind /home/t113618009/SSAST-Finetune-Project:/AST \
  /home/t113618009/SSAST-Finetune-Project/dalta.sif \
  python /AST/pretrain_dalta_ssl.py