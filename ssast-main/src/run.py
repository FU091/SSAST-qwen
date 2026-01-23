# -*- coding: utf-8 -*-
# @Time    : 6/11/21 12:57 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : run.py

import argparse
import os
import ast
import pickle
import sys
import time
import torch
import numpy as np
from torch.utils.data import WeightedRandomSampler
import warnings

warnings.filterwarnings("ignore", message=".*weights_only=False.*")

# --- 1. 路徑與環境設定 ---
# 檢查是否在 Docker 環境中
if os.path.exists('/ssast_hub'):  # Docker 環境標誌
    print("[Info] Running in Docker environment")
    BASE_DIR = '/ssast_hub'
    # 注意：這裡對應你 docker run -v 掛載進去的路徑
    TRAIN_DATA_DIR = '/data/train'
    VAL_DATA_DIR = '/data/val'
    TRAIN_JSON_PATH = '/ssast_hub/combined_train_data.json'
    VAL_JSON_PATH = '/ssast_hub/test.json'
    LABEL_CSV_PATH = '/ssast_hub/class_labels_indices.csv'
else:
    # 本地環境中的路徑
    print("[Info] Running in Local environment")
    BASE_DIR = 'C:/Users/Lin/Desktop/2_code/ssast_hub'
    TRAIN_DATA_DIR = 'D:/spectrogram_pt_name'
    VAL_DATA_DIR = 'D:/val_spectrogram_pt_name'
    TRAIN_JSON_PATH = os.path.join(BASE_DIR, 'combined_train_data.json')
    VAL_JSON_PATH = os.path.join(BASE_DIR, 'test.json')
    LABEL_CSV_PATH = os.path.join(BASE_DIR, 'class_labels_indices.csv')

# ### 修改點 1: 修正 import 路徑 ###
# 獲取 run.py 所在的目錄 (即 src 目錄)
current_file_path = os.path.dirname(os.path.abspath(__file__))
if current_file_path not in sys.path:
    sys.path.append(current_file_path)

# 模組導入 (確保這些檔案都在 src 資料夾內)
try:
    from dataloader_pt_reader import PrecomputedDataset
except ImportError:
    print("[Warning] Could not import PrecomputedDataset from dataloader_pt_reader")
    PrecomputedDataset = None

import dataloader
from models import ASTModel
from traintest import train, validate
try:
    from traintest_mask import trainmask
except ImportError:
    print("[Warning] Could not import trainmask from traintest_mask")
    trainmask = None

# 打印系統資訊
node_name = os.uname()[1] if hasattr(os, 'uname') else os.environ.get('COMPUTERNAME', 'unknown')
print("I am process %s, running on %s: starting (%s)" % (os.getpid(), node_name, time.asctime()))

# --- 2. 定義 Parser ---
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--data_train", type=str, default=None, help="training data json")
parser.add_argument("--data_val", type=str, default=None, help="validation data json")
parser.add_argument("--data_eval", type=str, default=None, help="evaluation data json")
parser.add_argument("--label-csv", type=str, default='', help="csv with class labels")
parser.add_argument("--n_class", type=int, default=527, help="number of classes")
parser.add_argument("--dataset", type=str, help="the dataset used for training")
parser.add_argument("--dataset_mean", type=float, help="the dataset mean, used for input normalization")
parser.add_argument("--dataset_std", type=float, help="the dataset std, used for input normalization")
parser.add_argument("--target_length", type=int, help="the input length in frames")
parser.add_argument("--num_mel_bins", type=int, default=128, help="number of input mel bins")
parser.add_argument("--exp-dir", type=str, default="", help="directory to dump experiments")
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--warmup', help='if use warmup learning rate scheduler', type=ast.literal_eval, default='True')
parser.add_argument("--optim", type=str, default="adam", help="training optimizer", choices=["sgd", "adam"])
parser.add_argument('-b', '--batch-size', default=12, type=int, metavar='N', help='mini-batch size')
parser.add_argument('-w', '--num-workers', default=16, type=int, metavar='NW', help='# of workers for dataloading')
parser.add_argument("--n-epochs", type=int, default=1, help="number of maximum training epochs")
parser.add_argument("--lr_patience", type=int, default=1, help="how many epoch to wait to reduce lr")
parser.add_argument('--adaptschedule', help='if use adaptive scheduler ', type=ast.literal_eval, default='False')
parser.add_argument("--n-print-steps", type=int, default=100, help="number of steps to print statistics")
parser.add_argument('--save_model', help='save the models or not', type=ast.literal_eval)
parser.add_argument('--freqm', help='frequency mask max length', type=int, default=0)
parser.add_argument('--timem', help='time mask max length', type=int, default=0)
parser.add_argument("--mixup", type=float, default=0, help="mixup ratio")
parser.add_argument("--bal", type=str, default=None, help="use balanced sampling or not")
parser.add_argument("--fstride", type=int, help="soft split freq stride")
parser.add_argument("--tstride", type=int, help="soft split time stride")
parser.add_argument("--fshape", type=int, help="shape of patch on the frequency dimension")
parser.add_argument("--tshape", type=int, help="shape of patch on the time dimension")
parser.add_argument('--model_size', help='the size of AST models', type=str, default='base384')
parser.add_argument("--task", type=str, default='ft_cls', help="task", choices=["ft_avgtok", "ft_cls", "pretrain_mpc", "pretrain_mpg", "pretrain_joint"])
parser.add_argument('--mask_patch', help='how many patches to mask', type=int, default=400)
parser.add_argument("--cluster_factor", type=int, default=3, help="mask clutering factor")
parser.add_argument("--epoch_iter", type=int, default=2000, help="iterations to verify and save")
parser.add_argument("--pretrained_mdl_path", type=str, default=None, help="pretrained model path")
parser.add_argument("--head_lr", type=int, default=1, help="factor of mlp-head_lr/lr")
parser.add_argument("--noise", help='if augment noise', type=ast.literal_eval)
parser.add_argument("--metrics", type=str, default="mAP", help="main evaluation metrics", choices=["mAP", "acc"])
parser.add_argument("--lrscheduler_start", default=10, type=int, help="when to start decay")
parser.add_argument("--lrscheduler_step", default=5, type=int, help="decay step")
parser.add_argument("--lrscheduler_decay", default=0.5, type=float, help="decay ratio")
parser.add_argument("--wa", help='if do weight averaging', type=ast.literal_eval)
parser.add_argument("--wa_start", type=int, default=16, help="wa start epoch")
parser.add_argument("--wa_end", type=int, default=30, help="wa end epoch")
parser.add_argument("--loss", type=str, default="BCE", help="loss function", choices=["BCE", "CE"])
# [修改 1] 新增 dataset_root 參數，用來接收你的 DATA_ROOT
parser.add_argument("--dataset_root", type=str, default=None, help="Root directory for unified .pt files")
parser.add_argument("--data_dir", type=str, default=None, help="directory for precomputed .pt files")
parser.add_argument("--data_val_dir", type=str, default=None, help="directory for validation .pt files")

# ### 修改點 2: 移除原本上方的 sys.argv 修改邏輯，改為在此處判斷 ###
# 只有當沒有提供參數 (sys.argv只有檔名) 時，才載入預設值
if len(sys.argv) == 1:
    print("----------------------------------------------------------------")
    print("[Auto-Config] No arguments provided. Loading default configuration...")
    print(f"[Auto-Config] Using Training Data: {TRAIN_JSON_PATH}")
    print(f"[Auto-Config] Using Precomputed Dir: {TRAIN_DATA_DIR}")
    print("----------------------------------------------------------------")

    default_args = [
        '--dataset', 'precomputed',
        '--data_train', TRAIN_JSON_PATH,
        '--data_val', VAL_JSON_PATH,
        '--data_dir', TRAIN_DATA_DIR,
        '--data_val_dir', VAL_DATA_DIR,
        '--label-csv', LABEL_CSV_PATH,
        '--dataset_mean', '-7.4482',
        '--dataset_std', '2.4689',
        '--target_length', '1024',
        '--num_mel_bins', '128',
        '--fshape', '16',
        '--tshape', '16',
        '--fstride', '16',
        '--tstride', '16',
        '--model_size', 'base',
        '--task', 'pretrain_joint',
        '--mask_patch', '400',
        '--batch-size', '24',
        '--lr', '0.0001',
        '--n-epochs', '10',
        '--exp-dir', os.path.join(BASE_DIR, 'exp'),
        '--save_model', 'True',
        '--freqm', '0',
        '--timem', '0',
        '--mixup', '0',
        '--n-print-steps', '100',
        '--lr_patience', '2',
        '--epoch_iter', '4000'
    ]
    args = parser.parse_args(default_args)
else:
    # 如果使用者有輸入指令參數，則使用輸入的參數
    args = parser.parse_args()

# 確保 dataset 參數不為 None，避免 dataloader.py 中的錯誤
if args.dataset is None:
    args.dataset = 'default'

# --- Main Execution ---
if __name__ == '__main__':
    # ... (後面的程式碼保持不變) ...
    # 這裡加入一行 debug 確保參數有吃進去
    print(f"[Debug] Configuration Loaded - data_train: {args.data_train}")

    # 設定 Audio Conf
    # 設定 Audio Conf (修正後)
    # 這裡加入了 'data_dir': args.data_dir
    audio_conf = {
            'num_mel_bins': args.num_mel_bins, 
            'target_length': args.target_length, 
            'freqm': args.freqm, 
            'timem': args.timem, 
            'mixup': args.mixup, 
            'dataset': args.dataset, 
            'mode':'train', 
            'mean':args.dataset_mean, 
            'std':args.dataset_std, 
            'noise':args.noise, 
            # 修改這裡：優先使用 dataset_root
            'data_dir': args.dataset_root if args.dataset_root else args.data_dir
        }
        
    val_audio_conf = {
            'num_mel_bins': args.num_mel_bins, 
            'target_length': args.target_length, 
            'freqm': 0, 
            'timem': 0, 
            'mixup': 0, 
            'dataset': args.dataset, 
            'mode': 'evaluation', 
            'mean': args.dataset_mean, 
            'std': args.dataset_std, 
            'noise': False,
            # 修改這裡：驗證集通常也放在同一個 root
            'data_dir': args.dataset_root if args.dataset_root else args.data_val_dir
        }
     
    # 1. 建立 Dataset
    if args.dataset == 'precomputed':
        print('Using precomputed dataset (reading .pt files)...')
        if PrecomputedDataset is None:
            raise RuntimeError("PrecomputedDataset is not available but dataset='precomputed' was specified")

        # [修改 3] 更新實例化方式，配合 dataloader_pt_reader.py 的新介面
        # 必須傳入: label_csv (讀取標籤對照), data_dir (讀取檔案), mixup (訓練用)

        train_dataset = PrecomputedDataset(
            dataset_json_file=args.data_train,
            audio_conf=audio_conf,
            data_dir=audio_conf['data_dir'],  # 這裡會拿到上面的 dataset_root
            label_csv=args.label_csv,         # 傳入 CSV 路徑
            mixup=args.mixup                  # 傳入 Mixup 參數
        )

        val_dataset = PrecomputedDataset(
            dataset_json_file=args.data_val,
            audio_conf=val_audio_conf,
            data_dir=val_audio_conf['data_dir'],
            label_csv=args.label_csv,
            mixup=0.0                         # 驗證集不使用 Mixup
        )

    else:
        # 【關鍵修改在這裡！】
        # 這是你目前指令會進入的區塊
        # 我們把 CLI 傳進來的 args.data_dir (JSON路徑) 強制塞給 dataset_json_file

        print(f"Loading Train JSON from: {args.data_train}")
        # 確保 audio_conf 中包含 dataset 參數以避免 dataloader.py 中的錯誤
        audio_conf['dataset'] = args.dataset
        train_dataset = dataloader.AudioDataset(
            dataset_json_file=args.data_train,  # 對應 .sh 的 --data_train
            label_csv=args.label_csv,
            audio_conf=audio_conf,
        )

        print(f"Loading Val JSON from: {args.data_val}")
        # 確保 val_audio_conf 中包含 dataset 參數以避免 dataloader.py 中的錯誤
        val_audio_conf['dataset'] = args.dataset
        val_dataset = dataloader.AudioDataset(
            dataset_json_file=args.data_val, # 對應 .sh 的 --data_val
            label_csv=args.label_csv,
            audio_conf=val_audio_conf,
        )

    # 2. 建立 Training DataLoader
    if args.dataset == 'precomputed':
        if args.bal == 'bal':
            print('Warning: balanced sampling is not supported for precomputed dataset, using regular shuffle.')
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False, drop_last=True)
    else:
        # 這裡是處理原始音訊資料的地方
        if args.bal == 'bal':
            print('balanced sampler is being used')
            # --- 修改開始：加入檔案檢查 ---
            weight_file_path = args.data_train[:-5] + '_weight.csv'
            if os.path.exists(weight_file_path):
                samples_weight = np.loadtxt(weight_file_path, delimiter=',')
                sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)
                train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=False, drop_last=True)
            else:
                print(f'[Warning] Weight file {weight_file_path} not found! Fallback to regular shuffle.')
                train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False, drop_last=True)
            # --- 修改結束 ---
        else:
            print('balanced sampler is not used')
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False, drop_last=True)

    # 3. 建立 Validation DataLoader
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers, pin_memory=False)

    print('Now train with {:s} with {:d} samples, evaluate with {:d} samples'.format(args.dataset, len(train_loader.dataset), len(val_loader.dataset)))

    # 4. 初始化模型
    if 'pretrain' in args.task:
        cluster = (args.num_mel_bins != args.fshape)
        print('Cluster masking: %s' % cluster)
        audio_model = ASTModel(fshape=args.fshape, tshape=args.tshape, fstride=args.fstride, tstride=args.tstride, input_fdim=args.num_mel_bins, input_tdim=args.target_length, model_size=args.model_size, pretrain_stage=True)
    else:
        audio_model = ASTModel(label_dim=args.n_class, fshape=args.fshape, tshape=args.tshape, fstride=args.fstride, tstride=args.tstride, input_fdim=args.num_mel_bins, input_tdim=args.target_length, model_size=args.model_size, pretrain_stage=False, load_pretrained_mdl_path=args.pretrained_mdl_path)

    if not isinstance(audio_model, torch.nn.DataParallel):
        audio_model = torch.nn.DataParallel(audio_model)

    # 建立實驗目錄
    if not os.path.exists("%s/models" % args.exp_dir):
        os.makedirs("%s/models" % args.exp_dir)
    with open("%s/args.pkl" % args.exp_dir, "wb") as f:
        pickle.dump(args, f)



    # 5. 開始訓練
    if 'pretrain' not in args.task:
        print('Now starting fine-tuning for {:d} epochs'.format(args.n_epochs))
        train(audio_model, train_loader, val_loader, args)
    else:
        print('Now starting self-supervised pretraining for {:d} epochs'.format(args.n_epochs))
        if trainmask is None:
            raise RuntimeError("trainmask function is not available for pretraining tasks")
        try:
            trainmask(audio_model, train_loader, val_loader, args)
        except RuntimeError as e:
            if "CUDA error" in str(e) or "no kernel image" in str(e).lower():
                print(f"CUDA error fallback to CPU: {e}")
                audio_model_cpu = audio_model.module.to("cpu")
                audio_model_cpu = torch.nn.DataParallel(audio_model_cpu)
                trainmask(audio_model_cpu, train_loader, val_loader, args)
            else:
                raise e

    # 6. Evaluation (如果是 Fine-tuning 且有評測集)
    if args.data_eval is not None and 'pretrain' not in args.task:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sd = torch.load(args.exp_dir + '/models/best_audio_model.pth', map_location=device)
        audio_model.load_state_dict(sd, strict=False)

        print('---------------evaluate on validation set---------------')
        stats, _ = validate(audio_model, val_loader, args, 'valid_set')
        
        print('---------------evaluate on evaluation set---------------')
        # 【重要修改】根據 dataset 類型選擇正確的資料集類別
        if args.dataset == 'precomputed':
            if PrecomputedDataset is None:
                raise RuntimeError("PrecomputedDataset is not available but dataset='precomputed' was specified")

            eval_dataset = PrecomputedDataset(
                dataset_json_file=args.data_eval,
                audio_conf=val_audio_conf,
                data_dir=val_audio_conf['data_dir'],
                label_csv=args.label_csv,
                mixup=0.0  # 評估集不使用 Mixup
            )
            eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size*2, shuffle=False, num_workers=args.num_workers, pin_memory=False)
        else:
            eval_loader = torch.utils.data.DataLoader(dataloader.AudioDataset(args.data_eval, label_csv=args.label_csv, audio_conf=val_audio_conf), batch_size=args.batch_size*2, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        stats, _ = validate(audio_model, eval_loader, args, 'eval_set')