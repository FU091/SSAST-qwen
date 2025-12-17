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
# 過濾掉 torch.load 的 FutureWarning
warnings.filterwarnings("ignore", message=".*weights_only=False.*")

# 預設配置選項
USE_PRECOMPUTED_CONFIG = True  # 設置為 True 啟用簡化配置

# 檢查是否在 Docker 環境中
if os.path.exists('/ssast'):  # Docker 環境標誌
    BASE_DIR = '/ssast'
    TRAIN_DATA_DIR = '/data/train'
    VAL_DATA_DIR = '/data/val'
    TRAIN_JSON_PATH = '/ssast/combined_train_data.json'
    VAL_JSON_PATH = '/ssast/test.json'
    LABEL_CSV_PATH = '/ssast/class_labels_indices.csv'
else:
    # 本地環境中的路徑
    BASE_DIR = 'C:/Users/Lin/Desktop/2_code/ssast_hub'
    TRAIN_DATA_DIR = 'D:/spectrogram_pt_name'
    VAL_DATA_DIR = 'D:/val_spectrogram_pt_name'
    TRAIN_JSON_PATH = os.path.join(BASE_DIR, 'combined_train_data.json')
    VAL_JSON_PATH = os.path.join(BASE_DIR, 'test.json')
    LABEL_CSV_PATH = os.path.join(BASE_DIR, 'class_labels_indices.csv')

# 動態注入預設參數
if USE_PRECOMPUTED_CONFIG and any('precomputed' in arg for arg in sys.argv + ['--dataset', 'precomputed']):
    default_args = [
        '--dataset', 'precomputed',
        '--data-train', TRAIN_JSON_PATH,
        '--data-val', VAL_JSON_PATH,
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
        '--save_model', 'False',
        '--freqm', '0',
        '--timem', '0',
        '--mixup', '0',
        '--n-print-steps', '100',
        '--lr_patience', '2',
        '--epoch_iter', '4000'
    ]

    for i in range(0, len(default_args), 2):
        arg = default_args[i]
        if arg not in sys.argv:
            sys.argv.extend(default_args[i:i+2])

# 正確設置 Python 路徑以導入 src 內容
current_file_path = os.path.dirname(os.path.abspath(__file__))
# 假設此 run.py 在根目錄，src 在子目錄；或根據您的邏輯向上跳兩層找 ssast-main/src
sys.path.append(os.path.join(current_file_path, 'ssast-main', 'src'))

# 模組導入
from dataloader_pt_reader import PrecomputedDataset
import dataloader
from models import ASTModel
from traintest import train, validate
from traintest_mask import trainmask

# 打印系統資訊
node_name = os.uname()[1] if hasattr(os, 'uname') else os.environ.get('COMPUTERNAME', 'unknown')
print("I am process %s, running on %s: starting (%s)" % (os.getpid(), node_name, time.asctime()))

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--data-train", type=str, default=None, help="training data json")
parser.add_argument("--data-val", type=str, default=None, help="validation data json")
parser.add_argument("--data-eval", type=str, default=None, help="evaluation data json")
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
parser.add_argument("--data_dir", type=str, default=None, help="directory for precomputed .pt files")
parser.add_argument("--data_val_dir", type=str, default=None, help="directory for validation .pt files")

args = parser.parse_args()

# 設定 Audio Conf
audio_conf = {'num_mel_bins': args.num_mel_bins, 'target_length': args.target_length, 'freqm': args.freqm, 'timem': args.timem, 'mixup': args.mixup, 'dataset': args.dataset, 'mode':'train', 'mean':args.dataset_mean, 'std':args.dataset_std, 'noise':args.noise}
val_audio_conf = {'num_mel_bins': args.num_mel_bins, 'target_length': args.target_length, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': args.dataset, 'mode': 'evaluation', 'mean': args.dataset_mean, 'std': args.dataset_std, 'noise': False}

# 1. 建立 Dataset
if args.dataset == 'precomputed':
    print('Using precomputed dataset (reading .pt files)...')
    audio_conf['data_dir'] = args.data_dir if args.data_dir else args.data_train
    val_audio_conf['data_dir'] = args.data_val_dir if args.data_val_dir else (args.data_val if args.data_val != args.data_train else args.data_dir)

    train_dataset = PrecomputedDataset(data_dir=audio_conf['data_dir'], dataset_json_file=args.data_train, audio_conf=audio_conf)
    val_dataset = PrecomputedDataset(data_dir=val_audio_conf['data_dir'], dataset_json_file=args.data_val, audio_conf=val_audio_conf)
else:
    train_dataset = dataloader.AudioDataset(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf)
    val_dataset = dataloader.AudioDataset(args.data_val, label_csv=args.label_csv, audio_conf=val_audio_conf)

# 2. 建立 Training DataLoader
if args.dataset == 'precomputed':
    if args.bal == 'bal':
        print('Warning: balanced sampling is not supported for precomputed dataset, using regular shuffle.')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False, drop_last=True)
else:
    if args.bal == 'bal':
        print('balanced sampler is being used')
        samples_weight = np.loadtxt(args.data_train[:-5]+'_weight.csv', delimiter=',')
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=False, drop_last=True)
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


if __name__ == '__main__':
    # 5. 開始訓練
    if 'pretrain' not in args.task:
        print('Now starting fine-tuning for {:d} epochs'.format(args.n_epochs))
        train(audio_model, train_loader, val_loader, args)
    else:
        print('Now starting self-supervised pretraining for {:d} epochs'.format(args.n_epochs))
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
        eval_loader = torch.utils.data.DataLoader(dataloader.AudioDataset(args.data_eval, label_csv=args.label_csv, audio_conf=val_audio_conf), batch_size=args.batch_size*2, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        stats, _ = validate(audio_model, eval_loader, args, 'eval_set')