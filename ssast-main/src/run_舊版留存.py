# run.py
import argparse
import os
import ast
import pickle
import sys
import time
import torch
from torch.utils.data import WeightedRandomSampler
from dataloader_pt_reader import PrecomputedDataset
basepath = os.path.dirname(os.path.dirname(sys.path[0]))
sys.path.append(basepath)
import dataloader
from models import ASTModel
import numpy as np
from traintest import train, validate
from traintest_mask import trainmask

from torch.utils.data import WeightedRandomSampler

# 預設配置選項
USE_PRECOMPUTED_CONFIG = True  # 設置為 True 啟用簡化配置

if USE_PRECOMPUTED_CONFIG and any('precomputed' in arg for arg in sys.argv):
    # 預設參數配置（適用於預計算 .pt 檔案模式）
    default_args = [
        '--dataset', 'precomputed',
        '--data-train', 'C:/Users/Lin/Desktop/2_code/ssast_hub/combined_train_data.json',
        '--data-val', 'C:/Users/Lin/Desktop/2_code/ssast_hub/test.json',
        '--data_dir', 'D:/spectrogram_pt_name',
        '--data_val_dir', 'D:/val_spectrogram_pt_name',
        '--label-csv', 'C:/Users/Lin/Desktop/2_code/ssast_hub/class_labels_indices.csv',
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
        '--batch_size', '24',
        '--lr', '0.0001',
        '--n-epochs', '10',
        '--exp-dir', 'C:/Users/Lin/Desktop/2_code/ssast_hub/exp',
        '--save_model', 'False',
        '--freqm', '0',
        '--timem', '0',
        '--mixup', '0',
        '--n-print-steps', '100',
        '--lr_patience', '2',
        '--epoch_iter', '4000'
    ]

    # 只添加缺失的參數
    for i in range(0, len(default_args), 2):
        arg_name = default_args[i]
        if arg_name not in sys.argv:
            sys.argv.extend(default_args[i:i+2])

# 設定路徑以便匯入 dataloader_pt_reader
basepath = os.path.dirname(os.path.dirname(os.path.abspath(sys.path[0])))
sys.path.append(basepath)

# 確保 src 目錄也被加入，如果 dataloader 在 src 內
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
if os.path.exists(src_path):
    sys.path.append(src_path)

try:
    from dataloader_pt_reader import PrecomputedDataset
    print("✅ 成功匯入 PrecomputedDataset")
except ImportError as e:
    print(f"❌ 匯入失敗: {e}")
    print(f"目前的 sys.path: {sys.path}")
# -*- coding: utf-8 -*-
# @Time    : 6/11/21 12:57 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : run.py


print("I am process %s, running on %s: starting (%s)" % (os.getpid(), os.name()[1], time.asctime()))

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
parser.add_argument('-w', '--num-workers', default=16, type=int, metavar='NW', help='# of workers for dataloading (default: 32)')
parser.add_argument("--n-epochs", type=int, default=1, help="number of maximum training epochs")
# only used in pretraining stage or from-scratch fine-tuning experiments
parser.add_argument("--lr_patience", type=int, default=1, help="how many epoch to wait to reduce lr if mAP doesn't improve")
parser.add_argument('--adaptschedule', help='if use adaptive scheduler ', type=ast.literal_eval, default='False')

parser.add_argument("--n-print-steps", type=int, default=100, help="number of steps to print statistics")
parser.add_argument('--save_model', help='save the models or not', type=ast.literal_eval)

parser.add_argument('--freqm', help='frequency mask max length', type=int, default=0)
parser.add_argument('--timem', help='time mask max length', type=int, default=0)
parser.add_argument("--mixup", type=float, default=0, help="how many (0-1) samples need to be mixup during training")
parser.add_argument("--bal", type=str, default=None, help="use balanced sampling or not")
# the stride used in patch spliting, e.g., for patch size 16*16, a stride of 16 means no overlapping, a stride of 10 means overlap of 6.
# during self-supervised pretraining stage, no patch split overlapping is used (to aviod shortcuts), i.e., fstride=fshape and tstride=tshape
# during fine-tuning, using patch split overlapping (i.e., smaller {f,t}stride than {f,t}shape) improves the performance.
# it is OK to use different {f,t} stride in pretraining and finetuning stages (though fstride is better to keep the same)
# but {f,t}stride in pretraining and finetuning stages must be consistent.
parser.add_argument("--fstride", type=int, help="soft split freq stride, overlap=patch_size-stride")
parser.add_argument("--tstride", type=int, help="soft split time stride, overlap=patch_size-stride")
parser.add_argument("--fshape", type=int, help="shape of patch on the frequency dimension")
parser.add_argument("--tshape", type=int, help="shape of patch on the time dimension")
parser.add_argument('--model_size', help='the size of AST models', type=str, default='base384')

parser.add_argument("--task", type=str, default='ft_cls', help="pretraining or fine-tuning task", choices=["ft_avgtok", "ft_cls", "pretrain_mpc", "pretrain_mpg", "pretrain_joint"])


# Add new arguments to specify the path to directories for precomputed dataset
parser.add_argument("--data_dir", type=str, default=None, help="directory for precomputed .pt files (for precomputed dataset mode)")
parser.add_argument("--data_val_dir", type=str, default=None, help="directory for precomputed validation .pt files (for precomputed dataset mode)")

# pretraining augments
#parser.add_argument('--pretrain_stage', help='True for self-supervised pretraining stage, False for fine-tuning stage', type=ast.literal_eval, default='False')
parser.add_argument('--mask_patch', help='how many patches to mask (used only for ssl pretraining)', type=int, default=400)
parser.add_argument("--cluster_factor", type=int, default=3, help="mask clutering factor")
parser.add_argument("--epoch_iter", type=int, default=2000, help="for pretraining, how many iterations to verify and save models")

# fine-tuning arguments
parser.add_argument("--pretrained_mdl_path", type=str, default=None, help="the ssl pretrained models path")
parser.add_argument("--head_lr", type=int, default=1, help="the factor of mlp-head_lr/lr, used in some fine-tuning experiments only")
parser.add_argument("--noise", help='if augment noise in finetuning', type=ast.literal_eval)
parser.add_argument("--metrics", type=str, default="mAP", help="the main evaluation metrics in finetuning", choices=["mAP", "acc"])
parser.add_argument("--lrscheduler_start", default=10, type=int, help="when to start decay in finetuning")
parser.add_argument("--lrscheduler_step", default=5, type=int, help="the number of step to decrease the learning rate in finetuning")
parser.add_argument("--lrscheduler_decay", default=0.5, type=float, help="the learning rate decay ratio in finetuning")
parser.add_argument("--wa", help='if do weight averaging in finetuning', type=ast.literal_eval)
parser.add_argument("--wa_start", type=int, default=16, help="which epoch to start weight averaging in finetuning")
parser.add_argument("--wa_end", type=int, default=30, help="which epoch to end weight averaging in finetuning")
parser.add_argument("--loss", type=str, default="BCE", help="the loss function for finetuning, depend on the task", choices=["BCE", "CE"])

args = parser.parse_args()

# # dataset spectrogram mean and std, used to normalize the input
# norm_stats = {'librispeech':[-4.2677393, 4.5689974], 'howto100m':[-4.2677393, 4.5689974], 'audioset':[-4.2677393, 4.5689974], 'esc50':[-6.6268077, 5.358466], 'speechcommands':[-6.845978, 5.5654526]}
# target_length = {'librispeech': 1024, 'howto100m':1024, 'audioset':1024, 'esc50':512, 'speechcommands':128}
# # if add noise for data augmentation, only use for speech commands
# noise = {'librispeech': False, 'howto100m': False, 'audioset': False, 'esc50': False, 'speechcommands':True}

audio_conf = {'num_mel_bins': args.num_mel_bins, 'target_length': args.target_length, 'freqm': args.freqm, 'timem': args.timem, 'mixup': args.mixup, 'dataset': args.dataset,
              'mode':'train', 'mean':args.dataset_mean, 'std':args.dataset_std, 'noise':args.noise}

val_audio_conf = {'num_mel_bins': args.num_mel_bins, 'target_length': args.target_length, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': args.dataset,
                  'mode': 'evaluation', 'mean': args.dataset_mean, 'std': args.dataset_std, 'noise': False}

# if use balanced sampling, note - self-supervised pretraining should not use balance sampling as it implicitly leverages the label information.

# ================= 替換開始 =================
# 1. 根據 dataset 參數決定使用哪種 Dataset
if args.dataset == 'precomputed':
    print('Using precomputed dataset (reading .pt files)...')

    # Create audio_conf with data_dir info for the PrecomputedDataset
    audio_conf['data_dir'] = args.data_dir if args.data_dir else args.data_train
    val_audio_conf['data_dir'] = args.data_val_dir if args.data_val_dir else (args.data_val if args.data_val != args.data_train else args.data_dir)

    # For precomputed dataset: Use the provided directories and JSON files for mapping
    train_dataset = PrecomputedDataset(
        data_dir=audio_conf['data_dir'],
        dataset_json_file=args.data_train,
        audio_conf=audio_conf
    )
    val_dataset = PrecomputedDataset(
        data_dir=val_audio_conf['data_dir'],
        dataset_json_file=args.data_val,
        audio_conf=val_audio_conf
    )

else:
    # 舊有的邏輯：使用原始 AudioDataset 讀取 JSON 和 Wav
    train_dataset = dataloader.AudioDataset(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf)
    val_dataset = dataloader.AudioDataset(args.data_val, label_csv=args.label_csv, audio_conf=val_audio_conf)

# 2. 建立 Training DataLoader
if args.dataset == 'precomputed':
    # Precomputed 模式下，暫時不使用 weighted sampler (bal)，直接 shuffle
    # Also handle the case where balanced sampling was requested but isn't supported
    if args.bal == 'bal':
        print('Warning: balanced sampling is not supported for precomputed dataset, using regular shuffle instead.')
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False, drop_last=True)
else:
    # 原始邏輯：支援平衡採樣 (Balanced Sampling)
    if args.bal == 'bal':
        print('balanced sampler is being used')
        samples_weight = np.loadtxt(args.data_train[:-5]+'_weight.csv', delimiter=',')
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=False, drop_last=True)
    else:
        print('balanced sampler is not used')
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False, drop_last=True)

# 3. 建立 Validation DataLoader (共用)
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers, pin_memory=False)
# ================= 替換結束 =================

print('Now train with {:s} with {:d} training samples, evaluate with {:d} samples'.format(args.dataset, len(train_loader.dataset), len(val_loader.dataset)))

# in the pretraining stage
if 'pretrain' in args.task:
    cluster = (args.num_mel_bins != args.fshape)
    if cluster == True:
        print('The num_mel_bins {:d} and fshape {:d} are different, not masking a typical time frame, using cluster masking.'.format(args.num_mel_bins, args.fshape))
    else:
        print('The num_mel_bins {:d} and fshape {:d} are same, masking a typical time frame, not using cluster masking.'.format(args.num_mel_bins, args.fshape))
    # no label dimension needed as it is self-supervised, fshape=fstride and tshape=tstride
    audio_model = ASTModel(fshape=args.fshape, tshape=args.tshape, fstride=args.fstride, tstride=args.tstride,
                       input_fdim=args.num_mel_bins, input_tdim=args.target_length, model_size=args.model_size, pretrain_stage=True)
# in the fine-tuning stage
else:
    audio_model = ASTModel(label_dim=args.n_class, fshape=args.fshape, tshape=args.tshape, fstride=args.fstride, tstride=args.tstride,
                       input_fdim=args.num_mel_bins, input_tdim=args.target_length, model_size=args.model_size, pretrain_stage=False,
                       load_pretrained_mdl_path=args.pretrained_mdl_path)

if not isinstance(audio_model, torch.nn.DataParallel):
    audio_model = torch.nn.DataParallel(audio_model)

print("\nCreating experiment directory: %s" % args.exp_dir)
if os.path.exists("%s/models" % args.exp_dir) == False:
    os.makedirs("%s/models" % args.exp_dir)
with open("%s/args.pkl" % args.exp_dir, "wb") as f:
    pickle.dump(args, f)

if 'pretrain' not in args.task:
    print('Now starting fine-tuning for {:d} epochs'.format(args.n_epochs))
    train(audio_model, train_loader, val_loader, args)
else:
    print('Now starting self-supervised pretraining for {:d} epochs'.format(args.n_epochs))

    # Add CUDA error handling and fallback to CPU if needed
    try:
        trainmask(audio_model, train_loader, val_loader, args)
    except RuntimeError as e:
        if "CUDA error" in str(e) or "no kernel image" in str(e).lower():
            print(f"CUDA error encountered: {e}")
            print("Attempting to continue with CPU...")

            # Recreate model on CPU
            if 'pretrain' in args.task:
                cluster = (args.num_mel_bins != args.fshape)
                if cluster == True:
                    print('The num_mel_bins {:d} and fshape {:d} are different, not masking a typical time frame, using cluster masking.'.format(args.num_mel_bins, args.fshape))
                else:
                    print('The num_mel_bins {:d} and fshape {:d} are same, masking a typical time frame, not using cluster masking.'.format(args.num_mel_bins, args.fshape))
                # no label dimension needed as it is self-supervised, fshape=fstride and tshape=tstride
                audio_model_cpu = ASTModel(fshape=args.fshape, tshape=args.tshape, fstride=args.fstride, tstride=args.tstride,
                                   input_fdim=args.num_mel_bins, input_tdim=args.target_length, model_size=args.model_size, pretrain_stage=True)
            # in the fine-tuning stage
            else:
                audio_model_cpu = ASTModel(label_dim=args.n_class, fshape=args.fshape, tshape=args.tshape, fstride=args.fstride, tstride=args.tstride,
                                   input_fdim=args.num_mel_bins, input_tdim=args.target_length, model_size=args.model_size, pretrain_stage=False,
                                   load_pretrained_mdl_path=args.pretrained_mdl_path)

            # Move to CPU and wrap with DataParallel
            audio_model_cpu = audio_model_cpu.to(torch.device("cpu"))
            if not isinstance(audio_model_cpu, torch.nn.DataParallel):
                audio_model_cpu = torch.nn.DataParallel(audio_model_cpu)

            print("Using CPU for training")
            trainmask(audio_model_cpu, train_loader, val_loader, args)
        else:
            # Re-raise if it's a different kind of error
            raise e

# if the dataset has a seperate evaluation set (e.g., speechcommands), then select the model using the validation set and eval on the evaluation set.
# this is only for fine-tuning
if args.data_eval != None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sd = torch.load(args.exp_dir + '/models/best_audio_model.pth', map_location=device)
    if not isinstance(audio_model, torch.nn.DataParallel):
        audio_model = torch.nn.DataParallel(audio_model)
    audio_model.load_state_dict(sd, strict=False)

    # best models on the validation set
    args.loss_fn = torch.nn.BCEWithLogitsLoss()
    stats, _ = validate(audio_model, val_loader, args, 'valid_set')
    # note it is NOT mean of class-wise accuracy
    val_acc = stats[0]['acc']
    val_mAUC = np.mean([stat['auc'] for stat in stats])
    print('---------------evaluate on the validation set---------------')
    print("Accuracy: {:.6f}".format(val_acc))
    print("AUC: {:.6f}".format(val_mAUC))

    # test the models on the evaluation set
    eval_loader = torch.utils.data.DataLoader(
        dataloader.AudioDataset(args.data_eval, label_csv=args.label_csv, audio_conf=val_audio_conf),
        batch_size=args.batch_size*2, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    stats, _ = validate(audio_model, eval_loader, args, 'eval_set')
    eval_acc = stats[0]['acc']
    eval_mAUC = np.mean([stat['auc'] for stat in stats])
    print('---------------evaluate on the test set---------------')
    print("Accuracy: {:.6f}".format(eval_acc))
    print("AUC: {:.6f}".format(eval_mAUC))
    np.savetxt(args.exp_dir + '/eval_result.csv', [val_acc, val_mAUC, eval_acc, eval_mAUC])

