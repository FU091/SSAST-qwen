# -*- coding: utf-8 -*-
# @Time    : 6/19/21 12:23 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : dataloader_pt.py

# Modified version to support precomputed .pt files
import csv
import json
import numpy as np
import torch
import torch.nn.functional
from torch.utils.data import Dataset
import random
import os
import re

def make_index_dict(label_csv):
    index_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            index_lookup[row['mid']] = row['index']
            line_count += 1
    return index_lookup

def make_name_dict(label_csv):
    name_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            name_lookup[row['index']] = row['display_name']
            line_count += 1
    return name_lookup

def lookup_list(index_list, label_csv):
    label_list = []
    table = make_name_dict(label_csv)
    for item in index_list:
        label_list.append(table[item])
    return label_list

class AudioDataset(Dataset):
    def __init__(self, dataset_json_file, audio_conf, label_csv=None):
        """
        Dataset that loads precomputed .pt files containing fbank and labels
        :param dataset_json_file: Path to json file with audio paths and labels (used to map to .pt files)
        :param audio_conf: Dictionary containing the audio loading and preprocessing settings
        :param label_csv: Path to csv file with class labels
        """
        self.datapath = dataset_json_file
        self.audio_conf = audio_conf
        self.data_dir = None  # Will be set based on the directory in audio_conf
        
        # Extract data directory from audio_conf
        if 'data_dir' in audio_conf:
            self.data_dir = audio_conf['data_dir']
        elif 'data_path' in audio_conf:
            self.data_dir = audio_conf['data_path']
        else:
            # Infer from dataset_json_file path - assuming .pt files are in a directory 
            # related to the json file or based on mode
            dataset_dir = os.path.dirname(dataset_json_file)
            self.data_dir = dataset_dir
            print(f"Warning: data_dir not specified in audio_conf, inferring from JSON path: {self.data_dir}")
        
        print('---------------the {:s} dataloader---------------'.format(self.audio_conf.get('mode')))
        
        # Load the JSON file to get the mapping of audio to labels
        with open(dataset_json_file, 'r') as fp:
            data_json = json.load(fp)
        self.data = data_json['data']
        
        # Verify .pt files exist and create mapping
        self.valid_indices = []
        self.pt_files = []
        
        for i, item in enumerate(self.data):
            original_path = item['wav']
            # Extract filename properly by splitting on both \ and /
            path_parts = re.split(r'[\\/]+', original_path)
            filename = path_parts[-1]  # Get last part which should be the actual filename
            filename_no_ext = os.path.splitext(filename)[0]
            pt_path = os.path.join(self.data_dir, f"{filename_no_ext}.pt")
            
            if os.path.exists(pt_path):
                self.valid_indices.append(i)
                self.pt_files.append(pt_path)
            else:
                print(f"Warning: .pt file not found for {original_path} -> {pt_path}")
        
        print(f"Out of {len(self.data)} JSON entries, {len(self.valid_indices)} .pt files exist.")
        
        if len(self.valid_indices) == 0:
            raise ValueError("No .pt files found matching the JSON entries!")

        # Set up audio config parameters from audio_conf
        self.melbins = self.audio_conf.get('num_mel_bins', 128)
        self.target_length = self.audio_conf.get('target_length', 1024)
        self.freqm = self.audio_conf.get('freqm', 0)
        self.timem = self.audio_conf.get('timem', 0)
        print('now using following mask: {:d} freq, {:d} time'.format(self.freqm, self.timem))
        self.mixup = self.audio_conf.get('mixup', 0.0)
        print('now using mix-up with rate {:f}'.format(self.mixup))
        self.dataset = self.audio_conf.get('dataset', 'audioset')
        print('now process ' + self.dataset)
        
        # dataset spectrogram mean and std, used to normalize the input
        self.norm_mean = self.audio_conf.get('mean')
        self.norm_std = self.audio_conf.get('std')
        # skip_norm is a flag that if you want to skip normalization to compute the normalization stats
        self.skip_norm = self.audio_conf.get('skip_norm', False)
        if self.skip_norm:
            print('now skip normalization (use it ONLY when you are computing the normalization stats).')
        else:
            print('use dataset mean {:.3f} and std {:.3f} to normalize the input.'.format(self.norm_mean, self.norm_std))
        
        # if add noise for data augmentation
        self.noise = self.audio_conf.get('noise', False)
        if self.noise == True:
            print('now use noise augmentation')

        # Create label index dictionary if label_csv is provided
        if label_csv is not None:
            self.index_dict = make_index_dict(label_csv)
            self.label_num = len(self.index_dict)
            print('number of classes is {:d}'.format(self.label_num))
        else:
            # For pretraining, we might not need actual labels, but we still need this for shape
            # We'll infer from the first sample
            self.index_dict = {}
            self.label_num = 0  # Will be inferred from precomputed file if needed

    def __getitem__(self, index):
        """
        returns: fbank, label
        where fbank is a FloatTensor of size (time_frame_num, frequency_bins) for spectrogram, e.g., (1024, 128)
        label is a FloatTensor of size (num_classes) or scalar for dummy labels in pretraining
        """
        # Get the actual index in the valid dataset
        actual_index = self.valid_indices[index]
        pt_path = self.pt_files[index]
        
        try:
            # Load precomputed fbank and label from .pt file
            data_dict = torch.load(pt_path)
            fbank = data_dict['x']  # This should already be normalized based on your preprocessing
            label = data_dict['y']
            
            # Convert to float32 if needed (in case they were saved as float16 for space saving)
            if fbank.dtype == torch.float16:
                fbank = fbank.to(torch.float32)
            if label.dtype == torch.float16:
                label = label.to(torch.float32)
            
            # Apply time/frequency masking if in training mode and enabled
            if self.audio_conf.get('mode') == 'train' and (self.freqm > 0 or self.timem > 0):
                # Apply SpecAugment-like masking
                fbank = self.apply_masking(fbank)
            
            # Apply noise augmentation if enabled
            if self.noise == True:
                fbank = fbank + torch.rand(fbank.shape[0], fbank.shape[1]) * np.random.rand() / 10
                fbank = torch.roll(fbank, np.random.randint(-10, 10), 0)
                
        except Exception as e:
            print(f"Error loading {pt_path}: {e}")
            # Return a default tensor in case of error
            fbank = torch.zeros(self.target_length, self.melbins, dtype=torch.float32)
            label = torch.zeros(self.label_num, dtype=torch.float32) if hasattr(self, 'label_num') and self.label_num > 0 else torch.zeros(1, dtype=torch.float32)
        
        return fbank, label

    def apply_masking(self, fbank):
        """
        Apply frequency and time masking to fbank (similar to the original dataloader)
        """
        # Transpose for masking operations
        fbank = torch.transpose(fbank, 0, 1)
        fbank = fbank.unsqueeze(0)  # Add batch dimension for transforms
        
        # Apply frequency masking
        if self.freqm > 0:
            # For frequency masking, we create a random frequency bin to mask
            freq_len = fbank.shape[2]
            if self.freqm < freq_len:
                freq_start = torch.randint(0, freq_len - self.freqm, (1,))
                fbank[0, :, freq_start:freq_start + self.freqm] = 0
        
        # Apply time masking
        if self.timem > 0:
            # For time masking, we create a random time frame to mask
            time_len = fbank.shape[3]
            if self.timem < time_len:
                time_start = torch.randint(0, time_len - self.timem, (1,))
                fbank[0, :, :, time_start:time_start + self.timem] = 0
        
        # Remove batch dimension and transpose back
        fbank = fbank.squeeze(0)
        fbank = torch.transpose(fbank, 0, 1)
        
        return fbank

    def __len__(self):
        return len(self.valid_indices)