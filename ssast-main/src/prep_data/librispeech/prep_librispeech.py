# -*- coding: utf-8 -*-
# @Time    : 7/11/21 6:55 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : prep_librispeech.py

# prepare librispeech data for ssl pretraining

import os,torchaudio,pickle,json,time

def walk(path, name):
    sample_cnt = 0
    pathdata = os.walk(path)
    wav_list = []
    begin_time = time.time()
    for root, dirs, files in pathdata:
        for file in files:
            if file.lower().endswith('.wav'):
                sample_cnt += 1

                cur_path = root + os.sep + file
                # give a dummy label of 'speech' ('/m/09x0r' in AudioSet label ontology) to all librispeech samples
                # the label is not used in the pretraining, it is just to make the dataloader.py satisfy.
                cur_dict = {"wav": cur_path, "labels": '/m/09x0r'}
                wav_list.append(cur_dict)

                if sample_cnt % 1000 == 0:
                    end_time = time.time()
                    print('find {:d}k .wav files, time eclipse: {:.1f} seconds.'.format(int(sample_cnt/1000), end_time-begin_time))
                    begin_time = end_time
                if sample_cnt % 1e4 == 0:
                    with open(name + '.json', 'w') as f:
                        json.dump({'data': wav_list}, f, indent=1)
                    print('file saved.')
    print(sample_cnt)
    with open(name + '.json', 'w') as f:
        json.dump({'data': wav_list}, f, indent=1)

# combine json files
def combine_json(file_list, name='librispeech_tr960'):
    wav_list = []
    for file in file_list:
        with open(file + '.json', 'r') as f:
            cur_json = json.load(f)
        wav_list = wav_list + cur_json['data']
    with open(name + '.json', 'w') as f:
        json.dump({'data': wav_list}, f, indent=1)

# 1. 處理 GW_audio
# 使用 r'...' (raw string) 避免 Windows 路徑中的反斜線 \ 被當成跳脫字元
path_gw = r"D:\Audio_Output"
walk(path_gw, '6s_audio_list')  # 產出 gw_audio_list.json

"""
# 2. 處理 FS_SOUND
path_fs = r'D:\FS_SOUND'
walk(path_fs, 'fs_sound_list')  # 產出 fs_sound_list.json

# 3. 合併這兩個清單
# 最終會產出一個名為 "combined_train_data.json" 的檔案供模型訓練使用
combine_json(['gw_audio_list', 'fs_sound_list'], name='combined_train_data')
"""