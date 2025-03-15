import numpy
from pathlib import Path
import pickle
import os
import numpy as np
import json
from os.path import join as ospj
from config_files.data_paths import *
from utils.misc_util import have_overlap
from tqdm import tqdm

# AMASS dataset names from website are slightly different from what used in BABEL
amass_dataset_rename_dict = {
    'ACCAD': 'ACCAD',
    'BMLmovi': 'BMLmovi',
    'BioMotionLab_NTroje': 'BMLrub',
    'MPI_HDM05': 'HDM05',
    'CMU': 'CMU',
    'Eyes_Japan_Dataset': 'EyesJapanDataset/Eyes_Japan_Dataset',
    'HumanEva': 'HumanEva',
    'TCD_handMocap': 'TCDHands',
    'KIT': 'KIT',
    'Transitions_mocap': 'Transitions',
    'DFaust_67': 'DFaust',
    'MPI_Limits': 'PosePrior',
    'SSM_synced': 'SSM',
    'MPI_mosh': 'MoSh',
}

raw_dataset_path = './data/amass/smplx_g/'
output_path = './data/mp_data/Canonicalized_h2_f8_num1_fps30/seq_info.json'

seq_info_dataset = {
    'train': [],
    'val': [],
}

# load babel labels
d_folder = babel_dir
splits = ['train', 'val']
babel = {}
for spl in splits:
    babel[spl] = json.load(open(ospj(d_folder, spl + '.json')))
    for sid in tqdm(babel[spl]):
        seq_info = {}
        feat_p = babel[spl][sid]['feat_p']
        file_path = os.path.join(*(feat_p.split(os.path.sep)[1:]))
        dataset_name = file_path.split(os.path.sep)[0]
        if dataset_name in amass_dataset_rename_dict:
            file_path = file_path.replace(dataset_name, amass_dataset_rename_dict[dataset_name])
        file_path = file_path.replace('poses.npz',
                                      'stageii.npz')  # file naming suffix changed in different amass versions
        # replace space
        file_path = file_path.replace(" ",
                                      "_")  # set replace count to string length, so all will be replaced
        seq_path = os.path.join(raw_dataset_path, file_path)
        seq_info['seq_path'] = seq_path

        if 'frame_ann' in babel[spl][sid] and babel[spl][sid]['frame_ann'] is not None:
            frame_labels = babel[spl][sid]['frame_ann']['labels']
            # process the transition labels, concatenate it with the target action
            for seg in frame_labels:
                if seg['proc_label'] == 'transition':
                    for seg2 in frame_labels:
                        if seg2['start_t'] == seg['end_t']:
                            seg['proc_label'] = 'transition to ' + seg2['proc_label']
                            seg['act_cat'] = seg['act_cat'] + seg2['act_cat']
                            break
                    if seg['proc_label'] == 'transition':
                        print('no consecutive transition found, try to find overlapping segments')
                        for seg2 in frame_labels:
                            if have_overlap([seg['start_t'], seg['end_t']], [seg2['start_t'], seg2['end_t']]) and seg2[
                                'end_t'] > seg['end_t']:
                                seg['proc_label'] = 'transition to ' + seg2['proc_label']
                                seg['act_cat'] = seg['act_cat'] + seg2['act_cat']
                                break
                        if seg['proc_label'] == 'transition':
                            print('the transition target action not found:')
                            seg['proc_label'] = 'transition to another action'
                            print(sid, seg)
            seq_info['frame_labels'] = frame_labels

        seq_info_dataset[spl].append(seq_info)

with open(output_path, 'w') as f:
    json.dump(seq_info_dataset, f)

