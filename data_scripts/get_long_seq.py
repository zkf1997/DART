import os
import random

import numpy as np
import json
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
raw_dataset_path = '/home/kaizhao/dataset/amass/smplx_g/'
target_fps = 30

babel = {}
with open('/home/kaizhao/projects/flowmdm/dataset/babel/babel-teach/val.json', 'r') as f:
    babel['val'] = json.load(f)
spl = 'val'

with open('./data/val_only_retrieval.json', 'r') as f:
    val_only_retrieval = json.load(f)

seq_cfg_list = []
for sid in tqdm(babel[spl].keys()):
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
    if not os.path.exists(seq_path):
        continue
    # if not 'frame_labels' in seq_info:
    #     continue

    if 'frame_ann' in babel[spl][sid] and babel[spl][sid]['frame_ann'] is not None:
        frame_labels = babel[spl][sid]['frame_ann']['labels']
    else:  # the sequence has only sequence label, which means the sequence has only one action
        seq_data = dict(np.load(seq_path, allow_pickle=True))
        fps = seq_data['mocap_frame_rate']
        downsample_rate = int(fps / target_fps)
        assert fps == 120.0
        motion_data = {}
        motion_data['trans'] = seq_data['trans'][::downsample_rate].astype(np.float32)
        motion_data['poses'] = seq_data['poses'][::downsample_rate, :66].astype(np.float32)
        motion_data['betas'] = seq_data['betas'][:10].astype(np.float32)
        motion_data['gender'] = str(seq_data['gender'].item())
        """move the code to remove short sequences to the dataset class"""
        # if len(motion_data['trans']) < self.seq_length:
        #     continue
        seq_data_dict = {'motion': motion_data, 'data_source': 'babel', 'seq_name': file_path, 'feat_p': feat_p}

        frame_labels = babel[spl][sid]['seq_ann']['labels']  # onle one element
        frame_labels[0]['start_t'] = 0
        frame_labels[0]['end_t'] = motion_data['trans'].shape[0] / target_fps

    # sort frame labels first by start time, then by end time
    time_points = []
    for seg in frame_labels:
        time_points.append(seg['start_t'])
        time_points.append(seg['end_t'])
    time_points = sorted(list(set(time_points)))
    # max_interval = 200 / target_fps
    # split_points = []
    # for idx in range(len(time_points) - 1):
    #     split_point = time_points[idx] + max_interval
    #     while split_point < time_points[idx + 1]:
    #         split_points.append(split_point)
    #         split_point += max_interval
    # time_points += split_points
    # time_points = sorted(list(set(time_points)))
    if time_points[-1] > 60:
        continue

    seq_cfg = {
        'id': sid,
        "scenario": "in-distribution",
        'text': [],
        'lengths': [],
    }
    # print(time_points)
    for idx in range(len(time_points) - 1):
        start_t = time_points[idx]
        end_t = time_points[idx + 1]
        num_frames = int((end_t - start_t) * target_fps)
        if num_frames < 0.5 * target_fps:
            continue
        texts = []
        for seg in frame_labels:
            if seg['proc_label'] == 'transition':  # ignore transition
                continue
            overlap_time = min(end_t, seg['end_t']) - max(start_t, seg['start_t'])
            if overlap_time > 1e-6:
                proc_label = seg['proc_label']
                # if proc_label in val_only_retrieval:  # replace with the retrieval label
                #     proc_label = val_only_retrieval[proc_label]
                texts.append(proc_label)
        if len(texts) == 0:
            continue
        # print(sid, start_t, end_t, texts)
        compo_text = ' and '.join(texts)
        # seq_cfg['text'].append(compo_text)
        seq_cfg['text'].append(texts[0])
        seq_cfg['lengths'].append(max(num_frames, 15))  # at least 15 frames, compatible with flowmdm
    if len(seq_cfg['text']) != 5:
        continue
    # if len(seq_cfg['text']) > 1:
    #     continue

    # print(seq_cfg)
    seq_cfg_list.append(seq_cfg)

    # break

random.shuffle(seq_cfg_list)
with open('./data/long_seq_5.json', 'w') as f:
    json.dump(seq_cfg_list[:16], f, indent=4)
