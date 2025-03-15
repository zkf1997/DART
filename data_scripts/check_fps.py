import numpy
from pathlib import Path
import pickle
import os
import numpy as np
import json
from os.path import join as ospj
from tqdm import tqdm
import time
import smplx
import torch
import pickle
import trimesh
import pyrender
from pytorch3d import transforms

# load babel data
# raw_dataset_path = '/home/kaizhao/dataset/amass/smplh_g/'
# d_folder = '/home/kaizhao/dataset/amass/babel-teach/'
# splits = ['train', 'val']
# babel = {}
# fps_dict = {}
# fps_set = set()
# for spl in splits:
#     babel[spl] = json.load(open(ospj(d_folder, spl + '.json')))
#     for sid in tqdm(babel[spl]):
#         # seq_info = {}
#         feat_p = babel[spl][sid]['feat_p']
#         file_path = os.path.join(*(feat_p.split(os.path.sep)[1:]))
#         # dataset_name = file_path.split(os.path.sep)[0]
#         # file_path = file_path.replace('poses.npz',
#         #                               'stageii.npz')  # file naming suffix changed in different amass versions
#         # replace space
#         # file_path = file_path.replace(" ",
#         #                               "_")  # set replace count to string length, so all will be replaced
#         seq_path = os.path.join(raw_dataset_path, file_path)
#         if not os.path.exists(seq_path):
#             print(f"Missing: feat_p:{feat_p} seq_path:{seq_path}")
#             continue
#         seq_data = dict(np.load(seq_path, allow_pickle=True))
#         fps = seq_data['mocap_framerate'].item()
#         if fps not in fps_set:
#             fps_set.add(fps)
#         # fps_set{100.0, 120.00004577636719, 150.0, 120.0, 250.0, 59.9999885559082, 60.0}
#         if abs(fps - 120.0) < 1e-3:
#             fps = 120.0
#         if abs(fps - 60.0) < 1e-3:
#             fps = 60.0
#         # manually correct mislabeled data  https://github.com/athn-nik/teach/blob/c9701ed4d9403cfedc7db558f2dc508142279d2f/scripts/process_amass.py#L114C5-L114C30
#         if seq_path.find('BMLhandball') >= 0:
#             fps = 240
#         if seq_path.find('20160930_50032') >= 0 or seq_path.find('20161014_50033') >= 0:
#             fps = 60
#         duration = babel[spl][sid]['dur']
#         n_frames = len(seq_data['poses'])
#         if abs(n_frames / float(fps) - duration) > 0.2:
#             print(f"fps mismatch: feat_p:{feat_p} fps:{fps} duration:{duration} n_frames:{n_frames}")
#         fps_dict[feat_p] = fps
#
# print(f'fps_set{fps_set}')
# # save fps_dict
# with open('./data/fps_dict.json', 'w') as f:
#     json.dump(fps_dict, f, indent=4)


raw_dataset_path = Path('/home/kaizhao/dataset/amass/smplh_g/')
fps_dict = {}
fps_set = set()
for seq_path in raw_dataset_path.glob('./*/*/*poses.npz'):
    # print(f'process: {seq_path}')
    seq_data = dict(np.load(seq_path, allow_pickle=True))
    feat_p = seq_path.relative_to(raw_dataset_path).as_posix()
    # print(f'feat_p: {feat_p}')
    fps = seq_data['mocap_framerate'].item()
    if fps not in fps_set:
        fps_set.add(fps)
    # fps_set{100.0, 120.00004577636719, 150.0, 120.0, 250.0, 59.9999885559082, 60.0}
    if abs(fps - 120.0) < 1e-3:
        fps = 120.0
    if abs(fps - 60.0) < 1e-3:
        fps = 60.0
    # manually correct mislabeled data  https://github.com/athn-nik/teach/blob/c9701ed4d9403cfedc7db558f2dc508142279d2f/scripts/process_amass.py#L114C5-L114C30
    if str(seq_path).find('BMLhandball') >= 0:
        fps = 240
    if str(seq_path).find('20160930_50032') >= 0 or str(seq_path).find('20161014_50033') >= 0:
        fps = 60
    fps_dict[feat_p] = fps

print(f'fps_set{fps_set}')
# save fps_dict
with open('./data/fps_dict_all.json', 'w') as f:
    json.dump(fps_dict, f, indent=4)