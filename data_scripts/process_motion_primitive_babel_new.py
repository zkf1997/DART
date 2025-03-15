from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys, glob
import numpy as np
from tqdm import tqdm
import torch
import smplx
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import gaussian_filter1d
import json
import csv
import pdb
import pickle
from copy import deepcopy

import sys, os, pdb
from os.path import join as ospj
import json
from collections import *
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.core.common import flatten
from omegaconf import DictConfig, OmegaConf
import hydra
from pytorch3d import transforms

from config_files.data_paths import *
from utils.smpl_utils import *


def have_overlap(seg1, seg2):
    if seg1[0] > seg2[1] or seg2[0] > seg1[1]:
        return False
    else:
        return True

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

device = 'cuda'
dtype = torch.float32
torch.set_default_device(device)
torch.set_default_dtype(dtype)
primitive_utility = PrimitiveUtility(device=device, dtype=dtype)

base_folder = amass_dir
# load babel labels
d_folder = babel_dir
babel_set = l_babel_dense_files = ['train', 'val']
# BABEL Dataset
babel = {}
for file in l_babel_dense_files:
    babel[file] = json.load(open(ospj(d_folder, file + '.json')))

@hydra.main(version_base=None, config_path="../config_files/config_hydra/motion_primitive", config_name="mp_2_8")
def main(omega_cfg: DictConfig):
    cfg = omega_cfg
    print(OmegaConf.to_yaml(cfg))

    # canonicalize
    N_MPS = 1
    MP_FRAME = cfg.history_length + cfg.future_length
    target_fps = cfg.fps
    downsample_rate = 120 // target_fps
    len_subseq = int(MP_FRAME * N_MPS)
    #### set input output dataset paths
    raw_dataset_path = base_folder / 'smplx_g'
    result_smplx_path = dataset_root_dir / 'mp_data' / f'Canonicalized_h{cfg.history_length}_f{cfg.future_length}_num{N_MPS}_fps{target_fps}'
    result_smplx_path.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, Path(result_smplx_path, "config.yaml"))

    nonexist_paths = []
    for spl in l_babel_dense_files:
        out_path = result_smplx_path / f'{spl}.pkl'
        dataset = []
        for sid in tqdm(babel[spl]):
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
                                if have_overlap([seg['start_t'], seg['end_t']], [seg2['start_t'], seg2['end_t']]) and seg2['end_t'] > seg['end_t']:
                                    seg['proc_label'] = 'transition to ' + seg2['proc_label']
                                    seg['act_cat'] = seg['act_cat'] + seg2['act_cat']
                                    break
                            if seg['proc_label'] == 'transition':
                                print('the transition target action not found:')
                                seg['proc_label'] = 'transition to another action'
                                print(sid, seg)


                # read data
                file_path = os.path.join(*(babel[spl][sid]['feat_p'].split(os.path.sep)[1:]))
                dataset_name = file_path.split(os.path.sep)[0]
                if dataset_name in amass_dataset_rename_dict:
                    file_path = file_path.replace(dataset_name, amass_dataset_rename_dict[dataset_name])
                file_path = file_path.replace('poses.npz',
                                              'stageii.npz')  # file naming suffix changed in different amass versions
                # replace space
                file_path = file_path.replace(" ",
                                              "_")  # set replace count to string length, so all will be replaced
                seq = os.path.join(raw_dataset_path, file_path)
                if not os.path.exists(seq):
                    nonexist_paths.append(seq)
                    continue
                print('loading:', seq)
                data = dict(np.load(seq, allow_pickle=True))
                # print(data.keys())
                if not 'mocap_frame_rate' in data:
                    continue
                fps = data['mocap_frame_rate']
                assert fps == 120.0

                ## read data and downsample
                transl_all = data['trans'][::downsample_rate]
                pose_all = data['poses'][::downsample_rate]
                betas = data['betas'][:10]

                ## skip too short sequences
                n_frames = transl_all.shape[0]
                if n_frames < len_subseq + 1:
                    continue

                t = 0
                while t < n_frames:
                    transl = deepcopy(transl_all[t:t + len_subseq + 1, :])  # plus 1 for velocity calculation
                    pose = deepcopy(pose_all[t:t + len_subseq + 1, :])
                    ## break if remaining frames are not sufficient
                    if transl.shape[0] < len_subseq + 1:
                        break

                    data_out = {}
                    data_out['betas'] = betas  # [10,]
                    data_out['gender'] = str(data['gender'].item())
                    data_out['mocap_framerate'] = target_fps
                    future_start, future_end = t + cfg.history_length, t + cfg.history_length + cfg.future_length - 1
                    future_start = future_start / target_fps
                    future_end = future_end / target_fps
                    texts = []
                    for seg in frame_labels:
                        if have_overlap([seg['start_t'], seg['end_t']], [future_start, future_end]):
                            texts.append(seg['proc_label'])
                    data_out['texts'] = texts

                    primitive_dict = {
                        'gender': str(data['gender'].item()),
                        'betas': torch.tensor(betas).expand(1, len_subseq + 1, 10).to(device=device, dtype=dtype),
                        'transl': torch.tensor(transl).unsqueeze(0).to(device=device, dtype=dtype),
                        'global_orient': transforms.axis_angle_to_matrix(torch.tensor(pose[:, :3]).unsqueeze(0)).to(device=device, dtype=dtype),
                        'body_pose': transforms.axis_angle_to_matrix(torch.tensor(pose[:, 3:66]).unsqueeze(0).reshape(1, len_subseq + 1, 21, 3)).to(device=device, dtype=dtype),
                        'transf_rotmat': torch.eye(3).unsqueeze(0),
                        'transf_transl': torch.zeros(1, 1, 3),
                    }
                    _, _, canonicalized_primitive_dict = primitive_utility.canonicalize(primitive_dict)
                    transf_rotmat, transf_transl = canonicalized_primitive_dict['transf_rotmat'], canonicalized_primitive_dict['transf_transl']
                    feature_dict = primitive_utility.calc_features(canonicalized_primitive_dict)
                    # ## perform transformation from the world coordinate to the amass coordinate
                    # ### get transformation from amass space to world space
                    # transf_rotmat, transf_transl = get_new_coordinate(bodymodel, betas[:10], transl[:1, :],
                    #                                                   pose[:1, :66])
                    # ### calibrate offset
                    # delta_T = calc_calibrate_offset(bodymodel, betas[:10], transl, pose[:, :66])
                    # # print('error', np.max(np.abs(delta_T[0] - delta_T[1])))
                    # # print('error', np.max(np.abs(delta_T[0] - delta_T[2])))
                    # # print('error', np.max(np.abs(delta_T[4] - delta_T[5])))
                    # ### get new global_orient
                    # global_ori = R.from_rotvec(pose[:, :3]).as_matrix()  # to [t,3,3] rotation mat
                    # global_ori_new = np.einsum('ij,tjk->tik', transf_rotmat.T, global_ori)
                    # pose[:, :3] = R.from_matrix(global_ori_new).as_rotvec()
                    # ### get new transl
                    # transl = np.einsum('ij,tj->ti', transf_rotmat.T, transl + delta_T - transf_transl) - delta_T
                    data_out['transf_rotmat'] = transf_rotmat
                    data_out['transf_transl'] = transf_transl
                    data_out['transl'] = feature_dict['transl'][0, :-1, :]  # [T, 3]
                    data_out['transl_delta'] = feature_dict['transl_delta'][0, :, :]  # [T, 3]
                    data_out['poses_6d'] = feature_dict['poses_6d'][0, :-1, :]  # [T, 66]
                    data_out['global_orient_delta_6d'] = feature_dict['global_orient_delta_6d'][0, :, :]  # [T, 3]
                    data_out['joints'] = feature_dict['joints'][0, :-1, :]  # [T, 22 * 3]
                    data_out['joints_delta'] = feature_dict['joints_delta'][0, :, :]  # [T, 22 * 3]
                    for key in data_out:
                        if torch.is_tensor(data_out[key]):
                            data_out[key] = data_out[key].cpu().numpy()

                    ## under this new amass coordinate, extract the joints/markers' locations
                    ## when get generated joints/markers, one can directly transform them back to world coord
                    ## note that hand pose is not considered here. In amass, the hand pose is regularized.
                    # body_param = {}
                    # body_param['transl'] = torch.FloatTensor(transl).cuda()
                    # body_param['global_orient'] = torch.FloatTensor(pose[:, :3]).cuda()
                    # body_param['betas'] = torch.FloatTensor(betas[:10]).unsqueeze(0).repeat(len_subseq + 1,
                    #                                                                         1).cuda()
                    # body_param['body_pose'] = torch.FloatTensor(pose[:, 3:66]).cuda()
                    # body_param = convert_smpl_aa_to_rotmat(body_param)
                    # smplxout = bodymodel(return_verts=True, **body_param)
                    # ### extract joints and markers
                    # joints = smplxout.joints[:, :22, :].detach().squeeze().cpu().numpy()
                    # data_out['joints'] = joints[:-1, :, :]
                    # data_out['joints_delta'] = joints[1:, :, :] - joints[:-1, :, :]
                    # data_out['transl_delta'] = transl[1:, :] - transl[:-1, :]
                    # data_out['global_orient_delta'] = (R.from_rotvec(pose[1:, :3]).inv() * R.from_rotvec(pose[:-1, :3])).as_rotvec()  # [T, 3], rotation from t-1 to t

                    dataset.append(data_out)
                    t = t + cfg.future_length

                # break
        with open(out_path, 'wb') as f:
            pickle.dump(dataset, f)
    print('cannot find these sequences:', nonexist_paths)

if __name__=='__main__':
    main()
