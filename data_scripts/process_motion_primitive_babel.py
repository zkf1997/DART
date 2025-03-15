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


from config_files.data_paths import *
from utils.smpl_utils import convert_smpl_aa_to_rotmat


def calc_calibrate_offset(body_mesh_model, betas, transl, pose):
    '''
    The factors to influence this offset is not clear. Maybe it is shape and pose dependent.
    Therefore, we calculate such delta_T for each individual body mesh.
    It takes a batch of body parameters
    input:
        body_params: dict, basically the input to the smplx model
        smplx_model: the model to generate smplx mesh, given body_params
    Output:
        the offset for params transform
    '''
    n_batches = transl.shape[0]
    bodyconfig = {}
    bodyconfig['body_pose'] = torch.FloatTensor(pose[:,3:]).cuda()
    bodyconfig['betas'] = torch.FloatTensor(betas).unsqueeze(0).repeat(n_batches,1).cuda()
    bodyconfig['transl'] = torch.zeros([n_batches,3], dtype=torch.float32).cuda()
    bodyconfig['global_orient'] = torch.zeros([n_batches,3], dtype=torch.float32).cuda()
    bodyconfig = convert_smpl_aa_to_rotmat(bodyconfig)
    smplx_out = body_mesh_model(return_verts=True, **bodyconfig)
    delta_T = smplx_out.joints[:,0,:] # we output all pelvis locations
    delta_T = delta_T.detach().cpu().numpy() #[t, 3]

    return delta_T




def get_new_coordinate(body_mesh_model, betas, transl, pose):
    '''
    this function produces transform from body local coordinate to the world coordinate.
    it takes only a single frame.
    local coodinate:
        - located at the pelvis
        - x axis: from left hip to the right hip
        - z axis: point up (negative gravity direction)
        - y axis: pointing forward, following right-hand rule
    '''
    bodyconfig = {}
    bodyconfig['transl'] = torch.FloatTensor(transl).cuda()
    bodyconfig['global_orient'] = torch.FloatTensor(pose[:,:3]).cuda()
    bodyconfig['body_pose'] = torch.FloatTensor(pose[:,3:]).cuda()
    bodyconfig['betas'] = torch.FloatTensor(betas).unsqueeze(0).cuda()
    bodyconfig = convert_smpl_aa_to_rotmat(bodyconfig)
    smplxout = body_mesh_model(**bodyconfig)
    joints = smplxout.joints.squeeze().detach().cpu().numpy()
    x_axis = joints[2,:] - joints[1,:]
    x_axis[-1] = 0
    x_axis = x_axis / np.linalg.norm(x_axis)
    z_axis = np.array([0,0,1])
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis/np.linalg.norm(y_axis)
    global_ori_new = np.stack([x_axis, y_axis, z_axis], axis=1)
    transl_new = joints[:1,:] # put the local origin to pelvis

    return global_ori_new, transl_new




def get_body_model(type, gender, device='cpu'):
    '''
    type: smpl, smplx smplh and others. Refer to smplx tutorial
    gender: male, female, neutral
    batch_size: an positive integar
    '''
    body_model = smplx.build_layer(body_model_dir, model_type=type,
                                    gender=gender, ext='npz',
                                    num_pca_comps=12).to(device).eval()
    return body_model

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
bm_male = get_body_model('smplx', 'male', 'cuda')
bm_female = get_body_model('smplx', 'female', 'cuda')

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
                bodymodel = bm_male if str(data['gender'].astype(str)) == 'male' else bm_female

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
                    future_start, future_end = t + cfg.history_length, t + cfg.history_length + cfg.future_length
                    future_start = future_start / target_fps
                    future_end = future_end / target_fps
                    texts = []
                    for seg in frame_labels:
                        if have_overlap([seg['start_t'], seg['end_t']], [future_start, future_end]):
                            texts.append(seg['proc_label'])
                    data_out['texts'] = texts
                    ## perform transformation from the world coordinate to the amass coordinate
                    ### get transformation from amass space to world space
                    transf_rotmat, transf_transl = get_new_coordinate(bodymodel, betas[:10], transl[:1, :],
                                                                      pose[:1, :66])
                    ### calibrate offset
                    delta_T = calc_calibrate_offset(bodymodel, betas[:10], transl, pose[:, :66])
                    # print('error', np.max(np.abs(delta_T[0] - delta_T[1])))
                    # print('error', np.max(np.abs(delta_T[0] - delta_T[2])))
                    # print('error', np.max(np.abs(delta_T[4] - delta_T[5])))
                    ### get new global_orient
                    global_ori = R.from_rotvec(pose[:, :3]).as_matrix()  # to [t,3,3] rotation mat
                    global_ori_new = np.einsum('ij,tjk->tik', transf_rotmat.T, global_ori)
                    pose[:, :3] = R.from_matrix(global_ori_new).as_rotvec()
                    ### get new transl
                    transl = np.einsum('ij,tj->ti', transf_rotmat.T, transl + delta_T - transf_transl) - delta_T
                    data_out['transf_rotmat'] = transf_rotmat
                    data_out['transf_transl'] = transf_transl
                    data_out['transl'] = transl[:-1, :]
                    data_out['poses'] = pose[:-1, :66]
                    data_out['betas'] = betas  # [10,]
                    data_out['gender'] = data['gender'].astype(str)
                    data_out['mocap_framerate'] = target_fps

                    ## under this new amass coordinate, extract the joints/markers' locations
                    ## when get generated joints/markers, one can directly transform them back to world coord
                    ## note that hand pose is not considered here. In amass, the hand pose is regularized.
                    body_param = {}
                    body_param['transl'] = torch.FloatTensor(transl).cuda()
                    body_param['global_orient'] = torch.FloatTensor(pose[:, :3]).cuda()
                    body_param['betas'] = torch.FloatTensor(betas[:10]).unsqueeze(0).repeat(len_subseq + 1,
                                                                                            1).cuda()
                    body_param['body_pose'] = torch.FloatTensor(pose[:, 3:66]).cuda()
                    body_param = convert_smpl_aa_to_rotmat(body_param)
                    smplxout = bodymodel(return_verts=True, **body_param)
                    ### extract joints and markers
                    joints = smplxout.joints[:, :22, :].detach().squeeze().cpu().numpy()
                    data_out['joints'] = joints[:-1, :, :]
                    data_out['joints_delta'] = joints[1:, :, :] - joints[:-1, :, :]
                    data_out['transl_delta'] = transl[1:, :] - transl[:-1, :]
                    data_out['global_orient_delta'] = (R.from_rotvec(pose[1:, :3]).inv() * R.from_rotvec(pose[:-1, :3])).as_rotvec()  # [T, 3], rotation from t-1 to t

                    dataset.append(data_out)
                    t = t + cfg.future_length

                # break
        with open(out_path, 'wb') as f:
            pickle.dump(dataset, f)
    print('cannot find these sequences:', nonexist_paths)

if __name__=='__main__':
    main()
