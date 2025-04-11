import numpy
from pathlib import Path
import pickle
import os
import numpy as np
import json
from os.path import join as ospj
from config_files.data_paths import *
from utils.misc_util import have_overlap
from utils.smpl_utils import *
from tqdm import tqdm
import time
import smplx
import torch
import pickle
import trimesh
import pyrender
from pytorch3d import transforms
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


target_fps = 30
dataset = {
    'train': [],
    'val': [],
}


model_path = body_model_dir
gender = "male"
body_model = smplx.create(model_path=model_path,
                             model_type='smplx',
                             gender=gender,
                             use_pca=False,
                             batch_size=1)
device = 'cuda'
primitive_utility = PrimitiveUtility(device=device)

def calc_joints_pelvis_delta(motion_data):
    pelvis_delta = primitive_utility.calc_calibrate_offset({
        'betas': torch.tensor(motion_data['betas'], device=device).reshape(1, 10),
        'gender': motion_data['gender'],
    })  # [1, 3]
    pelvis_delta = pelvis_delta.detach().cpu().numpy().squeeze()  # [3]
    num_frames = len(motion_data['trans'])
    poses = torch.tensor(motion_data['poses'], device=device)
    global_orient = transforms.axis_angle_to_matrix(poses[:, :3])  # [num_frames, 3, 3]
    body_pose = transforms.axis_angle_to_matrix(poses[:, 3:66].reshape(num_frames, 21, 3))  # [num_frames, 21, 3, 3]
    joints = primitive_utility.smpl_dict_inference(
        {
            'gender': motion_data['gender'],
            'betas': torch.tensor(motion_data['betas'], device=device).reshape(1, 10).repeat(num_frames, 1),
            'transl': torch.tensor(motion_data['trans'], device=device).reshape(num_frames, 3),
            'global_orient': global_orient,
            'body_pose': body_pose,
        }, return_vertices=False
    )  # [num_frames, 22, 3]
    joints = joints.detach().cpu().numpy()  # [num_frames, 22, 3]

    return joints, pelvis_delta

def downsample(fps, target_fps, seq_data):
    old_trans = seq_data['trans']
    old_poses = seq_data['poses'][:, :66].reshape((-1, 22, 3))
    old_num_frames = len(seq_data['trans'])
    new_num_frames = int((old_num_frames - 1) / fps * target_fps) + 1
    if new_num_frames < 2:
        return None, None
    old_time = np.array(range(old_num_frames)) / fps
    new_time = np.array(range(new_num_frames)) / target_fps
    trans = np.zeros((new_num_frames, 3))
    poses = np.zeros((new_num_frames, 22, 3))
    for i in range(3):  # linear interpolation for translation
        trans[:, i] = np.interp(x=new_time, xp=old_time, fp=old_trans[:, i])
    for joint_idx in range(22):
        slerp = Slerp(times=old_time, rotations=R.from_rotvec(old_poses[:, joint_idx, :]))
        poses[:, joint_idx, :] = slerp(new_time).as_rotvec()
    return trans, poses.reshape((-1, 66))

def add_babel(enforce_zero_male=False, process_transition=False):
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
    with open('./data/fps_dict.json', 'r') as f:
        fps_dict = json.load(f)

    # load babel data
    raw_dataset_path = amass_dir / 'smplx_g/'
    d_folder = babel_dir
    splits = ['train', 'val']
    babel = {}
    for spl in splits:
        babel[spl] = json.load(open(ospj(d_folder, spl + '.json')))
        for sid in tqdm(babel[spl]):
            # seq_info = {}
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

            seq_data = dict(np.load(seq_path, allow_pickle=True))
            """SMPLX-G fps label is not accurate, use the fps from the fps_dict"""
            # fps = seq_data['mocap_frame_rate']
            # assert fps == 120.0
            fps = fps_dict[feat_p]
            downsample_rate = int(fps / target_fps)
            motion_data = {}
            motion_data['betas'] = seq_data['betas'][:10].astype(np.float32)
            motion_data['gender'] = str(seq_data['gender'].item())
            if enforce_zero_male:
                motion_data['betas'] = np.zeros_like(motion_data['betas'])
                motion_data['gender'] = 'male'
            if downsample_rate * target_fps != fps:
                # https://github.com/athn-nik/teach/blob/c9701ed4d9403cfedc7db558f2dc508142279d2f/scripts/process_amass.py#L129
                # new_num_frames = int(target_fps / fps * len(seq_data['trans']))
                # downsample_idx = np.linspace(0, len(seq_data['trans']) - 1,
                #                              num=new_num_frames, dtype=int)
                # if len(downsample_idx) < 1:
                #     print(f"downsample_idx is empty: feat_p {feat_p} fps{fps} target_fps {target_fps}")
                #     continue
                # motion_data['trans'] = seq_data['trans'][downsample_idx].astype(np.float32)
                # motion_data['poses'] = seq_data['poses'][downsample_idx, :66].astype(np.float32)

                trans, poses = downsample(fps, target_fps, seq_data)
                if trans is None:
                    print(f'sequence too short: {feat_p}')
                    continue
                motion_data['trans'], motion_data['poses'] = trans.astype(np.float32), poses.astype(np.float32)
            else:
                motion_data['trans'] = seq_data['trans'][::downsample_rate].astype(np.float32)
                motion_data['poses'] = seq_data['poses'][::downsample_rate, :66].astype(np.float32)
            joints, pelvis_delta = calc_joints_pelvis_delta(motion_data)
            motion_data['joints'] = joints
            motion_data['pelvis_delta'] = pelvis_delta

            """move the code to remove short sequences to the dataset class"""
            # if len(motion_data['trans']) < self.seq_length:
            #     continue
            seq_data_dict = {'motion': motion_data, 'data_source': 'babel', 'seq_name': file_path, 'feat_p': feat_p}

            if 'frame_ann' in babel[spl][sid] and babel[spl][sid]['frame_ann'] is not None:
                frame_labels = babel[spl][sid]['frame_ann']['labels']
                # process the transition labels, concatenate it with the target action
                for seg in frame_labels:
                    if process_transition:
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
                seq_data_dict['frame_labels'] = frame_labels
            else:  # the sequence has only sequence label, which means the sequence has only one action
                frame_labels = babel[spl][sid]['seq_ann']['labels']  # onle one element
                frame_labels[0]['start_t'] = 0
                frame_labels[0]['end_t'] = motion_data['trans'].shape[0] / target_fps
                seq_data_dict['frame_labels'] = frame_labels
            if seq_path.find('20160930_50032') >= 0 or seq_path.find('20161014_50033') >= 0:
                print('correcting frame label:', seq_path)
                for seg in seq_data_dict['frame_labels']:
                    seg['start_t'] = seg['start_t'] * 120 / 60
                    seg['end_t'] = seg['end_t'] * 120 / 60


            dataset[spl].append(seq_data_dict)


enforce_zero_male = True
add_babel(enforce_zero_male=enforce_zero_male, process_transition=False)

output_path = f'./data/seq_data'
if enforce_zero_male:
    output_path = f'{output_path}_zero_male'
Path(output_path).mkdir(exist_ok=True, parents=True)
with open(ospj(output_path, 'train.pkl'), 'wb') as f:
    pickle.dump(dataset['train'], f)
with open(ospj(output_path, 'val.pkl'), 'wb') as f:
    pickle.dump(dataset['val'], f)

