import pdb

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
import pandas as pd

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

def mirror_sequence(trans, poses):
    """ left-right mirror of translation and joint rotations"""
    trans_mirror = deepcopy(trans)
    trans_mirror[:, 0] = -trans_mirror[:, 0]
    right_chain = [2, 5, 8, 11, 14, 17, 19, 21]
    left_chain = [1, 4, 7, 10, 13, 16, 18, 20]

    # mirror joint rotations
    def mirror_orient(poses, right_chain, left_chain):
        poses_mirror = deepcopy(poses).reshape((-1, 3))
        poses_quaternion = R.from_rotvec(poses_mirror).as_quat()
        poses_quaternion *= np.array([1, 1, -1, 1], dtype=np.float32).reshape((1, 4))
        poses_mirror = R.from_quat(poses_quaternion).as_rotvec().astype(np.float32)
        poses_mirror = poses_mirror.reshape((-1, 22, 3))
        poses_mirror[right_chain], poses_mirror[left_chain] = poses_mirror[left_chain], poses_mirror[right_chain]
        poses_mirror = poses_mirror.reshape((-1, 66))
        return poses_mirror

    poses_mirror = mirror_orient(poses, right_chain, left_chain)
    poses_mirror = poses_mirror.reshape((-1, 66))
    # pdb.set_trace()

    return trans_mirror, poses_mirror

def export_smpl(transl, poses, gender, betas, output_path):
    poses = np.concatenate([poses, np.zeros((poses.shape[0], 99))], axis=1)
    data_dict = {
        'mocap_framerate': 30,
        'gender': gender,
        'betas': betas,
        'poses': poses,
        'trans': transl,
    }
    with open(output_path, 'wb') as f:
        np.savez(f, **data_dict)

model_path = body_model_dir
gender = "male"
device = 'cuda'
primitive_utility = PrimitiveUtility(device=device, body_type='smplh')

splits = {}
with open('./data/HumanML3D/HumanML3D/train.txt', 'r') as f:
    splits['train'] = f.readlines()
with open('./data/HumanML3D/HumanML3D/val.txt', 'r') as f:
    splits['val'] = f.readlines()
with open('./data/HumanML3D/HumanML3D/test.txt', 'r') as f:
    splits['test'] = f.readlines()

with open('./data/fps_dict_all.json', 'r') as f:
    fps_dict = json.load(f)
index_path = './data/HumanML3D/index.csv'
index_file = pd.read_csv(index_path)
total_amount = index_file.shape[0]

# load babel data
raw_dataset_path = amass_dir / 'smplh_g/'
text_dir = Path('./data/HumanML3D/HumanML3D/texts/')

output_path = f'./data/hml3d_smplh/seq_data'
output_path = f'{output_path}_zero_male'
Path(output_path).mkdir(exist_ok=True, parents=True)

enforce_zero_male=True
process_transition=False
target_fps = 20
dataset = {}

for i in tqdm(range(total_amount)):
    source_path = index_file.loc[i]['source_path']
    # print(f"Processing {source_path}")
    if 'humanact12' in source_path:
        continue
    new_name = index_file.loc[i]['new_name']
    new_name = new_name.split('.')[0]
    start_frame = index_file.loc[i]['start_frame']
    end_frame = index_file.loc[i]['end_frame']

    feat_p = os.path.join(*(source_path.split(os.path.sep)[2:]))
    feat_p = feat_p.replace('npy', 'npz')
    """SMPLX-G fps label is not accurate, use the fps from the fps_dict"""
    fps = fps_dict[feat_p]
    file_path = feat_p
    seq_path = os.path.join(raw_dataset_path, file_path)
    if not os.path.exists(seq_path):
        print(f"seq_path not found: {seq_path}")
        continue
    # if not 'frame_labels' in seq_info:
    #     continue

    seq_data = dict(np.load(seq_path, allow_pickle=True))
    downsample_rate = int(fps / target_fps)
    betas = seq_data['betas'][:10].astype(np.float32)
    gender = str(seq_data['gender'].item())
    if enforce_zero_male:
        betas = np.zeros_like(betas)
        gender = 'male'
    if downsample_rate * target_fps != fps:
        trans, poses = downsample(fps, target_fps, seq_data)
        if trans is None:
            print(f'sequence too short: {feat_p}')
            continue
        trans, poses = trans.astype(np.float32), poses.astype(np.float32)
    else:
        trans = seq_data['trans'][::downsample_rate].astype(np.float32)
        poses = seq_data['poses'][::downsample_rate, :66].astype(np.float32)
    # print('trans:', trans.shape, 'poses:', poses.shape)
    if 'Eyes_Japan_Dataset' in source_path or 'MPI_HDM05' in source_path:
        trans = trans[3 * target_fps:]
        poses = poses[3 * target_fps:]
    if 'TotalCapture' in source_path or 'MPI_Limits' in source_path:
        trans = trans[1 * target_fps:]
        poses = poses[1 * target_fps:]
    if 'Transitions_mocap' in source_path:
        trans = trans[int(0.5 * target_fps):]
        poses = poses[int(0.5 * target_fps):]
    # print('cropped trans:', trans.shape, 'poses:', poses.shape)
    # print('start_frame:', start_frame, 'end_frame:', end_frame)
    trans = trans[start_frame:end_frame]
    poses = poses[start_frame:end_frame]

    motion_data = {'gender': gender, 'betas': betas, 'poses': poses, 'trans': trans}
    if int(new_name) % 1000 == 0:
        export_smpl(trans, poses, gender, betas, f'{output_path}/{new_name}.npz')
    joints, pelvis_delta = calc_joints_pelvis_delta(motion_data)
    motion_data['joints'] = joints
    motion_data['pelvis_delta'] = pelvis_delta
    seq_data_dict = {'motion': motion_data, 'data_source': 'hml3d', 'seq_name': new_name, 'feat_p': feat_p}
    text_path = text_dir / f'{new_name}.txt'
    with open(text_path, 'r') as f:
        texts = f.readlines()
    frame_labels = []
    for text in texts:
        # frame_labels.append({
        #     'proc_label': text.split('#')[0],
        #     'start_t': 0,
        #     'end_t': motion_data['trans'].shape[0] / target_fps
        # })

        sentence, token, start_t, end_t = text.split('#')
        start_t, end_t = float(start_t), float(end_t)
        start_t = 0.0 if np.isnan(start_t) else start_t
        end_t = motion_data['trans'].shape[0] / target_fps if np.isnan(end_t) or end_t == 0.0 else end_t
        start_t=min(start_t, motion_data['trans'].shape[0] / target_fps)
        end_t = min(end_t, motion_data['trans'].shape[0] / target_fps)
        frame_labels.append({
            'proc_label': sentence,
            'start_t': start_t,
            'end_t': end_t
        })
    seq_data_dict['frame_labels'] = frame_labels
    dataset[new_name] = seq_data_dict

    # mirror motion: fails for smpl
    # trans_mirror, poses_mirror = mirror_sequence(trans, poses)
    # motion_data = {'gender': gender, 'betas': betas, 'poses': poses_mirror, 'trans': trans_mirror}
    # export_smpl(trans_mirror, poses_mirror, gender, betas, f'{output_path}/M{new_name}.npz')
    # joints, pelvis_delta = calc_joints_pelvis_delta(motion_data)
    # motion_data['joints'] = joints
    # motion_data['pelvis_delta'] = pelvis_delta
    # seq_data_dict = {'motion': motion_data, 'data_source': 'babel', 'seq_name': f'M{new_name}', 'feat_p': feat_p}
    # text_path = text_dir / f'M{new_name}.txt'
    # with open(text_path, 'r') as f:
    #     texts = f.readlines()
    # frame_labels = []
    # for text in texts:
    #     frame_labels.append({
    #         'proc_label': text.split('#')[0],
    #         'start_t': 0,
    #         'end_t': motion_data['trans'].shape[0] / target_fps
    #     })
    # seq_data_dict['frame_labels'] = frame_labels
    # dataset[f'M{new_name}'] = seq_data_dict

    # break


with open(ospj(output_path, 'all.pkl'), 'wb') as f:
    pickle.dump(dataset, f)

# with open(ospj(output_path, 'all.pkl'), 'rb') as f:
#     dataset = pickle.load(f)

for split in splits:
    split_data = []
    for seq_name in splits[split]:
        seq_name = seq_name.strip()
        if seq_name in dataset:
            split_data.append(dataset[seq_name])
    with open(ospj(output_path, f'{split}.pkl'), 'wb') as f:
        pickle.dump(split_data, f)

