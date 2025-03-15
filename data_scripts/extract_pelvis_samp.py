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

def save_pelvis_traj(export_dir, seq_path, text, target_fps=30, zup=True):
    # load samp data
    with open(seq_path, 'rb') as f:
        seq_data = pickle.load(f, encoding='latin1')
    fps = seq_data['mocap_framerate']
    assert fps == 120.0
    motion_data = {}
    downsample_rate = int(fps / target_fps)
    motion_data['trans'] = torch.tensor(seq_data['pose_est_trans'][::downsample_rate].astype(np.float32))
    motion_data['poses'] = torch.tensor(seq_data['pose_est_fullposes'][::downsample_rate, :66].astype(np.float32))
    motion_data['betas'] = torch.tensor(seq_data['shape_est_betas'][:10].astype(np.float32))
    motion_data['gender'] = str(seq_data['shape_est_templatemodel']).split('/')[-2]
    joints, pelvis_delta = calc_joints_pelvis_delta(motion_data)
    motion_data['joints'] = torch.tensor(joints)
    motion_data['pelvis_delta'] = torch.tensor(pelvis_delta)
    num_frames = len(motion_data['trans'])
    poses = torch.tensor(motion_data['poses'], device=device)
    global_orient = transforms.axis_angle_to_matrix(poses[:, :3])  # [num_frames, 3, 3]
    body_pose = transforms.axis_angle_to_matrix(poses[:, 3:66].reshape(num_frames, 21, 3))  # [num_frames, 21, 3, 3]

    max_len = 6 * target_fps
    start_frame = 0
    while start_frame + max_len < num_frames:
        end_frame = min(start_frame + max_len, num_frames)
        if end_frame - start_frame < 2 * target_fps:
            break
        segment_data = {
            'transl': motion_data['trans'][start_frame:end_frame].unsqueeze(0),
            'body_pose': body_pose[start_frame:end_frame].unsqueeze(0),
            'global_orient': global_orient[start_frame:end_frame].unsqueeze(0),
            'joints': motion_data['joints'][start_frame:end_frame].reshape(-1, 22 * 3).unsqueeze(0),
            'gender': motion_data['gender'],
            'betas': motion_data['betas'].expand(end_frame - start_frame, 10).unsqueeze(0),
            'pelvis_delta': motion_data['pelvis_delta'].unsqueeze(0),
            'transf_rotmat': torch.eye(3).unsqueeze(0),
            'transf_transl': torch.zeros(1, 1, 3),
        }
        tensor_dict_to_device(segment_data, device)
        _, _, canonical_segment_data = primitive_utility.canonicalize(segment_data)


        foot_joints = canonical_segment_data['joints'][0, 0].reshape(22, 3)[FOOT_JOINTS_IDX]  # [2, 3]
        foot_height = foot_joints[:, 2].amin()
        transl = canonical_segment_data['transl'].clone()
        transl[:, :, 2] -= foot_height
        joints = canonical_segment_data['joints'][0].reshape(-1, 22, 3)  # [num_frames, 22, 3]
        joints[:, :, 2] -= foot_height

        pelvis = joints[:, 0].detach().cpu().numpy()
        print('pelvis:', pelvis.shape)
        zup_to_yup = np.array([[1, 0, 0],
                               [0, 0, 1],
                               [0, 1, 0]])
        if not zup:
            pelvis = pelvis @ zup_to_yup.T

        export_path = export_dir / f'{Path(seq_path).stem}' / f'{start_frame:04d}_{end_frame:04d}_pelvis.pkl'
        export_path.parent.mkdir(parents=True, exist_ok=True)
        with open(export_path, 'wb') as f:
            pickle.dump({
                'pelvis_traj': pelvis,
                'fps': target_fps,
                'text': text,
            }, f)

        smplx_data = {
            'gender': canonical_segment_data['gender'],
            'betas': canonical_segment_data['betas'][0, 0].detach().cpu(),  # [10]
            'transl': transl[0].detach().cpu(),  # [T, 3]
            'global_orient': canonical_segment_data['global_orient'][0].detach().cpu(),  # [T, 3, 3]
            'body_pose': canonical_segment_data['body_pose'][0].detach().cpu(),  # [T, 21, 3, 3]
        }
        export_path = export_dir / f'{Path(seq_path).stem}' / f'{start_frame:04d}_{end_frame:04d}_smplx.pkl'
        with open(export_path, 'wb') as f:
            pickle.dump(smplx_data, f)

        start_frame += max_len


seq_select = {
    './data/samp/run_random_stageII.pkl': 'run',
    './data/samp/locomotion_random_stageII.pkl': 'walk',
    './data/samp/locomotion_side_stepping_stageII.pkl': 'side step',
}
export_dir = Path('./data/samp_pelvis/ours')
export_dir.mkdir(parents=True, exist_ok=True)
for seq_path in seq_select.keys():
    save_pelvis_traj(export_dir, seq_path, text=seq_select[seq_path], target_fps=30, zup=True)

# export_dir = Path('./data/samp_pelvis/hml3d')
# export_dir.mkdir(parents=True, exist_ok=True)
# for seq_path in seq_select.keys():
#     save_pelvis_traj(export_dir, seq_path, text=seq_select[seq_path], target_fps=20, zup=False)


