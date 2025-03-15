from __future__ import annotations

import os
import pdb
import random
import time
from typing import Literal
from dataclasses import dataclass, asdict, make_dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
import yaml
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import pickle
import json
import copy

from utils.smpl_utils import *
from pytorch3d import transforms

from scipy.spatial.transform import Rotation as R

debug = 0
zup_to_yup = np.array([[1, 0, 0],
                           [0, 0, 1],
                           [0, 1, 0]])
hml3d_to_canonical = R.from_euler('xyz', [90, 0, 180], degrees=True).as_matrix()

def convert_to_hml3d(seq_path, body_type='smplx'):
    device = 'cuda'
    primitive_utility = PrimitiveUtility(device=device, body_type=body_type)
    with open(seq_path, 'rb') as f:
        input_sequence = pickle.load(f)
    seq_length = input_sequence['transl'].shape[0]
    print(seq_path, seq_length)

    body_pose = torch.tensor(input_sequence['body_pose'], dtype=torch.float32)
    body_pose = transforms.axis_angle_to_matrix(body_pose.reshape(-1, 3)).reshape(-1, 21, 3, 3).unsqueeze(
        0)  # [1, T, 21, 3, 3]
    global_orient = torch.tensor(input_sequence['global_orient'], dtype=torch.float32)
    global_orient = transforms.axis_angle_to_matrix(global_orient.reshape(-1, 3)).reshape(-1, 3, 3).unsqueeze(
        0)  # [1, T, 3, 3]
    transl = torch.tensor(input_sequence['transl'], dtype=torch.float32).unsqueeze(0)  # [1, T, 3]
    betas = torch.zeros(10, dtype=torch.float32)
    betas = betas.expand(1, seq_length, 10)  # [1, T, 10]
    seq_dict = {
        'gender': 'male',
        'betas': betas,
        'transl': transl,
        'body_pose': body_pose,
        'global_orient': global_orient,
        'transf_rotmat': torch.eye(3).unsqueeze(0),
        'transf_transl': torch.zeros(1, 1, 3),
    }
    seq_dict = tensor_dict_to_device(seq_dict, device)
    _, _, canonicalized_primitive_dict = primitive_utility.canonicalize(seq_dict)
    body_model = primitive_utility.get_smpl_model(seq_dict['gender'])
    joints = body_model(return_verts=False,
                        betas=canonicalized_primitive_dict['betas'][0],
                        body_pose=canonicalized_primitive_dict['body_pose'][0],
                        global_orient=canonicalized_primitive_dict['global_orient'][0],
                        transl=canonicalized_primitive_dict['transl'][0]
                        ).joints[:, :22, :]  # [T, 22, 3]
    joints_copy = joints.clone()
    joints = joints.detach().cpu().numpy()
    floor_height = joints[0, FOOT_JOINTS_IDX, 2].min()
    joints[:, :, 2] -= floor_height
    # joints = joints @ zup_to_yup.T
    joints = joints @ hml3d_to_canonical
    output_path = str(seq_path).replace('.pkl', '_hml3d.npy')
    np.save(output_path, joints)

    # transf_rotmat = torch.tensor(zup_to_yup, dtype=torch.float32).to(device)
    # transf_rotmat = torch.tensor(hml3d_to_canonical.T, dtype=torch.float32).to(device)
    # transf_transl = torch.tensor([0, 0, -floor_height], dtype=torch.float32).to(device)
    # transf_transl = torch.einsum('ij,j->i', transf_rotmat, transf_transl).reshape(1, 1, 3)
    # print(transf_rotmat, transf_transl)
    # seq_dict = {
    #     'gender': 'male',
    #     'betas': betas,
    #     'transl': canonicalized_primitive_dict['transl'],
    #     'body_pose': canonicalized_primitive_dict['body_pose'],
    #     'global_orient': canonicalized_primitive_dict['global_orient'],
    #     'joints': joints_copy,
    #     'transf_rotmat': transf_rotmat.reshape(1, 3, 3),
    #     'transf_transl': transf_transl,
    # }
    # seq_dict = tensor_dict_to_device(seq_dict, device)
    # sequence = primitive_utility.transform_primitive_to_world(seq_dict)
    # poses = transforms.matrix_to_axis_angle(
    #     torch.cat([sequence['global_orient'][0].reshape(-1, 1, 3, 3), sequence['body_pose'][0]], dim=1)
    # ).reshape(-1, 22 * 3)
    # poses = torch.cat([poses, torch.zeros(poses.shape[0], 99).to(dtype=poses.dtype, device=poses.device)],
    #                   dim=1)
    # data_dict = {
    #     'mocap_framerate': 30,  # 30
    #     'gender': sequence['gender'],
    #     'betas': sequence['betas'][0, 0, :10].detach().cpu().numpy(),
    #     'poses': poses.detach().cpu().numpy(),
    #     'trans': sequence['transl'][0].detach().cpu().numpy(),
    # }
    # output_path = seq_path.replace('.pkl', '_smpl.npz')
    # with open(output_path, 'wb') as f:
    #     np.savez(f, **data_dict)


# seq_list = [
# './data/opt_eval_20fps/0_walk.pkl',
# './data/opt_eval_20fps/1_run forward.pkl',
# './data/opt_eval_20fps/2_jump forward.pkl',
# './data/opt_eval_20fps/3_pace in circles.pkl',
# './data/opt_eval_20fps/4_crawl.pkl',
# './data/opt_eval_20fps/5_dance.pkl',
# './data/opt_eval_20fps/6_walk backwards.pkl',
# './data/opt_eval_20fps/7_climb down stairs.pkl',
# './data/opt_eval_20fps/8_sit down.pkl',
# ]
seq_list = list(Path('./data/opt_eval_20fps_smplh_1f/').glob('*.pkl'))
body_type = 'smplh'

for seq_path in seq_list:
    convert_to_hml3d(seq_path, body_type=body_type)
