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
import trimesh

from model.mld_denoiser import DenoiserMLP, DenoiserTransformer
from model.mld_vae import AutoMldVae
from data_loaders.humanml.data.dataset import WeightedPrimitiveSequenceDataset, SinglePrimitiveDataset
from utils.smpl_utils import *
from utils.misc_util import encode_text, compose_texts_with_and
from pytorch3d import transforms
from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps
from diffusion.resample import create_named_schedule_sampler

from mld.train_mvae import Args as MVAEArgs
from mld.train_mvae import DataArgs, TrainArgs
from mld.train_mld import DenoiserArgs, MLDArgs, create_gaussian_diffusion, DenoiserMLPArgs, DenoiserTransformerArgs
from mld.rollout_mld import load_mld, ClassifierFreeWrapper

import smplx
from VolumetricSMPL import attach_volume
from coap import attach_coap
from time import perf_counter

def get_scene_points(scene_path):
    scene_path = Path(scene_path).parent / 'points_16384.ply'
    if scene_path in scene_assets:
        return scene_assets[scene_path]
    else:
        scene_points = trimesh.load(scene_path).vertices  # [N, 3]
        scene_points = torch.tensor(scene_points, dtype=torch.float32, device=device)
        scene_assets[scene_path] = scene_points
        return scene_points

def calc_coll_coap(seq_data):
    scene_points = get_scene_points(seq_data['scene_path']).unsqueeze(0)
    num_frames = seq_data['betas'].shape[0]
    coll = []
    for idx in range(num_frames):
        smpl_params = {
            'betas': seq_data['betas'][[idx]].reshape(-1, 10),
            'body_pose': transforms.matrix_to_axis_angle(seq_data['body_pose'][[idx]]).reshape(-1, 21 * 3),
            'global_orient': transforms.matrix_to_axis_angle(seq_data['global_orient'][[idx]]).reshape(-1, 3),
            'transl': seq_data['transl'][[idx]].reshape(-1, 3),
        }
        smpl_params = tensor_dict_to_device(smpl_params, device=device)

        smpl_output = coap(**smpl_params, return_full_pose=True)
        # Ensure valid SMPL variables (pose parameters, joints, and vertices)
        assert coap.joint_mapper is None, "VolumetricSMPL requires valid SMPL joints as input."

        loss_collision, _ = coap.coap.collision_loss(scene_points, smpl_output)  # collisions with other geometris
        loss_collision = loss_collision.mean()
        coll.append(loss_collision.item())

    return np.mean(coll)

def calc_volsmpl(seq_data):
    scene_points = get_scene_points(seq_data['scene_path']).unsqueeze(0)
    num_frames = seq_data['betas'].shape[0]
    coll = []
    for idx in range(num_frames):
        smpl_params = {
            'betas': seq_data['betas'][[idx]].reshape(-1, 10),
            'body_pose': transforms.matrix_to_axis_angle(seq_data['body_pose'][[idx]]).reshape(-1, 21 * 3),
            'global_orient': transforms.matrix_to_axis_angle(seq_data['global_orient'][[idx]]).reshape(-1, 3),
            'transl': seq_data['transl'][[idx]].reshape(-1, 3),
        }
        smpl_params = tensor_dict_to_device(smpl_params, device=device)
        smpl_output = vol_smpl(**smpl_params, return_full_pose=True)

        # Ensure valid SMPL variables (pose parameters, joints, and vertices)
        assert vol_smpl.joint_mapper is None, "VolumetricSMPL requires valid SMPL joints as input."

        point_sdfs = vol_smpl.volume.query(points=scene_points, smpl_output=smpl_output)  # [B=1, N]
        loss_collision = torch.relu(-point_sdfs).sum(-1)
        coll.append(loss_collision.item())

    return np.mean(coll)

def eval_results(results_list, is_coap=False):
    metrics = {
        'coll_coap': [],
        'coll_volsmpl': [],
        'goal_dist': [],
        'time': [],
        'time_per_frame': [],
        'memory': [],
        'memory_per_frame': [],
    }
    for result in tqdm(results_list):
        with open(result, 'rb') as f:
            data = pickle.load(f)
        time = data['sync_time']
        memory = data['max_memory']
        num_frames = 16 if is_coap else data['betas'].shape[0]
        metrics['time'].append(time)
        metrics['memory'].append(memory)
        metrics['time_per_frame'].append(time / num_frames)
        metrics['memory_per_frame'].append(memory / num_frames)
        pelvis = data['joints'][-1, 0]  # [3,]
        scene_name = Path(data['scene_path']).parent.stem
        goal_location = goal_dict[scene_name]
        goal_dist = np.linalg.norm(pelvis[:2] - goal_location[:2])
        metrics['goal_dist'].append(goal_dist)
        metrics['coll_coap'].append(calc_coll_coap(data))
        metrics['coll_volsmpl'].append(calc_volsmpl(data))

    for key in metrics.keys():
        metrics[key] = np.mean(metrics[key])
    return metrics

device = 'cuda'
print('create vol smpl model')
vol_smpl = smplx.create(body_model_dir, model_type='smplx',
                        gender='neutral',
                        ext='npz',
                        num_pca_comps=10,
                        batch_size=1)
attach_volume(vol_smpl)
vol_smpl = vol_smpl.to(device)
print('create coap model')
coap = smplx.create(body_model_dir, model_type='smplx',
                    gender='neutral', ext='npz',
                    num_pca_comps=10,
                    batch_size=1)
attach_coap(coap)
coap = coap.to(device)
scene_assets = {}
with open('./VolSMPL/cfg/egobody.json', 'r') as f:
    task_cfg = json.load(f)
goal_dict = {}
for task in task_cfg:
    scene_name = Path(task['scene_dir']).stem
    goal_dict[scene_name] = np.array(task['interactions'][0]['goal_loc'])

volsmpl_results = Path('./mld_denoiser/mld_fps_clip_repeat_euler/checkpoint_300000/optim_volsmpl').glob('*_coll1.0*/sample*.pkl')

metrics = {
    'VolSMPL': eval_results(list(volsmpl_results), is_coap=False),
}

print(metrics)
with open('./VolSMPL/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)
