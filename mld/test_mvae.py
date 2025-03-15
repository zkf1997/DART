from __future__ import annotations

import os
import random
import time
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

from model.mld_vae import AutoMldVae
from data_loaders.humanml.data.dataset import PrimitiveSequenceDataset, WeightedPrimitiveSequenceDataset, WeightedPrimitiveSequenceDatasetV2
from data_loaders.humanml.data.dataset_hml3d import HML3dDataset
from utils.smpl_utils import get_smplx_param_from_6d
from pytorch3d import transforms
from mld.train_mvae import VAEArgs, DataArgs, TrainArgs, Args
from utils.smpl_utils import *

debug = 0

@dataclass
class TestArgs:
    seed: int = 0
    torch_deterministic: bool = True
    device: str = "cuda"
    batch_size: int = 4

    checkpoint_path: str = None
    test_seq_path: str = None
    pred_mode: str = "rec"
    use_predicted_joints: int = 0
    export_smpl: int = 1

def calc_jerk(joints):
    vel = joints[:, 1:] - joints[:, :-1]  # --> B x T-1 x 22 x 3
    acc = vel[:, 1:] - vel[:, :-1]  # --> B x T-2 x 22 x 3
    jerk = acc[:, 1:] - acc[:, :-1]  # --> B x T-3 x 22 x 3
    jerk = torch.abs(jerk).sum(dim=-1)  # --> B x T-3 x 22, compute L1 norm of jerk
    jerk = jerk.amax(dim=[2])  # --> B, T-3, Get the max of the jerk across all joints

    return jerk


if __name__ == "__main__":
    test_args = tyro.cli(TestArgs)
    # TRY NOT TO MODIFY: seeding
    random.seed(test_args.seed)
    np.random.seed(test_args.seed)
    torch.manual_seed(test_args.seed)
    torch.set_default_dtype(torch.float32)
    torch.backends.cudnn.deterministic = test_args.torch_deterministic
    device = torch.device(test_args.device if torch.cuda.is_available() else "cpu")

    checkpoint_dir = Path(test_args.checkpoint_path).parent
    arg_path = checkpoint_dir / "args.yaml"
    with open(arg_path, "r") as f:
        args = tyro.extras.from_yaml(Args, yaml.safe_load(f))

    model_args = args.model_args
    print('model args:', asdict(model_args))
    model = AutoMldVae(
        **asdict(model_args),
    ).to(device)
    checkpoint = torch.load(test_args.checkpoint_path, map_location=device)
    model_state_dict = checkpoint['model_state_dict']
    if 'latent_mean' not in model_state_dict:
        model_state_dict['latent_mean'] = torch.tensor(0)
    if 'latent_std' not in model_state_dict:
        model_state_dict['latent_std'] = torch.tensor(1)
    print('latent mean:', model_state_dict['latent_mean'], 'latent std:', model_state_dict['latent_std'])
    model.load_state_dict(model_state_dict)
    start_step = checkpoint['num_steps']
    print(f"Loading checkpoint from {test_args.checkpoint_path} at step {start_step}")
    model.eval()

    # load dataset, for normalization and denormalization
    data_args = args.data_args
    # load dataset
    if data_args.dataset == 'mp_seq_v2':
        dataset_class = WeightedPrimitiveSequenceDatasetV2
    elif data_args.dataset == 'hml3d':
        dataset_class = HML3dDataset
    else:
        dataset_class = WeightedPrimitiveSequenceDataset
    dataset = dataset_class(dataset_path=data_args.data_dir,
                                               dataset_name=data_args.dataset,
                                               cfg_path=data_args.cfg_path, prob_static=data_args.prob_static,
                                               enforce_gender=data_args.enforce_gender,
                                               enforce_zero_beta=data_args.enforce_zero_beta,
                                               split='train', device=device,
                                               weight_scheme=data_args.weight_scheme,
                                               load_data=False,
                                               )

    history_length = args.data_args.history_length
    future_length = args.data_args.future_length
    primitive_length = history_length + future_length
    primitive_utility = PrimitiveUtility(device=device, dtype=torch.float32)

    with open(test_args.test_seq_path, 'rb') as f:
        input_sequence = pickle.load(f)
    seq_length = input_sequence['transl'].shape[0]
    gender = 'male'
    body_pose = torch.tensor(input_sequence['body_pose'], dtype=torch.float32).to(device)
    if body_pose.dim() == 2:
        body_pose = transforms.axis_angle_to_matrix(body_pose.reshape(-1, 3)).reshape(-1, 21, 3, 3)
    body_pose = body_pose.unsqueeze(0).repeat(test_args.batch_size, 1, 1, 1, 1)  # [1, T, 21, 3, 3]
    global_orient = torch.tensor(input_sequence['global_orient'], dtype=torch.float32).to(device)
    if global_orient.dim() == 2:
        global_orient = transforms.axis_angle_to_matrix(global_orient.reshape(-1, 3)).reshape(-1, 3, 3)
    global_orient = global_orient.unsqueeze(0).repeat(test_args.batch_size, 1, 1, 1)  # [1, T, 3, 3]
    transl = torch.tensor(input_sequence['transl'], dtype=torch.float32).unsqueeze(0).to(device).repeat(test_args.batch_size, 1, 1)  # [1, T, 3]
    betas = torch.zeros(10, dtype=torch.float32, device=device)
    betas = betas.expand(test_args.batch_size, seq_length, 10)  # [B, T, 10]
    pelvis_delta = primitive_utility.calc_calibrate_offset({
        'betas': betas[:, 0, :],
        'gender': gender,
    })  # [B, 3]
    body_model = primitive_utility.get_smpl_model(gender)
    joints = body_model(return_verts=False, body_pose=body_pose[0], global_orient=global_orient[0], betas=betas[0], transl=transl[0]).joints  # [T, 22, 3]

    transf_rotmat = torch.eye(3, device=device, dtype=torch.float32).unsqueeze(0).repeat(test_args.batch_size, 1, 1)
    transf_transl = torch.zeros(3, device=device, dtype=torch.float32).reshape(1, 1, 3).repeat(test_args.batch_size, 1, 1)
    frame_idx = 0
    rollout_history = None
    motion_sequences = None
    while frame_idx <= seq_length - primitive_length - 1:
        seq_dict = {
            'gender': 'male',
            'betas': betas[:, frame_idx: frame_idx + primitive_length + 1],
            'transl': transl[:, frame_idx: frame_idx + primitive_length + 1],
            'body_pose': body_pose[:, frame_idx: frame_idx + primitive_length + 1],
            'global_orient': global_orient[:, frame_idx: frame_idx + primitive_length + 1],
            'transf_rotmat': torch.eye(3, device=device, dtype=torch.float32).unsqueeze(0).repeat(test_args.batch_size, 1, 1),
            'transf_transl': torch.zeros(3, device=device, dtype=torch.float32).reshape(1, 1, 3).repeat(test_args.batch_size, 1, 1),
        }
        seq_dict = tensor_dict_to_device(seq_dict, device=device)
        _, _, canonicalized_primitive_dict = primitive_utility.canonicalize(seq_dict)
        feature_dict = primitive_utility.calc_features(canonicalized_primitive_dict)
        feature_dict['transl'] = feature_dict['transl'][:, :-1, :]  # [B, T, 3]
        feature_dict['poses_6d'] = feature_dict['poses_6d'][:, :-1, :]  # [B, T, 66]
        feature_dict['joints'] = feature_dict['joints'][:, :-1, :]  # [B, T, 22 * 3]
        motion_tensor_gt = primitive_utility.dict_to_tensor(feature_dict)  # [B, T, D]
        motion_tensor_gt = dataset.normalize(motion_tensor_gt)  # [B, T, D]
        history_motion = motion_tensor_gt[:, :history_length, :] if frame_idx == 0 else rollout_history
        future_motion_gt = motion_tensor_gt[:, history_length:, :]
        if frame_idx == 0:
            transf_rotmat, transf_transl = canonicalized_primitive_dict['transf_rotmat'], canonicalized_primitive_dict['transf_transl']

        if test_args.pred_mode == "rec":
            latent, dist = model.encode(history_motion=history_motion, future_motion=future_motion_gt)
            # print('latent:', latent.mean(), latent.std())
        else:
            latent_shape = [args.model_args.latent_dim[0], test_args.batch_size, args.model_args.latent_dim[1]] # [1, B, D]
            latent = torch.randn(*latent_shape, device=device)
        sample = model.decode(latent, history_motion=history_motion, nfuture=future_length).detach()  # [B, F, D]
        # print(sample.shape)
        future_frames = dataset.denormalize(sample)
        new_history_frames = future_frames[:, -history_length:, :]

        """transform primitive to world coordinate, prepare for serialization"""
        if frame_idx == 0:  # add init history motion
            future_frames = torch.cat([dataset.denormalize(history_motion), future_frames], dim=1)
        # diff = future_frames[0] - future_frames[1]
        # print(diff.abs().max(), diff.abs().min(), diff.abs().mean())
        future_feature_dict = primitive_utility.tensor_to_dict(future_frames)
        future_feature_dict.update(
            {
                'transf_rotmat': transf_rotmat,
                'transf_transl': transf_transl,
                'gender': gender,
                'betas': betas[:, :future_length, :] if frame_idx > 0 else betas[:, :primitive_length, :],
                'pelvis_delta': pelvis_delta,
            }
        )
        future_primitive_dict = primitive_utility.feature_dict_to_smpl_dict(future_feature_dict)
        future_primitive_dict = primitive_utility.transform_primitive_to_world(future_primitive_dict)
        if motion_sequences is None:
            motion_sequences = future_primitive_dict
        else:
            for key in ['transl', 'global_orient', 'body_pose', 'betas', 'joints']:
                motion_sequences[key] = torch.cat([motion_sequences[key], future_primitive_dict[key]],
                                                  dim=1)  # [B, T, ...]

        """update history motion seed, update global transform"""
        history_feature_dict = primitive_utility.tensor_to_dict(new_history_frames)
        history_feature_dict.update(
            {
                'transf_rotmat': transf_rotmat,
                'transf_transl': transf_transl,
                'gender': gender,
                'betas': betas[:, :history_length, :],
                'pelvis_delta': pelvis_delta,
            }
        )
        canonicalized_history_primitive_dict, blended_feature_dict = primitive_utility.get_blended_feature(
            history_feature_dict, use_predicted_joints=test_args.use_predicted_joints)
        transf_rotmat, transf_transl = canonicalized_history_primitive_dict['transf_rotmat'], canonicalized_history_primitive_dict['transf_transl']
        history_motion_tensor = primitive_utility.dict_to_tensor(blended_feature_dict)
        rollout_history = dataset.normalize(history_motion_tensor)  # [B, T, D]

        frame_idx += future_length

    seq_name = Path(test_args.test_seq_path).name.split('.')[0]
    seq_name = seq_name.replace(' ', '_')
    output_dir = checkpoint_dir / str(start_step) /seq_name / test_args.pred_mode
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving results to {output_dir}")
    print()

    # export gt sequence
    sequence = {
        'gender': gender,
        'betas': betas[0, :frame_idx + history_length].cpu(),
        'transl': transl[0, :frame_idx + history_length].cpu(),
        'global_orient': global_orient[0, :frame_idx + history_length].cpu(),
        'body_pose': body_pose[0, :frame_idx + history_length].cpu(),
        'history_length': history_length,
        'future_length': future_length,
    }
    with open(output_dir / 'gt.pkl', 'wb') as f:
        pickle.dump(sequence, f)

    # export generated sequences
    if motion_sequences is None:
        print("No generated sequences")
        exit()
    tensor_dict_to_device(motion_sequences, device='cpu')
    # TODO: cal_jerk
    gt_jerk = calc_jerk(joints[None])
    jerk = calc_jerk(motion_sequences['joints'])
    jerk_dict = {
        'jerk': jerk.cpu().numpy().tolist(),
        'jerk_max': jerk.amax(dim=1).cpu().numpy().tolist(),
        'gt_jerk': gt_jerk.cpu().numpy().tolist(),
        'gt_jerk_max': gt_jerk.amax(dim=1).cpu().numpy().tolist(),
    }
    print('gt jerk max:', jerk_dict['gt_jerk_max'])
    print('jerk max:', jerk_dict['jerk_max'])
    with open(output_dir / 'jerk.json', 'w') as f:
        json.dump(jerk_dict, f)
    for idx in range(test_args.batch_size):
        sequence = {
            'gender': motion_sequences['gender'],
            'betas': motion_sequences['betas'][idx, :frame_idx + history_length],
            'transl': motion_sequences['transl'][idx, :frame_idx + history_length],
            'global_orient': motion_sequences['global_orient'][idx, :frame_idx + history_length],
            'body_pose': motion_sequences['body_pose'][idx, :frame_idx + history_length],
            'joints': motion_sequences['joints'][idx, :frame_idx + history_length],
            'history_length': history_length,
            'future_length': future_length,
        }
        with open(output_dir / f'seq{idx}.pkl', 'wb') as f:
            pickle.dump(sequence, f)

        if test_args.export_smpl:
            poses = transforms.matrix_to_axis_angle(
                torch.cat([sequence['global_orient'].reshape(-1, 1, 3, 3), sequence['body_pose']], dim=1)
            ).reshape(-1, 22 * 3)
            poses = torch.cat([poses, torch.zeros(poses.shape[0], 99).to(dtype=poses.dtype, device=poses.device)], dim=1)
            data_dict = {
                'mocap_framerate': dataset.target_fps,  # 30
                'gender': sequence['gender'],
                'betas': sequence['betas'][0, :10].detach().cpu().numpy(),
                'poses': poses.detach().cpu().numpy(),
                'trans': sequence['transl'].detach().cpu().numpy(),
            }
            with open(output_dir / f'sample_{idx}_smplx.npz', 'wb') as f:
                np.savez(f, **data_dict)


