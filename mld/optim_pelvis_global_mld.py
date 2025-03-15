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
from PIL.ImageSequence import all_frames
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import pickle
import json
import copy

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

debug = 0

def calc_jerk(joints):
    vel = joints[:, 1:] - joints[:, :-1]  # --> B x T-1 x 22 x 3
    acc = vel[:, 1:] - vel[:, :-1]  # --> B x T-2 x 22 x 3
    jerk = acc[:, 1:] - acc[:, :-1]  # --> B x T-3 x 22 x 3
    jerk = torch.sqrt((jerk ** 2).sum(dim=-1))  # --> B x T-3 x 22, compute L1 norm of jerk
    jerk = jerk.amax(dim=[1, 2])  # --> B, Get the max of the jerk across all joints and frames

    return jerk.mean()

def calc_skate(joints, foot_floor_thresh=0.033):
    foot_joints = joints[:, :, FOOT_JOINTS_IDX, :]
    foot_joints_height = foot_joints[:, :, :, 2]  # [B, T, foot]
    foot_joints_diff = torch.norm(foot_joints[:, 1:] - foot_joints[:, :-1], dim=-1, p=2)  # [B, T-1, foot]
    foot_joints_height_consecutive_max = torch.maximum(foot_joints_height[:, :-1],
                                                       foot_joints_height[:, 1:])  # maximum height of current or previous frame
    skate = foot_joints_diff * (
                2 - 2 ** (foot_joints_height_consecutive_max / foot_floor_thresh).clamp(min=0, max=1))  # [B, F, foot]
    return skate.mean()

@dataclass
class OptimArgs:
    seed: int = 0
    torch_deterministic: bool = True
    device: str = "cuda"
    save_dir = None

    denoiser_checkpoint: str = ''
    optim_input: str = ''

    respacing: str = 'ddim10'
    guidance_param: float = 5.0
    export_smpl: int = 0
    zero_noise: int = 0
    use_predicted_joints: int = 0
    batch_size: int = 1
    fps: int = 30
    input_path: str = ''

    optim_lr: float = 0.01
    optim_steps: int = 300
    optim_unit_grad: int = 1
    optim_anneal_lr: int = 1
    weight_skate: float = 0.0
    weight_floor: float = 0.0
    weight_jerk: float = 0.0
    mode: str = 'global'
    init_scale: float = 1.0

def optimize_global(input_path, denoiser_args, denoiser_model, vae_args, vae_model, diffusion, dataset, optim_args):
    device = optim_args.device
    batch_size = optim_args.batch_size
    future_length = dataset.future_length
    history_length = dataset.history_length
    primitive_length = history_length + future_length
    assert 'ddim' in optim_args.respacing
    sample_fn = diffusion.ddim_sample_loop_full_chain

    with open(input_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    frame_idx = torch.tensor(data['frame_idx'], dtype=torch.long, device=device)
    seq_length = np.max(data['frame_idx']) + 1
    joint_traj = torch.tensor(data['traj'], dtype=torch.float32, device=device)  # [T, 3]
    joint_idx = data['joint_idx']
    text_prompt = data['text']
    num_rollout = int(np.ceil((seq_length - history_length) / future_length))
    texts = [text_prompt] * num_rollout
    all_text_embedding = encode_text(dataset.clip_model, texts, force_empty_zero=True).to(dtype=torch.float32,
                                                                                      device=device)
    primitive_utility = dataset.primitive_utility

    out_path = input_path.parent / 'mld_optim_global'
    filename = f'guidance{optim_args.guidance_param}_seed{optim_args.seed}_lr{optim_args.optim_lr}_steps{optim_args.optim_steps}'
    if optim_args.respacing != '':
        filename = f'{optim_args.respacing}_{filename}'
    filename = f'init{optim_args.init_scale}_{filename}'
    if optim_args.zero_noise:
        filename = f'zero_noise_{filename}'
    if optim_args.use_predicted_joints:
        filename = f'use_pred_joints_{filename}'
    filename = f'floor{optim_args.weight_floor}_skate{optim_args.weight_skate}_jerk{optim_args.weight_jerk}_{filename}'
    out_path = out_path / filename
    out_path.mkdir(parents=True, exist_ok=True)

    batch = dataset.get_batch(batch_size=optim_args.batch_size)
    input_motions, model_kwargs = batch[0]['motion_tensor_normalized'], {'y': batch[0]}
    del model_kwargs['y']['motion_tensor_normalized']
    gender = model_kwargs['y']['gender'][0]
    betas = model_kwargs['y']['betas'][:, :primitive_length, :].to(device)  # [B, H+F, 10]
    pelvis_delta = primitive_utility.calc_calibrate_offset({
        'betas': betas[:, 0, :],
        'gender': gender,
    })
    # print(input_motions, model_kwargs)
    input_motions = input_motions.to(device)  # [B, D, 1, T]
    motion_tensor = input_motions.squeeze(2).permute(0, 2, 1)  # [B, T, D]
    motion_dict = primitive_utility.tensor_to_dict(dataset.denormalize(motion_tensor))
    joints = motion_dict['joints'].reshape(batch_size, primitive_length, 22, 3)  # [B, T, 22, 3]
    init_foot_height = joints[:, 0, FOOT_JOINTS_IDX, 2].amin(dim=-1)  # [B]
    history_motion_gt = motion_tensor[:, :history_length, :]  # [B, H, D]
    if text_prompt == '':
        optim_args.guidance_param = 0.  # Force unconditioned generation

    def rollout(noise):
        motion_sequences = None
        history_motion = history_motion_gt
        transf_rotmat = torch.eye(3, device=device, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1, 1)
        transf_transl = torch.zeros(3, device=device, dtype=torch.float32).reshape(1, 1, 3).repeat(batch_size, 1, 1)
        # pdb.set_trace()
        transf_transl[:, :, 2] -= init_foot_height.unsqueeze(1)  # set z to floor height
        for segment_id in range(num_rollout):
            text_embedding = all_text_embedding[segment_id].expand(batch_size, -1)  # [B, 512]
            guidance_param = torch.ones(batch_size, *denoiser_args.model_args.noise_shape).to(device=device) * optim_args.guidance_param
            y = {
                'text_embedding': text_embedding,
                'history_motion_normalized': history_motion,
                'scale': guidance_param,
            }

            x_start_pred = sample_fn(
                denoiser_model,
                (batch_size, *denoiser_args.model_args.noise_shape),
                clip_denoised=False,
                model_kwargs={'y': y},
                skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                init_image=None,
                progress=False,
                noise=noise[segment_id],
            )  # [B, T=1, D]
            # x_start_pred = x_start_pred.clamp(min=-3, max=3)
            # print('x_start_pred:', x_start_pred.mean(), x_start_pred.std(), x_start_pred.min(), x_start_pred.max())
            latent_pred = x_start_pred.permute(1, 0, 2)  # [T=1, B, D]
            future_motion_pred = vae_model.decode(latent_pred, history_motion, nfuture=future_length,
                                                       scale_latent=denoiser_args.rescale_latent)  # [B, F, D], normalized

            future_frames = dataset.denormalize(future_motion_pred)
            new_history_frames = future_frames[:, -history_length:, :]

            """transform primitive to world coordinate, prepare for serialization"""
            if segment_id == 0:  # add init history motion
                future_frames = torch.cat([dataset.denormalize(history_motion), future_frames], dim=1)
            future_feature_dict = primitive_utility.tensor_to_dict(future_frames)
            future_feature_dict.update(
                {
                    'transf_rotmat': transf_rotmat,
                    'transf_transl': transf_transl,
                    'gender': gender,
                    'betas': betas[:, :future_length, :] if segment_id > 0 else betas[:, :primitive_length, :],
                    'pelvis_delta': pelvis_delta,
                }
            )
            future_primitive_dict = primitive_utility.feature_dict_to_smpl_dict(future_feature_dict)
            future_primitive_dict = primitive_utility.transform_primitive_to_world(future_primitive_dict)
            if motion_sequences is None:
                motion_sequences = future_primitive_dict
            else:
                for key in ['transl', 'global_orient', 'body_pose', 'betas', 'joints']:
                    motion_sequences[key] = torch.cat([motion_sequences[key], future_primitive_dict[key]], dim=1)  # [B, T, ...]

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
                history_feature_dict, use_predicted_joints=optim_args.use_predicted_joints)
            transf_rotmat, transf_transl = canonicalized_history_primitive_dict['transf_rotmat'], \
            canonicalized_history_primitive_dict['transf_transl']
            history_motion = primitive_utility.dict_to_tensor(blended_feature_dict)
            history_motion = dataset.normalize(history_motion)  # [B, T, D]

        motion_sequences['texts'] = texts
        return motion_sequences

    optim_steps = optim_args.optim_steps
    lr = optim_args.optim_lr
    noise = torch.randn(num_rollout, batch_size, *denoiser_args.model_args.noise_shape,
                        device=device, dtype=torch.float32)
    # noise = noise.clip(min=-1, max=1)
    noise = noise * optim_args.init_scale
    noise.requires_grad_(True)
    reduction_dims = list(range(1, len(noise.shape)))
    criterion = torch.nn.HuberLoss(reduction='mean', delta=1.0)
    # criterion = torch.nn.MSELoss(reduction='mean')

    optimizer = torch.optim.Adam([noise], lr=lr)
    for i in tqdm(range(optim_steps)):
        optimizer.zero_grad()
        if optim_args.optim_anneal_lr:
            frac = 1.0 - i / optim_steps
            lrnow = frac * lr
            optimizer.param_groups[0]["lr"] = lrnow

        motion_sequences = rollout(noise)
        goal_joints = joint_traj.unsqueeze(0).repeat(batch_size, 1, 1)  # [B, T, 3]
        gen_joints = motion_sequences['joints'][:, frame_idx, joint_idx, :3]  # [B, T, 3]
        if joint_idx == 0:
            gen_joints = gen_joints[:, :, :2]
            goal_joints = goal_joints[:, :, :2]
        loss_joints = criterion(gen_joints, goal_joints)  # only consider xy plane distance
        loss_jerk = calc_jerk(motion_sequences['joints'][:, :seq_length])
        floor_height = 0
        foot_height = motion_sequences['joints'][:, :seq_length, FOOT_JOINTS_IDX, 2].amin(dim=-1)  # [B, T]
        loss_floor = (foot_height - floor_height).abs().mean()
        loss_skate = calc_skate(motion_sequences['joints'][:, :seq_length])
        loss = loss_joints + optim_args.weight_jerk * loss_jerk + optim_args.weight_floor * loss_floor + optim_args.weight_skate * loss_skate
        loss.backward()
        if optim_args.optim_unit_grad:
            noise.grad.data /= noise.grad.norm(p=2, dim=reduction_dims, keepdim=True).clamp(min=1e-6)
        optimizer.step()
        # with torch.no_grad():
        #     noise.clamp_(-4, 4)
        print(f'noise: mean{noise.mean()} std{noise.std()} min{noise.min()} max{noise.max()}')
        print(f'[{i}/{optim_steps}] loss: {loss.item()} joints_diff: {loss_joints.item()} jerk: {loss_jerk.item()} floor: {loss_floor.item()} skate: {loss_skate.item()}')
    print(f'[{i}/{optim_steps}] loss: {loss.item()} joints_diff: {loss_joints.item()} jerk: {loss_jerk.item()} floor: {loss_floor.item()} skate: {loss_skate.item()}')
    with open(out_path / f'loss_{loss.item()}.txt', 'w') as f:
        f.write(f'loss: {loss.item()} joints_diff: {loss_joints.item()} jerk: {loss_jerk.item()} floor: {loss_floor.item()} skate: {loss_skate.item()}')
        f.write(f'noise: mean{noise.mean()} std{noise.std()} min{noise.min()} max{noise.max()}')

    motion_sequences = rollout(noise)
    for idx in range(optim_args.batch_size):
        sequence = {
            'texts': texts,
            'gender': motion_sequences['gender'],
            'betas': motion_sequences['betas'][idx, :seq_length],
            'transl': motion_sequences['transl'][idx, :seq_length],
            'global_orient': motion_sequences['global_orient'][idx, :seq_length],
            'body_pose': motion_sequences['body_pose'][idx, :seq_length],
            'joints': motion_sequences['joints'][idx, :seq_length],
            'history_length': history_length,
            'future_length': future_length,
        }
        tensor_dict_to_device(sequence, 'cpu')
        with open(out_path / f'sample_{idx}.pkl', 'wb') as f:
            pickle.dump(sequence, f)

        # export smplx sequences for blender
        if optim_args.export_smpl:
            poses = transforms.matrix_to_axis_angle(
                torch.cat([sequence['global_orient'].reshape(-1, 1, 3, 3), sequence['body_pose']], dim=1)
            ).reshape(-1, 22 * 3)
            poses = torch.cat([poses, torch.zeros(poses.shape[0], 99).to(dtype=poses.dtype, device=poses.device)],
                              dim=1)
            data_dict = {
                'mocap_framerate': max(30, optim_args.fps),  # 30
                'gender': sequence['gender'],
                'betas': sequence['betas'][0, :10].detach().cpu().numpy(),
                'poses': poses.detach().cpu().numpy(),
                'trans': sequence['transl'].detach().cpu().numpy(),
            }
            with open(out_path / f'sample_{idx}_smplx.npz', 'wb') as f:
                np.savez(f, **data_dict)

    abs_path = out_path.absolute()
    print(f'[Done] Results are at [{abs_path}]')

def optimize_stage(input_path, denoiser_args, denoiser_model, vae_args, vae_model, diffusion, dataset, optim_args):
    device = optim_args.device
    batch_size = optim_args.batch_size
    future_length = dataset.future_length
    history_length = dataset.history_length
    primitive_length = history_length + future_length
    assert 'ddim' in optim_args.respacing
    sample_fn = diffusion.ddim_sample_loop_full_chain

    with open(input_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    frame_idx = data['frame_idx']
    print('target frames:', frame_idx)
    joint_traj = torch.tensor(data['traj'], dtype=torch.float32, device=device)  # [T, 3]
    joint_idx = data['joint_idx']
    text_prompt = data['text']

    primitive_utility = dataset.primitive_utility

    out_path = input_path.parent / 'mld_optim_stage'
    filename = f'guidance{optim_args.guidance_param}_seed{optim_args.seed}_lr{optim_args.optim_lr}_steps{optim_args.optim_steps}'
    if optim_args.respacing != '':
        filename = f'{optim_args.respacing}_{filename}'
    if optim_args.zero_noise:
        filename = f'zero_noise_{filename}'
    filename = f'init{optim_args.init_scale}_{filename}'
    if optim_args.use_predicted_joints:
        filename = f'use_pred_joints_{filename}'
    filename = f'floor{optim_args.weight_floor}_jerk{optim_args.weight_jerk}_{filename}'
    out_path = out_path / filename
    out_path.mkdir(parents=True, exist_ok=True)

    batch = dataset.get_batch(batch_size=optim_args.batch_size)
    input_motions, model_kwargs = batch[0]['motion_tensor_normalized'], {'y': batch[0]}
    del model_kwargs['y']['motion_tensor_normalized']
    gender = model_kwargs['y']['gender'][0]
    betas = model_kwargs['y']['betas'][:, :primitive_length, :].to(device)  # [B, H+F, 10]
    pelvis_delta = primitive_utility.calc_calibrate_offset({
        'betas': betas[:, 0, :],
        'gender': gender,
    })
    # print(input_motions, model_kwargs)
    input_motions = input_motions.to(device)  # [B, D, 1, T]
    motion_tensor = input_motions.squeeze(2).permute(0, 2, 1)  # [B, T, D]
    motion_dict = primitive_utility.tensor_to_dict(dataset.denormalize(motion_tensor))
    joints = motion_dict['joints'].reshape(batch_size, primitive_length, 22, 3)  # [B, T, 22, 3]
    init_foot_height = joints[:, 0, FOOT_JOINTS_IDX, 2].amin(dim=-1)  # [B]
    history_motion_gt = motion_tensor[:, :history_length, :]  # [B, H, D]
    if text_prompt == '':
        optim_args.guidance_param = 0.  # Force unconditioned generation

    transf_rotmat = torch.eye(3, device=device, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1, 1)
    transf_transl = torch.zeros(3, device=device, dtype=torch.float32).reshape(1, 1, 3).repeat(batch_size, 1, 1)
    # pdb.set_trace()
    transf_transl[:, :, 2] -= init_foot_height.unsqueeze(1)  # set z to floor height

    def rollout(noise, texts, all_text_embedding, num_rollout, last_frame, target_frame, history_motion, transf_rotmat, transf_transl):
        motion_sequences = None
        for segment_id in range(num_rollout):
            rest_frames = target_frame - last_frame - segment_id * future_length
            text_embedding = all_text_embedding[segment_id].expand(batch_size, -1)  # [B, 512]
            guidance_param = torch.ones(batch_size, *denoiser_args.model_args.noise_shape).to(device=device) * optim_args.guidance_param
            y = {
                'text_embedding': text_embedding,
                'history_motion_normalized': history_motion,
                'scale': guidance_param,
            }

            x_start_pred = sample_fn(
                denoiser_model,
                (batch_size, *denoiser_args.model_args.noise_shape),
                clip_denoised=False,
                model_kwargs={'y': y},
                skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                init_image=None,
                progress=False,
                noise=noise[segment_id],
            )  # [B, T=1, D]
            # x_start_pred = x_start_pred.clamp(min=-3, max=3)
            # print('x_start_pred:', x_start_pred.mean(), x_start_pred.std(), x_start_pred.min(), x_start_pred.max())
            latent_pred = x_start_pred.permute(1, 0, 2)  # [T=1, B, D]
            future_motion_pred = vae_model.decode(latent_pred, history_motion, nfuture=future_length,
                                                       scale_latent=denoiser_args.rescale_latent)  # [B, F, D], normalized

            future_frames = dataset.denormalize(future_motion_pred)
            future_frames = future_frames[:, :rest_frames, :]
            primitive_frames = torch.cat([dataset.denormalize(history_motion), future_frames], dim=1)
            """transform primitive to world coordinate, prepare for serialization"""
            if segment_id == 0 and last_frame == history_length - 1:  # add init history motion
                future_frames = primitive_frames
            new_history_frames = primitive_frames[:, -history_length:, :]
            future_feature_dict = primitive_utility.tensor_to_dict(future_frames)
            future_feature_dict.update(
                {
                    'transf_rotmat': transf_rotmat,
                    'transf_transl': transf_transl,
                    'gender': gender,
                    'betas': betas[:, :future_length, :] if segment_id > 0 else betas[:, :primitive_length, :],
                    'pelvis_delta': pelvis_delta,
                }
            )
            future_primitive_dict = primitive_utility.feature_dict_to_smpl_dict(future_feature_dict)
            future_primitive_dict = primitive_utility.transform_primitive_to_world(future_primitive_dict)
            if motion_sequences is None:
                motion_sequences = future_primitive_dict
            else:
                for key in ['transl', 'global_orient', 'body_pose', 'betas', 'joints']:
                    motion_sequences[key] = torch.cat([motion_sequences[key], future_primitive_dict[key]], dim=1)  # [B, T, ...]

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
                history_feature_dict, use_predicted_joints=optim_args.use_predicted_joints)
            transf_rotmat, transf_transl = canonicalized_history_primitive_dict['transf_rotmat'], \
            canonicalized_history_primitive_dict['transf_transl']
            history_motion = primitive_utility.dict_to_tensor(blended_feature_dict)
            history_motion = dataset.normalize(history_motion)  # [B, T, D]

        motion_sequences['texts'] = texts  # need to set text_idx, length can be not multiple of future_length, skip for now
        return motion_sequences, history_motion, transf_rotmat, transf_transl

    last_frame = history_length - 1
    motion_sequences = None
    history_motion = history_motion_gt
    for target_idx, target_frame in enumerate(frame_idx):
        num_rollout = int(np.ceil((target_frame - last_frame) / future_length))
        texts = [text_prompt] * num_rollout
        all_text_embedding = encode_text(dataset.clip_model, texts, force_empty_zero=True).to(dtype=torch.float32,
                                                                                              device=device)
        optim_steps = optim_args.optim_steps
        lr = optim_args.optim_lr
        noise = torch.randn(num_rollout, batch_size, *denoiser_args.model_args.noise_shape,
                            device=device, dtype=torch.float32)
        # noise = noise.clip(min=-1, max=1)
        noise = noise * optim_args.init_scale
        noise.requires_grad_(True)
        reduction_dims = list(range(1, len(noise.shape)))
        criterion = torch.nn.HuberLoss(reduction='mean', delta=1.0)
        # criterion = torch.nn.MSELoss(reduction='mean')

        optimizer = torch.optim.Adam([noise], lr=lr)
        for i in tqdm(range(optim_steps)):
            optimizer.zero_grad()
            if optim_args.optim_anneal_lr:
                frac = 1.0 - i / optim_steps
                lrnow = frac * lr
                optimizer.param_groups[0]["lr"] = lrnow

            tmp_motion_sequences, tmp_history_motion, tmp_transf_rotmat, tmp_transf_transl = rollout(noise, texts, all_text_embedding, num_rollout, last_frame, target_frame, history_motion, transf_rotmat, transf_transl)
            # pdb.set_trace()

            goal_joints = joint_traj[target_idx].unsqueeze(0).repeat(batch_size, 1)  # [B, 3]
            gen_joints = tmp_motion_sequences['joints'][:, -1, joint_idx, :3]  # [B, 3]
            if joint_idx == 0:
                gen_joints = gen_joints[:, :2]
                goal_joints = goal_joints[:, :2]
            loss_joints = criterion(gen_joints, goal_joints)  # only consider xy plane distance
            loss_jerk = calc_jerk(tmp_motion_sequences['joints'])
            foot_height = tmp_motion_sequences['joints'][:, :, FOOT_JOINTS_IDX, 2].amin(dim=-1)  # [B, T]
            floor_height = 0.0
            loss_floor = (foot_height - floor_height).abs().mean()
            loss_skate = calc_skate(tmp_motion_sequences['joints'][:, :])
            loss = loss_joints + optim_args.weight_jerk * loss_jerk + optim_args.weight_floor * loss_floor + optim_args.weight_skate * loss_skate
            loss.backward()
            if optim_args.optim_unit_grad:
                noise.grad.data /= noise.grad.norm(p=2, dim=reduction_dims, keepdim=True).clamp(min=1e-6)
            optimizer.step()
            # with torch.no_grad():
            #     noise.clamp_(-4, 4)
            print(f'noise: mean{noise.mean()} std{noise.std()} min{noise.min()} max{noise.max()}')
            print(f'[{i}/{optim_steps}] loss: {loss.item()} joints_diff: {loss_joints.item()} jerk: {loss_jerk.item()} floor: {loss_floor.item()} skate: {loss_skate.item()}')
        print(f'[{i}/{optim_steps}] loss: {loss.item()} joints_diff: {loss_joints.item()} jerk: {loss_jerk.item()} floor: {loss_floor.item()} skate: {loss_skate.item()}')
        with open(out_path / f'target{target_idx}_loss_{loss.item()}.txt', 'w') as f:
            f.write(f'loss: {loss.item()} joints_diff: {loss_joints.item()} jerk: {loss_jerk.item()} floor: {loss_floor.item()} skate: {loss_skate.item()}')
            f.write(f'noise: mean{noise.mean()} std{noise.std()} min{noise.min()} max{noise.max()}')

        last_frame = target_frame
        if motion_sequences is None:
            motion_sequences = tmp_motion_sequences
            print('init goal')
        else:
            for key in ['transl', 'global_orient', 'body_pose', 'betas', 'joints']:
                motion_sequences[key] = torch.cat([motion_sequences[key], tmp_motion_sequences[key].detach()], dim=1)  # [B, T, ...]
            motion_sequences['texts'] += tmp_motion_sequences['texts']
            print('texts:', motion_sequences['texts'])
        history_motion = tmp_history_motion.detach()
        transf_transl, transf_rotmat = tmp_transf_transl.detach(), tmp_transf_rotmat.detach()

    for idx in range(optim_args.batch_size):
        sequence = {
            'texts': motion_sequences['texts'],
            'gender': motion_sequences['gender'],
            'betas': motion_sequences['betas'][idx],
            'transl': motion_sequences['transl'][idx],
            'global_orient': motion_sequences['global_orient'][idx],
            'body_pose': motion_sequences['body_pose'][idx],
            'joints': motion_sequences['joints'][idx],
            'history_length': history_length,
            'future_length': future_length,
        }
        tensor_dict_to_device(sequence, 'cpu')
        with open(out_path / f'sample_{idx}.pkl', 'wb') as f:
            pickle.dump(sequence, f)

        # export smplx sequences for blender
        if optim_args.export_smpl:
            poses = transforms.matrix_to_axis_angle(
                torch.cat([sequence['global_orient'].reshape(-1, 1, 3, 3), sequence['body_pose']], dim=1)
            ).reshape(-1, 22 * 3)
            poses = torch.cat([poses, torch.zeros(poses.shape[0], 99).to(dtype=poses.dtype, device=poses.device)],
                              dim=1)
            data_dict = {
                'mocap_framerate': max(30, optim_args.fps),  # 30
                'gender': sequence['gender'],
                'betas': sequence['betas'][0, :10].detach().cpu().numpy(),
                'poses': poses.detach().cpu().numpy(),
                'trans': sequence['transl'].detach().cpu().numpy(),
            }
            with open(out_path / f'sample_{idx}_smplx.npz', 'wb') as f:
                np.savez(f, **data_dict)

    abs_path = out_path.absolute()
    print(f'[Done] Results are at [{abs_path}]')

if __name__ == '__main__':
    optim_args = tyro.cli(OptimArgs)
    # TRY NOT TO MODIFY: seeding
    random.seed(optim_args.seed)
    np.random.seed(optim_args.seed)
    torch.manual_seed(optim_args.seed)
    torch.set_default_dtype(torch.float32)
    torch.backends.cudnn.deterministic = optim_args.torch_deterministic
    device = torch.device(optim_args.device if torch.cuda.is_available() else "cpu")
    optim_args.device = device

    denoiser_args, denoiser_model, vae_args, vae_model = load_mld(optim_args.denoiser_checkpoint, device)
    denoiser_checkpoint = Path(optim_args.denoiser_checkpoint)
    save_dir = denoiser_checkpoint.parent / denoiser_checkpoint.name.split('.')[0] / 'optim_pelvis_global'
    save_dir.mkdir(parents=True, exist_ok=True)
    optim_args.save_dir = save_dir

    diffusion_args = denoiser_args.diffusion_args
    diffusion_args.respacing = optim_args.respacing
    print('diffusion_args:', asdict(diffusion_args))
    diffusion = create_gaussian_diffusion(diffusion_args)

    # load initial seed dataset
    seq_path = Path(optim_args.optim_input)
    dataset = SinglePrimitiveDataset(cfg_path=vae_args.data_args.cfg_path,  # cfg path from model checkpoint
                                     dataset_path=vae_args.data_args.data_dir,  # dataset path from model checkpoint
                                     sequence_path='./data/stand.pkl' if optim_args.fps == 30 else './data/stand_20fps.pkl',
                                     batch_size=optim_args.batch_size,
                                     device=device,
                                     enforce_gender='male',
                                     enforce_zero_beta=1,
                                     clip_to_seq_length=True,
                                     )
    print('dataset fps:', dataset.target_fps)

    if optim_args.mode == 'global':
        optimize_global(Path(optim_args.input_path), denoiser_args, denoiser_model, vae_args, vae_model, diffusion, dataset, optim_args)
    else:
        optimize_stage(Path(optim_args.input_path), denoiser_args, denoiser_model, vae_args, vae_model, diffusion,
                        dataset, optim_args)



