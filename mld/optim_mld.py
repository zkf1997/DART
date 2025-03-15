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

@dataclass
class OptimArgs:
    seed: int = 0
    torch_deterministic: bool = True
    device: str = "cuda"
    save_dir = None

    denoiser_checkpoint: str = ''
    optim_input: str = ''
    text_prompt: str = None

    respacing: str = 'ddim10'
    guidance_param: float = 5.0
    export_smpl: int = 0
    zero_noise: int = 0
    use_predicted_joints: int = 0
    batch_size: int = 1
    seed_type: str= 'gt'

    optim_lr: float = 0.01
    optim_steps: int = 300
    optim_unit_grad: int = 1
    optim_anneal_lr: int = 1
    weight_jerk: float = 0.0
    weight_floor: float = 0.0
    init_noise_scale: float = 1.0


def calc_jerk(joints):
    vel = joints[:, 1:] - joints[:, :-1]  # --> B x T-1 x 22 x 3
    acc = vel[:, 1:] - vel[:, :-1]  # --> B x T-2 x 22 x 3
    jerk = acc[:, 1:] - acc[:, :-1]  # --> B x T-3 x 22 x 3
    jerk = torch.sqrt((jerk ** 2).sum(dim=-1))  # --> B x T-3 x 22, compute L1 norm of jerk
    jerk = jerk.amax(dim=[1, 2])  # --> B, Get the max of the jerk across all joints and frames

    return jerk.mean()

def optimize(text_prompt, canonicalized_primitive_dict, goal_joints, joints_mask, denoiser_args, denoiser_model, vae_args, vae_model, diffusion, dataset, optim_args):
    device = optim_args.device
    batch_size = optim_args.batch_size
    future_length = dataset.future_length
    history_length = dataset.history_length
    primitive_length = history_length + future_length
    start_idx = history_length - 1 if optim_args.seed_type == 'repeat' else 0
    end_idx = start_idx + seq_length - 1
    assert 'ddim' in optim_args.respacing
    sample_fn = diffusion.ddim_sample_loop_full_chain

    texts = []
    if ',' in text_prompt:  # contain a time line of multipel actions
        num_rollout = 0
        for segment in text_prompt.split(','):
            action, num_mp = segment.split('*')
            action = compose_texts_with_and(action.split(' and '))
            texts = texts + [action] * int(num_mp)
            num_rollout += int(num_mp)
    else:
        action, num_rollout = text_prompt.split('*')
        action = compose_texts_with_and(action.split(' and '))
        num_rollout = int(num_rollout)
        for _ in range(num_rollout):
            texts.append(action)
    all_text_embedding = encode_text(dataset.clip_model, texts, force_empty_zero=True).to(dtype=torch.float32,
                                                                                      device=device)
    primitive_utility = dataset.primitive_utility

    out_path = optim_args.save_dir
    filename = f'guidance{optim_args.guidance_param}_seed{optim_args.seed}'
    if text_prompt != '':
        filename = text_prompt[:40].replace(' ', '_').replace('.', '') + '_' + filename
    if optim_args.respacing != '':
        filename = f'{optim_args.respacing}_{filename}'
    # if optim_args.smooth:
    #     filename = f'smooth_{filename}'
    if optim_args.zero_noise:
        filename = f'zero_noise_{filename}'
    if optim_args.use_predicted_joints:
        filename = f'use_pred_joints_{filename}'
    filename = f'scale{optim_args.init_noise_scale}_floor{optim_args.weight_floor}_jerk{optim_args.weight_jerk}_{filename}'
    out_path = out_path / f'{optim_args.seed_type}seed' / filename
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
    history_motion_gt = motion_tensor[:, :history_length, :]  # [B, H, D]
    if text_prompt == '':
        optim_args.guidance_param = 0.  # Force unconditioned generation

    def rollout(noise):
        motion_sequences = None
        history_motion = history_motion_gt
        transf_rotmat = torch.eye(3, device=device, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1, 1)
        transf_transl = torch.zeros(3, device=device, dtype=torch.float32).reshape(1, 1, 3).repeat(batch_size, 1, 1)
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
    noise = noise * optim_args.init_noise_scale
    noise.requires_grad_(True)
    reduction_dims = list(range(1, len(noise.shape)))
    criterion = torch.nn.HuberLoss(reduction='mean', delta=1.0)

    optimizer = torch.optim.Adam([noise], lr=lr)
    for i in tqdm(range(optim_steps)):
        optimizer.zero_grad()
        if optim_args.optim_anneal_lr:
            frac = 1.0 - i / optim_steps
            lrnow = frac * lr
            optimizer.param_groups[0]["lr"] = lrnow

        motion_sequences = rollout(noise)
        # joints_diff = (motion_sequences['joints'][:, seq_length - 1, joints_mask] - goal_joints[:, joints_mask]) ** 2
        # joints_diff = torch.sqrt(joints_diff.sum(dim=-1)).mean(dim=1).mean(dim=0)
        # loss_joints = joints_diff
        # print('joints shape:', motion_sequences['joints'].shape, goal_joints.shape, joints_mask.shape)
        loss_joints = criterion(motion_sequences['joints'][:, end_idx, joints_mask], goal_joints[:, joints_mask])
        loss_jerk = calc_jerk(motion_sequences['joints'][:, start_idx:end_idx + 1])
        floor_height = motion_sequences['joints'][:, 0, FOOT_JOINTS_IDX, 2].amin(dim=-1)  # [B], assuming first frame on floor
        foot_height = motion_sequences['joints'][:, start_idx:end_idx + 1, FOOT_JOINTS_IDX, 2].amin(dim=-1)  # [B, T]
        loss_floor = -(foot_height - floor_height.unsqueeze(1)).clamp(max=0).mean()
        loss = loss_joints + optim_args.weight_jerk * loss_jerk + optim_args.weight_floor * loss_floor
        loss.backward()
        if optim_args.optim_unit_grad:
            noise.grad.data /= noise.grad.norm(p=2, dim=reduction_dims, keepdim=True).clamp(min=1e-6)
        optimizer.step()
        # print(f'[{i}/{optim_steps}] loss: {loss.item()} joints_diff: {loss_joints.item()} jerk: {loss_jerk.item()} floor: {loss_floor.item()}')
    print(f'[{i}/{optim_steps}] loss: {loss.item()} joints_diff: {loss_joints.item()} jerk: {loss_jerk.item()} floor: {loss_floor.item()}')

    motion_sequences = rollout(noise)
    # export input sequence
    sequence = {
        'texts': texts,
        'gender': canonicalized_primitive_dict['gender'],
        'betas': canonicalized_primitive_dict['betas'][0],
        'transl': canonicalized_primitive_dict['transl'][0],
        'global_orient': canonicalized_primitive_dict['global_orient'][0],
        'body_pose': canonicalized_primitive_dict['body_pose'][0],
        'joints': canonicalized_primitive_dict['joints'][0],
        'history_length': history_length,
        'future_length': future_length,
    }
    tensor_dict_to_device(sequence, 'cpu')
    with open(os.path.join(out_path, f'input.pkl'), 'wb') as f:
        pickle.dump(sequence, f)

    for idx in range(optim_args.batch_size):
        sequence = {
            'texts': texts,
            'gender': motion_sequences['gender'],
            'betas': motion_sequences['betas'][idx, start_idx:end_idx + 1],
            'transl': motion_sequences['transl'][idx, start_idx:end_idx + 1],
            'global_orient': motion_sequences['global_orient'][idx, start_idx:end_idx + 1],
            'body_pose': motion_sequences['body_pose'][idx, start_idx:end_idx + 1],
            'joints': motion_sequences['joints'][idx, start_idx:end_idx + 1],
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
                'mocap_framerate': dataset.target_fps,  # 30
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
    save_dir = denoiser_checkpoint.parent / denoiser_checkpoint.name.split('.')[0] / 'optim'
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
                                     sequence_path=seq_path,
                                     body_type=vae_args.data_args.body_type,
                                     batch_size=optim_args.batch_size,
                                     device=device,
                                     enforce_gender='male',
                                     enforce_zero_beta=1,
                                     )
    future_length = dataset.future_length
    history_length = dataset.history_length
    primitive_length = history_length + future_length
    primitive_utility = dataset.primitive_utility
    print('body type:', primitive_utility.body_type)

    with open(seq_path, 'rb') as f:
        input_sequence = pickle.load(f)
    seq_length = input_sequence['transl'].shape[0]
    num_rollout = int(np.ceil((seq_length - 1) / future_length)) if optim_args.seed_type == 'repeat' else int(np.ceil((seq_length - history_length) / future_length))
    print(f'seq_length: {seq_length}, num_rollout: {num_rollout}')
    text_prompt = input_sequence['texts'][0] if optim_args.text_prompt is None else optim_args.text_prompt
    text_prompt = f"{text_prompt}*{num_rollout}"

    body_pose = torch.tensor(input_sequence['body_pose'], dtype=torch.float32)
    body_pose = transforms.axis_angle_to_matrix(body_pose.reshape(-1, 3)).reshape(-1, 21, 3, 3).unsqueeze(
        0)  # [1, T, 21, 3, 3]
    global_orient = torch.tensor(input_sequence['global_orient'], dtype=torch.float32)
    global_orient = transforms.axis_angle_to_matrix(global_orient.reshape(-1, 3)).reshape(-1, 3, 3).unsqueeze(
        0)  # [1, T, 3, 3]
    transl = torch.tensor(input_sequence['transl'], dtype=torch.float32).unsqueeze(0)  # [1, T, 3]
    betas = torch.tensor(input_sequence['betas'],
                         dtype=torch.float32) if not dataset.enforce_zero_beta else torch.zeros(10, dtype=torch.float32)
    betas = betas.expand(1, seq_length, 10)  # [1, T, 10]
    # transl[:, 1:-1] = transl[:, 0]
    # body_pose[:, 1:-1] = body_pose[:, 0]
    # global_orient[:, 1:-1] = global_orient[:, 0]
    seq_dict = {
        'gender': dataset.enforce_gender,
        'betas': betas,
        'transl': transl,
        'body_pose': body_pose,
        'global_orient': global_orient,
        'transf_rotmat': torch.eye(3).unsqueeze(0),
        'transf_transl': torch.zeros(1, 1, 3),
    }
    seq_dict = tensor_dict_to_device(seq_dict, device)
    _, _, canonicalized_primitive_dict = primitive_utility.canonicalize(seq_dict)
    body_model = primitive_utility.get_smpl_model(dataset.enforce_gender)
    joints = body_model(return_verts=False,
                        betas=canonicalized_primitive_dict['betas'][0],
                        body_pose=canonicalized_primitive_dict['body_pose'][0],
                        global_orient=canonicalized_primitive_dict['global_orient'][0],
                        transl=canonicalized_primitive_dict['transl'][0]
                        ).joints[:, :22, :]  # [T, 22, 3]
    canonicalized_primitive_dict['joints'] = joints.unsqueeze(0)  # [1, T, 22, 3]
    goal_joints = joints[[-1]].expand(optim_args.batch_size, -1, -1)  # [B, 22, 3]
    joints_mask = torch.ones(22, dtype=torch.bool, device=device)

    optimize(text_prompt, canonicalized_primitive_dict, goal_joints, joints_mask, denoiser_args, denoiser_model, vae_args, vae_model, diffusion, dataset, optim_args)



