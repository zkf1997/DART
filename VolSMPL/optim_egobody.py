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

debug = 0

@dataclass
class OptimArgs:
    seed: int = 0
    torch_deterministic: bool = True
    device: str = "cuda"
    save_dir = None

    denoiser_checkpoint: str = ''

    respacing: str = 'ddim10'
    guidance_param: float = 5.0
    export_smpl: int = 0
    zero_noise: int = 0
    use_predicted_joints: int = 0
    batch_size: int = 1

    optim_lr: float = 0.01
    optim_steps: int = 300
    optim_unit_grad: int = 1
    optim_anneal_lr: int = 1
    weight_jerk: float = 0.0
    weight_collision: float = 0.0
    weight_contact: float = 0.0
    weight_skate: float = 0.0
    load_cache: int = 0
    contact_thresh: float = 0.03
    init_noise_scale: float = 1.0

    interaction_cfg: str = './data/optim_interaction/climb_up_stairs.json'
    loss_type: str = 'coap'
    volsmpl_gender: str = 'neutral'
    num_points: int = 16384
    coap_max_frames: int = 10
    num_seq: int = 1


def calc_coll_cont_volsmpl(motion_sequences, scene_assets, contact_thresh, vol_smpl, beta=1.0):
    B, T = motion_sequences['joints'].shape[:2]
    scene_points = scene_assets['scene_points'].unsqueeze(0).repeat(B * T, 1, 1)
    smpl_params = {
        'betas': motion_sequences['betas'].reshape(-1, 10),
        'body_pose': transforms.matrix_to_axis_angle(motion_sequences['body_pose']).reshape(-1, 21 * 3),
        'global_orient': transforms.matrix_to_axis_angle(motion_sequences['global_orient']).reshape(-1, 3),
        'transl': motion_sequences['transl'].reshape(-1, 3),
    }
    smpl_output = vol_smpl(**smpl_params, return_full_pose=True)
    # Ensure valid SMPL variables (pose parameters, joints, and vertices)
    assert vol_smpl.joint_mapper is None, "VolumetricSMPL requires valid SMPL joints as input."

    point_sdfs = vol_smpl.volume.query(points=scene_points, smpl_output=smpl_output)  # [B*T, N]
    point_sdfs = point_sdfs.reshape(B, T, -1)  # [B, T, N]
    loss_collision = torch.relu(-point_sdfs).sum(-1)  # [B, T]
    loss_contact = -1.0 / beta * torch.log(torch.exp(-beta * torch.relu(point_sdfs)).sum(-1))  # [B, T]

    return loss_collision.mean(), loss_contact.mean()

def calc_coll_cont_coap(motion_sequences, scene_assets, contact_thresh, coap):
    B, T = motion_sequences['joints'].shape[:2]
    assert B == 1
    # select max_frames frames to calculate collision and contact loss
    if T > optim_args.coap_max_frames:
        indices = torch.linspace(0, T - 1, optim_args.coap_max_frames, dtype=torch.long, device=motion_sequences['joints'].device)
    else:
        indices = torch.arange(T, device=motion_sequences['joints'].device)
    scene_points = scene_assets['scene_points'].unsqueeze(0).repeat(indices.shape[0], 1, 1)
    smpl_params = {
        'betas': motion_sequences['betas'].reshape(-1, 10)[indices],
        'body_pose': transforms.matrix_to_axis_angle(motion_sequences['body_pose']).reshape(-1, 21 * 3)[indices],
        'global_orient': transforms.matrix_to_axis_angle(motion_sequences['global_orient']).reshape(-1, 3)[indices],
        'transl': motion_sequences['transl'].reshape(-1, 3)[indices],
    }
    smpl_output = coap(**smpl_params, return_full_pose=True)
    # Ensure valid SMPL variables (pose parameters, joints, and vertices)
    assert coap.joint_mapper is None, "VolumetricSMPL requires valid SMPL joints as input."

    loss_collision, _ = coap.coap.collision_loss(scene_points, smpl_output)  # collisions with other geometris
    loss_collision = loss_collision.mean()
    loss_contact = torch.zeros_like(loss_collision)

    return loss_collision, loss_contact


def calc_jerk(joints):
    vel = joints[:, 1:] - joints[:, :-1]  # --> B x T-1 x 22 x 3
    acc = vel[:, 1:] - vel[:, :-1]  # --> B x T-2 x 22 x 3
    jerk = acc[:, 1:] - acc[:, :-1]  # --> B x T-3 x 22 x 3
    jerk = torch.sqrt((jerk ** 2).sum(dim=-1))  # --> B x T-3 x 22, compute L1 norm of jerk
    jerk = jerk.amax(dim=[1, 2])  # --> B, Get the max of the jerk across all joints and frames

    return jerk.mean()

def optimize(history_motion_tensor, transf_rotmat, transf_transl, text_prompt, goal_joints, joints_mask,
             gender, betas, pelvis_delta, scene_assets
             ):
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

    print('create vol smpl model')
    vol_smpl = smplx.create(body_model_dir, model_type='smplx',
                            gender=optim_args.volsmpl_gender,  # the original experiments are run using neutral gender model
                            ext='npz',
                            num_pca_comps=10,
                            batch_size=(num_rollout * future_length + history_length) * batch_size)
    attach_volume(vol_smpl)
    vol_smpl = vol_smpl.to(device)
    print('create coap model')
    num_frames = (num_rollout * future_length + history_length)
    coap = smplx.create(body_model_dir, model_type='smplx',
                        gender=optim_args.volsmpl_gender,  # the original experiments are run using neutral gender model
                        ext='npz',
                        num_pca_comps=10,
                        batch_size=min(num_frames, optim_args.coap_max_frames))
    attach_coap(coap)
    coap = coap.to(device)

    def rollout(noise, history_motion_tensor, transf_rotmat, transf_transl):
        motion_sequences = None
        history_motion = history_motion_tensor
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

        return motion_sequences, history_motion.detach(), transf_rotmat.detach(), transf_transl.detach()

    optim_steps = optim_args.optim_steps
    lr = optim_args.optim_lr
    noise = torch.randn(num_rollout, batch_size, *denoiser_args.model_args.noise_shape,
                        device=device, dtype=torch.float32)
    # noise = noise.clip(min=-1, max=1)
    noise = noise * optim_args.init_noise_scale
    noise.requires_grad_(True)
    reduction_dims = list(range(1, len(noise.shape)))
    criterion = torch.nn.HuberLoss(reduction='mean', delta=1.0)

    sync_times = []
    max_memory = []

    optimizer = torch.optim.Adam([noise], lr=lr)
    for i in tqdm(range(optim_steps)):
        optimizer.zero_grad()
        if optim_args.optim_anneal_lr:
            frac = 1.0 - i / optim_steps
            lrnow = frac * lr
            optimizer.param_groups[0]["lr"] = lrnow

        motion_sequences, new_history_motion_tensor, new_transf_rotmat, new_transf_transl = rollout(noise,
                                                                                                    history_motion_tensor,
                                                                                                    transf_rotmat,
                                                                                                    transf_transl)
        global_joints = motion_sequences['joints']  # [B, T, 22, 3]
        B, T, _, _ = global_joints.shape

        torch.cuda.reset_peak_memory_stats(device=device)
        torch.cuda.synchronize()
        start_time = perf_counter()

        if optim_args.loss_type == 'coap':
            loss_collision, loss_floor_contact = calc_coll_cont_coap(motion_sequences, scene_assets,
                                                                     optim_args.contact_thresh, coap)
        else:
            loss_collision, loss_floor_contact = calc_coll_cont_volsmpl(motion_sequences, scene_assets,
                                                                     optim_args.contact_thresh, vol_smpl)

        torch.cuda.synchronize()
        end_time = perf_counter()
        sync_times.append(end_time - start_time)
        max_memory.append(torch.cuda.max_memory_allocated(device=device) / 1024 / 1024 / 1024)

        loss_joints = criterion(motion_sequences['joints'][:, -1, joints_mask], goal_joints[:, joints_mask])
        loss_jerk = calc_jerk(motion_sequences['joints'])
        loss = loss_joints + optim_args.weight_collision * loss_collision + optim_args.weight_jerk * loss_jerk + optim_args.weight_contact * loss_floor_contact

        loss.backward()
        if optim_args.optim_unit_grad:
            noise.grad.data /= noise.grad.norm(p=2, dim=reduction_dims, keepdim=True).clamp(min=1e-6)
        if i == optim_steps - 1:
            motion_sequences['loss_joints'] = loss_joints.item()
            motion_sequences['loss_jerk'] = loss_jerk.item()
            motion_sequences['loss_collision'] = loss_collision.item()
        else:
            optimizer.step()
        print(f'[{i}/{optim_steps}] loss: {loss.item()} loss_joints: {loss_joints.item()} loss_collision: {loss_collision.item()} loss_jerk: {loss_jerk.item()} loss_floor_contact: {loss_floor_contact.item()}')

    for key in motion_sequences:
        if torch.is_tensor(motion_sequences[key]):
            motion_sequences[key] = motion_sequences[key].detach()
            motion_sequences[key].requires_grad_(False)

    motion_sequences['texts'] = texts
    motion_sequences['sync_time'] = np.array(sync_times).mean()
    motion_sequences['max_memory'] = np.array(max_memory).mean()
    motion_sequences['num_frames'] = num_frames
    return motion_sequences, new_history_motion_tensor.detach(), new_transf_rotmat.detach(), new_transf_transl.detach()

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
    save_dir = denoiser_checkpoint.parent / denoiser_checkpoint.name.split('.')[0] / f'optim_{optim_args.loss_type}'
    save_dir.mkdir(parents=True, exist_ok=True)
    optim_args.save_dir = save_dir

    diffusion_args = denoiser_args.diffusion_args
    assert 'ddim' in optim_args.respacing
    diffusion_args.respacing = optim_args.respacing
    print('diffusion_args:', asdict(diffusion_args))
    diffusion = create_gaussian_diffusion(diffusion_args)
    sample_fn = diffusion.ddim_sample_loop_full_chain

    # load initial seed dataset
    dataset = SinglePrimitiveDataset(cfg_path=vae_args.data_args.cfg_path,  # cfg path from model checkpoint
                                     dataset_path=vae_args.data_args.data_dir,  # dataset path from model checkpoint
                                     sequence_path='./data/stand.pkl',
                                     batch_size=optim_args.batch_size,
                                     device=device,
                                     enforce_gender='male',
                                     enforce_zero_beta=1,
                                     )
    future_length = dataset.future_length
    history_length = dataset.history_length
    primitive_length = history_length + future_length
    primitive_utility = dataset.primitive_utility
    batch_size = optim_args.batch_size
    assert batch_size == 1, 'batch size should be 1'

    def generate_scene(interaction_cfg):

        scene_dir = Path(interaction_cfg['scene_dir'])
        interaction_name = scene_dir.stem + '_' + interaction_cfg['interaction_name'].replace(' ', '_')
        scene_with_floor_mesh = trimesh.load(scene_dir / 'mesh_floor_zup.ply', process=False, force='mesh')
        scene_points = trimesh.load(scene_dir / f'points_{optim_args.num_points}.ply', process=False, force='mesh').vertices
        scene_points = torch.tensor(scene_points, dtype=torch.float32, device=device)  # [N, 3]
        print('scene_points:', scene_points.shape)

        scene_assets = {
            'scene_with_floor_mesh': scene_with_floor_mesh,
            'scene_points': scene_points,
            'floor_height': interaction_cfg['floor_height'],
        }

        out_path = optim_args.save_dir
        filename = f'guidance{optim_args.guidance_param}_seed{optim_args.seed}'
        if optim_args.respacing != '':
            filename = f'{optim_args.respacing}_{filename}'
        if optim_args.zero_noise:
            filename = f'zero_noise_{filename}'
        if optim_args.use_predicted_joints:
            filename = f'use_pred_joints_{filename}'
        filename = f'{interaction_name}_{filename}'
        filename = f'{filename}_lr{optim_args.optim_lr}_coll{optim_args.weight_collision}_contact{optim_args.weight_contact}_jerk{optim_args.weight_jerk}'
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
        input_motions = input_motions.to(device)  # [B, D, 1, T]
        motion_tensor = input_motions.squeeze(2).permute(0, 2, 1)  # [B, T, D]
        init_history_motion = motion_tensor[:, :history_length, :]  # [B, H, D]

        for idx in range(optim_args.num_seq):
            all_motion_sequences = None
            for interaction_idx, interaction in enumerate(interaction_cfg['interactions']):
                cache_path = out_path / f'cache_{interaction_idx}.pkl'
                if cache_path.exists() and optim_args.load_cache:
                    with open(cache_path, 'rb') as f:
                        all_motion_sequences, history_motion_tensor, transf_rotmat, transf_transl = pickle.load(f)
                    tensor_dict_to_device(all_motion_sequences, device)
                    history_motion_tensor = history_motion_tensor.to(device)
                    transf_rotmat = transf_rotmat.to(device)
                    transf_transl = transf_transl.to(device)
                else:
                    text_prompt = interaction['text_prompt']
                    goal_joints = torch.zeros(batch_size, 22, 3, device=device, dtype=torch.float32)
                    goal_joints[:, 0] = torch.tensor(interaction['goal_loc'], device=device, dtype=torch.float32)
                    joints_mask = torch.zeros(22, device=device, dtype=torch.bool)
                    joints_mask[0] = 1

                    if interaction_idx == 0:
                        history_motion_tensor = init_history_motion
                        init_loc = torch.tensor(interaction['init_loc'], device=device,
                                                      dtype=torch.float32)  # [3]
                        transf_rotmat = torch.eye(3, device=device, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1, 1)
                        transf_transl = torch.zeros(3, device=device, dtype=torch.float32).reshape(1, 1, 3).repeat(batch_size,
                                                                                                                   1, 1)
                        motion_dict = primitive_utility.tensor_to_dict(dataset.denormalize(history_motion_tensor))
                        joints = motion_dict['joints'].reshape(batch_size, history_length, 22, 3)  # [B, T, 22, 3]
                        init_floor_height = joints[:, 0, :, 2].amin(dim=-1)  # [B]
                        transf_transl[:, :, 0] = init_loc[0]
                        transf_transl[:, :, 1] = init_loc[1]
                        transf_transl[:, :, 2] = -init_floor_height.unsqueeze(-1)
                        goal_joints[:, 0, 2] = -init_floor_height

                    motion_sequences, history_motion_tensor, transf_rotmat, transf_transl = optimize(
                        history_motion_tensor, transf_rotmat, transf_transl, text_prompt, goal_joints, joints_mask,
                        gender, betas, pelvis_delta, scene_assets
                    )

                    if all_motion_sequences is None:
                        all_motion_sequences = motion_sequences
                        all_motion_sequences['goal_location_list'] = [goal_joints[0, 0].cpu()]
                        num_frames = all_motion_sequences['joints'].shape[1]
                        all_motion_sequences['goal_location_idx'] = [0] * num_frames
                    else:
                        for key in motion_sequences:
                            if torch.is_tensor(motion_sequences[key]):
                                # print(key, all_motion_sequences[key].shape, motion_sequences[key].shape)
                                all_motion_sequences[key] = torch.cat([all_motion_sequences[key], motion_sequences[key]], dim=1)
                        all_motion_sequences['texts'] += motion_sequences['texts']
                        all_motion_sequences['goal_location_list'] += [goal_joints[0, 0].cpu()]
                        num_goals = len(all_motion_sequences['goal_location_list'])
                        num_frames = all_motion_sequences['joints'].shape[1]
                        all_motion_sequences['goal_location_idx'] += [num_goals - 1] * num_frames
                    with open(cache_path, 'wb') as f:
                        pickle.dump([all_motion_sequences, history_motion_tensor, transf_rotmat, transf_transl], f)


            sequence = {
                'texts': all_motion_sequences['texts'],
                'scene_path': scene_dir / 'mesh_floor_zup.ply',
                'goal_location_list': all_motion_sequences['goal_location_list'],
                'goal_location_idx': all_motion_sequences['goal_location_idx'],
                'gender': all_motion_sequences['gender'],
                'betas': all_motion_sequences['betas'][0],
                'transl': all_motion_sequences['transl'][0],
                'global_orient': all_motion_sequences['global_orient'][0],
                'body_pose': all_motion_sequences['body_pose'][0],
                'joints': all_motion_sequences['joints'][0],
                'history_length': history_length,
                'future_length': future_length,
                'sync_time': all_motion_sequences['sync_time'],
                'max_memory': all_motion_sequences['max_memory'],
                'loss_joints': all_motion_sequences['loss_joints'],
                'loss_jerk': all_motion_sequences['loss_jerk'],
                'loss_collision': all_motion_sequences['loss_collision'],
                'num_frames': all_motion_sequences['num_frames'],
            }
            print('sync_time:', all_motion_sequences['sync_time'], 'max_memory:', all_motion_sequences['max_memory'])
            print('loss_joints:', all_motion_sequences['loss_joints'], 'loss_jerk:', all_motion_sequences['loss_jerk'])
            print('loss_collision:', all_motion_sequences['loss_collision'])
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

            print(f'[Done] Results are at [{out_path.absolute()}]')

    """optimization config"""
    with open(optim_args.interaction_cfg, 'r') as f:
        interaction_cfg_list = json.load(f)
    for interaction_cfg in interaction_cfg_list:
        generate_scene(interaction_cfg)

