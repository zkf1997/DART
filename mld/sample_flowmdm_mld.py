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
class FlowMDMArgs:
    seed: int = 0
    torch_deterministic: bool = True
    device: str = "cuda"
    batch_size: int = 0  # to be filled
    flowmdm_dir: str = ''

    denoiser_checkpoint: str = ''
    respacing: str = ''

    guidance_param: float = 1.0
    export_smpl: int = 0
    zero_noise: int = 0
    use_predicted_joints: int = 0

    fix_floor: int = 0

def rollout(config, denoiser_args, denoiser_model, vae_args, vae_model, diffusion, dataset, rollout_args):
    device = rollout_args.device
    batch_size = rollout_args.batch_size
    future_length = dataset.future_length
    history_length = dataset.history_length
    primitive_length = history_length + future_length
    sample_fn = diffusion.p_sample_loop if rollout_args.respacing == '' else diffusion.ddim_sample_loop

    texts = config['text']
    lengths = config['lengths']
    all_text_embedding = encode_text(dataset.clip_model, texts, force_empty_zero=True).to(dtype=torch.float32,
                                                                                      device=device)
    primitive_utility = PrimitiveUtility(device=device, dtype=torch.float32)

    batch = dataset.get_batch(batch_size=rollout_args.batch_size)
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

    motion_sequences = None
    history_motion = history_motion_gt
    # transform to similar axis to flowmdm
    flowmdm_joints = torch.tensor([[0.0012, -0.3668, 0.9377],
                                   [0.0010, -0.4273, 0.8429],
                                   [-0.0034, -0.3135, 0.8290]], device=device, dtype=torch.float32).reshape(1, 3, 3)
    transf_rotmat, transf_transl = get_new_coordinate(flowmdm_joints)
    transf_rotmat = transf_rotmat.repeat(batch_size, 1, 1)
    transf_transl = transf_transl.repeat(batch_size, 1, 1)

    for segment_id in range(len(lengths)):
        segment_length = lengths[segment_id]
        text_embedding = all_text_embedding[segment_id].expand(batch_size, -1)  # [B, 512]
        num_primitives = int(np.ceil(segment_length / future_length))
        guidance_param = torch.ones(batch_size, *denoiser_args.model_args.noise_shape).to(device=device) * rollout_args.guidance_param

        for primitive_id in range(num_primitives):
            valid_length = min(future_length, segment_length - primitive_id * future_length)
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
                dump_steps=None,
                noise=torch.zeros_like(guidance_param) if rollout_args.zero_noise else None,
                const_noise=False,
            )  # [B, T=1, D]
            # print('x_start_pred:', x_start_pred.mean(), x_start_pred.std())
            latent_pred = x_start_pred.permute(1, 0, 2)  # [T=1, B, D]
            future_motion_pred = vae_model.decode(latent_pred, history_motion, nfuture=future_length,
                                                       scale_latent=denoiser_args.rescale_latent)  # [B, F, D], normalized

            future_frames = dataset.denormalize(future_motion_pred)
            all_frames = torch.cat([dataset.denormalize(history_motion), future_frames], dim=1)  # [B, H+F, D]
            future_start, future_end = 0, valid_length
            # print(f'primitive_id: {primitive_id}, future_start: {future_start}, future_end: {future_end}')
            valid_future_frames = future_frames[:, future_start:future_end, :]  # ignore the initial standing seed
            new_history_end = history_length + valid_length
            new_history_start = new_history_end - history_length
            # print(f'primitive_id: {primitive_id}, new_history_start: {new_history_start}, new_history_end: {new_history_end}')
            new_history_frames = all_frames[:, new_history_start:new_history_end, :]

            """transform primitive to world coordinate, prepare for serialization"""
            future_feature_dict = primitive_utility.tensor_to_dict(valid_future_frames)
            future_feature_dict.update(
                {
                    'transf_rotmat': transf_rotmat,
                    'transf_transl': transf_transl,
                    'gender': gender,
                    'betas': betas[:, future_start:future_end, :],
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
                    'betas': betas[:, new_history_start:new_history_end, :],
                    'pelvis_delta': pelvis_delta,
                }
            )
            if rollout_args.fix_floor and primitive_id == num_primitives - 1:  # fix the first frame feet of each segment to be on floor
                foot_height = history_feature_dict['joints'].reshape(-1, history_length, 22, 3)[:, 0, FOOT_JOINTS_IDX, 2].amin(dim=-1)  # [B]
                foot_height_world = foot_height + history_feature_dict['transf_transl'][:, 0, 2]  # [B]
                history_feature_dict['transf_transl'][:, 0, 2] -= foot_height_world
            canonicalized_history_primitive_dict, blended_feature_dict = primitive_utility.get_blended_feature(
                history_feature_dict, use_predicted_joints=rollout_args.use_predicted_joints)
            transf_rotmat, transf_transl = canonicalized_history_primitive_dict['transf_rotmat'], \
            canonicalized_history_primitive_dict['transf_transl']
            history_motion = primitive_utility.dict_to_tensor(blended_feature_dict)
            history_motion = dataset.normalize(history_motion)  # [B, T, D]

    motion_sequences['texts'] = texts
    motion_sequences['text_idx'] = []  # idx of corresponding text of each frame
    for seg_idx, seg_len in enumerate(lengths):
        motion_sequences['text_idx'] += [seg_idx] * seg_len
    return motion_sequences

if __name__ == '__main__':
    rollout_args = tyro.cli(FlowMDMArgs)
    # TRY NOT TO MODIFY: seeding
    random.seed(rollout_args.seed)
    np.random.seed(rollout_args.seed)
    torch.manual_seed(rollout_args.seed)
    torch.set_default_dtype(torch.float32)
    torch.backends.cudnn.deterministic = rollout_args.torch_deterministic
    device = torch.device(rollout_args.device if torch.cuda.is_available() else "cpu")
    rollout_args.device = device

    denoiser_args, denoiser_model, vae_args, vae_model = load_mld(rollout_args.denoiser_checkpoint, device)
    denoiser_checkpoint = Path(rollout_args.denoiser_checkpoint)
    save_dir = denoiser_checkpoint.parent / denoiser_checkpoint.name.split('.')[0] / 'rollout'
    save_dir.mkdir(parents=True, exist_ok=True)
    rollout_args.save_dir = save_dir

    diffusion_args = denoiser_args.diffusion_args
    diffusion_args.respacing = rollout_args.respacing
    print('diffusion_args:', asdict(diffusion_args))
    diffusion = create_gaussian_diffusion(diffusion_args)

    flowmdm_dir = Path(rollout_args.flowmdm_dir)
    print(f'Loading flowmdm results from {flowmdm_dir}')
    ours_dir = flowmdm_dir.parents[2] / flowmdm_dir.name
    filename = f'guidance{rollout_args.guidance_param}_seed{rollout_args.seed}'
    if rollout_args.respacing != '':
        filename = f'{rollout_args.respacing}_{filename}'
    if rollout_args.use_predicted_joints:
        filename = f'predicted_joints_{filename}'
    if rollout_args.zero_noise:
        filename = f'zero_noise_{filename}'
    if 'extrapolation' in flowmdm_dir.name:
        filename = f'extrapolation_{filename}'
    model_path = Path(rollout_args.denoiser_checkpoint)
    ours_dir = ours_dir / '_'.join([model_path.parent.name, model_path.stem, filename]) / 'evaluation_precomputed'
    print(f'Saving to {ours_dir}')
    num_replicate = len(list(flowmdm_dir.iterdir()))
    rollout_args.batch_size = num_replicate  # flowmdm gene

    # load initial seed dataset
    dataset = SinglePrimitiveDataset(cfg_path=vae_args.data_args.cfg_path,  # cfg path from model checkpoint
                                     dataset_path=vae_args.data_args.data_dir,  # dataset path from model checkpoint
                                     sequence_path=f'./data/stand.pkl',
                                     # sequence_path=f'./data/jump forward.pkl',
                                     batch_size=rollout_args.batch_size,
                                     device=device,
                                     enforce_gender='male',
                                     enforce_zero_beta=1,
                                     )

    for config_file in tqdm(flowmdm_dir.glob('00/*_kwargs.json')):
        config_idx = config_file.name.split('_')[0]
        with open(config_file, 'r') as f:
            config = json.load(f)

        t1 = time.time()
        motion_sequences = rollout(config, denoiser_args, denoiser_model, vae_args, vae_model, diffusion, dataset, rollout_args)

        for replicate_dir in flowmdm_dir.iterdir():
            if replicate_dir.is_dir():
                replicate_idx = replicate_dir.name
                output_dir = ours_dir / replicate_idx
                output_dir.mkdir(parents=True, exist_ok=True)
                npy_path = output_dir / f'{config_idx}.npy'

                if not (output_dir / config_file.name).exists():
                    os.symlink(config_file, output_dir / config_file.name)

                # save sequence
                idx = int(replicate_idx)
                sequence = {
                    'texts': motion_sequences['texts'],
                    'text_idx': motion_sequences['text_idx'],
                    'gender': motion_sequences['gender'],
                    'betas': motion_sequences['betas'][idx],
                    'transl': motion_sequences['transl'][idx],
                    'global_orient': motion_sequences['global_orient'][idx],
                    'body_pose': motion_sequences['body_pose'][idx],
                    'joints': motion_sequences['joints'][idx],
                    'history_length': vae_args.data_args.history_length,
                    'future_length': vae_args.data_args.future_length,
                }
                tensor_dict_to_device(sequence, 'cpu')
                with open(output_dir / f'{config_idx}.pkl', 'wb') as f:
                    pickle.dump(sequence, f)

                # save npy for evaluation
                transl = motion_sequences['transl'][idx] # [T, 3]
                poses = torch.cat([motion_sequences['global_orient'][idx].reshape(-1, 1, 3, 3),
                                   motion_sequences['body_pose'][idx]], dim=1)  # [T, 22, 3, 3]
                with open(npy_path, 'wb') as f:
                    np.save(f, {'transl': transl.detach().cpu().numpy(), 'rots': poses.detach().cpu().numpy()})

                # export smplx sequences for blender
                if rollout_args.export_smpl:
                    poses = transforms.matrix_to_axis_angle(poses).reshape(-1, 22 * 3)
                    poses = torch.cat([poses, torch.zeros(poses.shape[0], 99).to(dtype=poses.dtype, device=poses.device)], dim=1)
                    data_dict = {
                        'mocap_framerate': 30,  # 30
                        'gender': 'male',
                        'betas': np.zeros((16, )),
                        'poses': poses.detach().cpu().numpy(),
                        'trans': transl.detach().cpu().numpy(),
                    }
                    smpl_path = output_dir / f'{config_idx}_smpl.npz'
                    with open(smpl_path, 'wb') as f:
                        np.savez(f, **data_dict)


