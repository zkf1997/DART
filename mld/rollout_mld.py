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

debug = 0

@dataclass
class RolloutArgs:
    seed: int = 0
    torch_deterministic: bool = True
    device: str = "cuda"
    batch_size: int = 4
    save_dir = None
    dataset: str = 'babel'

    denoiser_checkpoint: str = ''
    respacing: str = ''

    text_prompt: str = ''
    guidance_param: float = 1.0
    export_smpl: int = 0
    zero_noise: int = 0
    use_predicted_joints: int = 0
    fix_floor: int = 0


class ClassifierFreeWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model  # model is the actual model to run

        assert self.model.cond_mask_prob > 0, 'Cannot run a guided diffusion on a model that has not been trained with no conditions'

    def forward(self, x, timesteps, y=None):
        y['uncond'] = False
        out = self.model(x, timesteps, y)
        y_uncond = y
        y_uncond['uncond'] = True
        out_uncond = self.model(x, timesteps, y_uncond)
        # print('scale:', y['scale'])
        return out_uncond + (y['scale'] * (out - out_uncond))

def load_mld(denoiser_checkpoint, device):
    # load denoiser
    denoiser_dir = Path(denoiser_checkpoint).parent
    with open(denoiser_dir / "args.yaml", "r") as f:
        denoiser_args = tyro.extras.from_yaml(MLDArgs, yaml.safe_load(f)).denoiser_args
    # load mvae model and freeze
    print('denoiser model type:', denoiser_args.model_type)
    print('denoiser model args:', asdict(denoiser_args.model_args))
    denoiser_class = DenoiserMLP if isinstance(denoiser_args.model_args, DenoiserMLPArgs) else DenoiserTransformer
    denoiser_model = denoiser_class(
        **asdict(denoiser_args.model_args),
    ).to(device)
    checkpoint = torch.load(denoiser_checkpoint)
    model_state_dict = checkpoint['model_state_dict']
    print(f"Loading denoiser checkpoint from {denoiser_checkpoint}")
    denoiser_model.load_state_dict(model_state_dict)
    for param in denoiser_model.parameters():
        param.requires_grad = False
    denoiser_model.eval()
    denoiser_model = ClassifierFreeWrapper(denoiser_model)

    # load vae
    vae_checkpoint = denoiser_args.mvae_path
    vae_dir = Path(vae_checkpoint).parent
    with open(vae_dir / "args.yaml", "r") as f:
        vae_args = tyro.extras.from_yaml(MVAEArgs, yaml.safe_load(f))
    # load mvae model and freeze
    print('vae model args:', asdict(vae_args.model_args))
    vae_model = AutoMldVae(
        **asdict(vae_args.model_args),
    ).to(device)
    checkpoint = torch.load(denoiser_args.mvae_path)
    model_state_dict = checkpoint['model_state_dict']
    if 'latent_mean' not in model_state_dict:
        model_state_dict['latent_mean'] = torch.tensor(0)
    if 'latent_std' not in model_state_dict:
        model_state_dict['latent_std'] = torch.tensor(1)
    vae_model.load_state_dict(model_state_dict)
    vae_model.latent_mean = model_state_dict[
        'latent_mean']  # register buffer seems to be not loaded by load_state_dict
    vae_model.latent_std = model_state_dict['latent_std']
    print(f"Loading vae checkpoint from {denoiser_args.mvae_path}")
    print(f"latent_mean: {vae_model.latent_mean}")
    print(f"latent_std: {vae_model.latent_std}")
    for param in vae_model.parameters():
        param.requires_grad = False
    vae_model.eval()

    return denoiser_args, denoiser_model, vae_args, vae_model

def rollout(text_prompt, denoiser_args, denoiser_model, vae_args, vae_model, diffusion, dataset, rollout_args):
    device = rollout_args.device
    batch_size = rollout_args.batch_size
    future_length = dataset.future_length
    history_length = dataset.history_length
    primitive_length = history_length + future_length
    sample_fn = diffusion.p_sample_loop if rollout_args.respacing == '' else diffusion.ddim_sample_loop

    texts = []
    if rollout_args.dataset == 'babel':
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
    else:
        if ';' in text_prompt:  # contain a time line of multipel actions
            num_rollout = 0
            for segment in text_prompt.split(';'):
                action, num_mp = segment.split('*')
                texts = texts + [action] * int(num_mp)
                num_rollout += int(num_mp)
        else:
            action, num_rollout = text_prompt.split('*')
            num_rollout = int(num_rollout)
            for _ in range(num_rollout):
                texts.append(action)
    all_text_embedding = encode_text(dataset.clip_model, texts, force_empty_zero=True).to(dtype=torch.float32,
                                                                                      device=device)
    primitive_utility = dataset.primitive_utility
    print('body_type:', primitive_utility.body_type)

    out_path = rollout_args.save_dir
    filename = f'guidance{rollout_args.guidance_param}_seed{rollout_args.seed}'
    if text_prompt != '':
        filename = text_prompt[:40].replace(' ', '_').replace('.', '') + '_' + filename
    if rollout_args.respacing != '':
        filename = f'{rollout_args.respacing}_{filename}'
    if rollout_args.zero_noise:
        filename = f'zero_noise_{filename}'
    if rollout_args.use_predicted_joints:
        filename = f'use_pred_joints_{filename}'
    if rollout_args.fix_floor:
        filename = f'fixfloor_{filename}'
    out_path = out_path / filename
    out_path.mkdir(parents=True, exist_ok=True)

    batch = dataset.get_batch(batch_size=rollout_args.batch_size)
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
    history_motion_gt = motion_tensor[:, :history_length, :]  # [B, H, D]
    if text_prompt == '':
        rollout_args.guidance_param = 0.  # Force unconditioned generation

    motion_sequences = None
    history_motion = history_motion_gt
    transf_rotmat = torch.eye(3, device=device, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1, 1)
    transf_transl = torch.zeros(3, device=device, dtype=torch.float32).reshape(1, 1, 3).repeat(batch_size, 1, 1)
    if rollout_args.fix_floor:
        motion_dict = primitive_utility.tensor_to_dict(dataset.denormalize(history_motion_gt))
        joints = motion_dict['joints'].reshape(batch_size, history_length, 22, 3)  # [B, T, 22, 3]
        init_floor_height = joints[:, 0, :, 2].amin(dim=-1)  # [B]
        transf_transl[:, :, 2] = -init_floor_height.unsqueeze(-1)

    for segment_id in tqdm(range(num_rollout)):
        text_embedding = all_text_embedding[segment_id].expand(batch_size, -1)  # [B, 512]
        guidance_param = torch.ones(batch_size, *denoiser_args.model_args.noise_shape).to(device=device) * rollout_args.guidance_param
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
        latent_pred = x_start_pred.permute(1, 0, 2)  # [T=1, B, D]
        future_motion_pred = vae_model.decode(latent_pred, history_motion, nfuture=future_length,
                                                   scale_latent=denoiser_args.rescale_latent)  # [B, F, D], normalized

        future_frames = dataset.denormalize(future_motion_pred)
        all_frames = torch.cat([dataset.denormalize(history_motion), future_frames], dim=1)

        """transform primitive to world coordinate, prepare for serialization"""
        if segment_id == 0:  # add init history motion
            future_frames = all_frames
        if rollout_args.fix_floor:
            future_feature_dict = primitive_utility.tensor_to_dict(future_frames)
            joints = future_feature_dict['joints'].reshape(batch_size, -1, 22, 3)  # [B, T, 22, 3]
            joints = torch.einsum('bij,btkj->btki', transf_rotmat, joints) + transf_transl.unsqueeze(1)
            min_height = joints[:, :, :, 2].amin(dim=-1)  # [B, T]
            transl_floor = torch.zeros(batch_size, joints.shape[1], 3, device=device, dtype=torch.float32)  # [B, T, 3]
            transl_floor[:, :, 2] = - min_height
            future_feature_dict['transl'] += transl_floor
            transl_delta_local = torch.einsum('bij,bti->btj', transf_rotmat, transl_floor)
            joints += transl_delta_local.unsqueeze(2)
            future_feature_dict['joints'] = joints.reshape(batch_size, -1, 66)
            future_frames = primitive_utility.dict_to_tensor(future_feature_dict)
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
        new_history_frames = all_frames[:, -history_length:, :]
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
            history_feature_dict, use_predicted_joints=rollout_args.use_predicted_joints)
        transf_rotmat, transf_transl = canonicalized_history_primitive_dict['transf_rotmat'], \
        canonicalized_history_primitive_dict['transf_transl']
        history_motion = primitive_utility.dict_to_tensor(blended_feature_dict)
        history_motion = dataset.normalize(history_motion)  # [B, T, D]

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    for idx in range(rollout_args.batch_size):
        sequence = {
            'texts': texts,
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
        if rollout_args.export_smpl:
            poses = transforms.matrix_to_axis_angle(
                torch.cat([sequence['global_orient'].reshape(-1, 1, 3, 3), sequence['body_pose']], dim=1)
            ).reshape(-1, 22 * 3)
            poses = torch.cat([poses, torch.zeros(poses.shape[0], 99).to(dtype=poses.dtype, device=poses.device)],
                              dim=1)
            data_dict = {
                'mocap_framerate': min(dataset.target_fps, 30),  # has to be greater than 30 to be loaded by blender
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
    rollout_args = tyro.cli(RolloutArgs)
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

    # load initial seed dataset
    dataset = SinglePrimitiveDataset(cfg_path=vae_args.data_args.cfg_path,  # cfg path from model checkpoint
                                     dataset_path=vae_args.data_args.data_dir,  # dataset path from model checkpoint
                                     body_type=vae_args.data_args.body_type,
                                     sequence_path=f'./data/stand.pkl' if rollout_args.dataset == 'babel' else f'./data/stand_20fps.pkl',
                                     batch_size=rollout_args.batch_size,
                                     device=device,
                                     enforce_gender='male',
                                     enforce_zero_beta=1,
                                     )


    if Path(rollout_args.text_prompt).exists():
        with open(rollout_args.text_prompt, 'r') as f:
            texts = f.readlines()
            texts = [text.strip() for text in texts]
            for text_prompt in texts:
                print(f'Generating [{text_prompt}]')
                rollout(text_prompt, denoiser_args, denoiser_model, vae_args, vae_model, diffusion, dataset, rollout_args)
    else:
        rollout(rollout_args.text_prompt, denoiser_args, denoiser_model, vae_args, vae_model, diffusion, dataset, rollout_args)


