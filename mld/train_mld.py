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
from torch.cuda import amp
import tyro
import yaml
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import pickle
import json
import copy

from mld.train_mvae import Args as MVAEArgs
from mld.train_mvae import DataArgs, TrainArgs
from model.mld_denoiser import DenoiserMLP, DenoiserTransformer
from model.mld_vae import AutoMldVae
from data_loaders.humanml.data.dataset import PrimitiveSequenceDataset, WeightedPrimitiveSequenceDataset, WeightedPrimitiveSequenceDatasetV2
from data_loaders.humanml.data.dataset_hml3d import HML3dDataset
from utils.smpl_utils import get_smplx_param_from_6d
from pytorch3d import transforms
from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps
from diffusion.resample import create_named_schedule_sampler

debug = 0

@dataclass
class DiffusionArgs:
    diffusion_steps: int = 10
    respacing: str = ''
    noise_schedule: Literal['linear', 'cosine'] = 'cosine'
    sigma_small: bool = True


@dataclass
class DenoiserMLPArgs:
    h_dim: int = 512
    n_blocks: int = 2
    dropout: float = 0.1
    activation: str = "gelu"

    cond_mask_prob: float = 0.1

    clip_dim: int = 512
    history_shape: tuple = (2, 276)
    noise_shape: tuple = (1, 128)


@dataclass
class DenoiserTransformerArgs:
    h_dim: int = 512
    ff_size: int = 1024
    num_layers: int = 8
    num_heads: int = 4
    dropout: float = 0.1
    activation: str = "gelu"

    cond_mask_prob: float = 0.1

    clip_dim: int = 512
    history_shape: tuple = (2, 276)
    noise_shape: tuple = (1, 128)


@dataclass
class DenoiserArgs:
    mvae_path: str = ''
    rescale_latent: int = 1

    train_rollout_type: Literal["single", "full"] = "single"
    train_rollout_history: str = "gt"  # "rollout" or "gt"

    model_type: str = "mlp"
    model_args: DenoiserMLPArgs | DenoiserTransformerArgs = DenoiserMLPArgs()

    diffusion_args: DiffusionArgs = DiffusionArgs()


@dataclass
class MLDArgs:
    train_args: TrainArgs = TrainArgs()
    data_args: DataArgs = DataArgs()
    denoiser_args: DenoiserArgs = DenoiserArgs()

    exp_name: str = "mld"
    seed: int = 0
    torch_deterministic: bool = True
    device: str = "cuda"
    save_dir: str = "./mld_denoiser"

    track: int = 1
    wandb_project_name: str = "mld_denoiser"
    wandb_entity: str = "interaction"


def create_gaussian_diffusion(args, enable_ddim=True):
    # default params
    predict_xstart = True  # we always predict x_start (a.k.a. x0), that's our deal!
    steps = args.diffusion_steps
    scale_beta = 1.  # no scaling
    timestep_respacing = args.respacing if enable_ddim else ''  # can be used for ddim sampling, we don't use it.
    learn_sigma = False
    rescale_timesteps = False

    betas = gd.get_named_beta_schedule(args.noise_schedule, steps, scale_beta)
    loss_type = gd.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not args.sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )


class Trainer:
    def __init__(self, args):
        self.args = args
        args.save_dir = Path(args.save_dir) / args.exp_name
        args.save_dir.mkdir(parents=True, exist_ok=True)
        train_args = args.train_args
        data_args = args.data_args
        denoiser_args = args.denoiser_args

        # TRY NOT TO MODIFY: seeding
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.set_default_dtype(torch.float32)
        torch.backends.cudnn.deterministic = args.torch_deterministic
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")

        # load dataset
        if data_args.dataset == 'mp_seq_v2':
            dataset_class = WeightedPrimitiveSequenceDatasetV2
        elif data_args.dataset == 'hml3d':
            dataset_class = HML3dDataset
        else:
            dataset_class = WeightedPrimitiveSequenceDataset
        train_dataset = dataset_class(dataset_path=data_args.data_dir,
                                      dataset_name=data_args.dataset,
                                      cfg_path=data_args.cfg_path, prob_static=data_args.prob_static,
                                      enforce_gender=data_args.enforce_gender,
                                      enforce_zero_beta=data_args.enforce_zero_beta,
                                      body_type=data_args.body_type,
                                      split='train', device=device,
                                      weight_scheme=data_args.weight_scheme,
                                      text_tolerance=data_args.text_tolerance,
                                      )
        val_dataset = train_dataset
        # if 'text' in data_args.weight_scheme or 'samp:1' in data_args.weight_scheme:
        #     val_dataset = train_dataset
        # else:
        #     val_dataset = dataset_class(dataset_path=data_args.data_dir, dataset_name=data_args.dataset,
        #                                                    cfg_path=data_args.cfg_path, prob_static=data_args.prob_static,
        #                                                    enforce_gender=data_args.enforce_gender,
        #                                                    enforce_zero_beta=data_args.enforce_zero_beta,
        #                                                    split='val', device=device,
        #                                                    weight_scheme=data_args.weight_scheme,
        #                                                    text_tolerance=data_args.text_tolerance,
        #                                                    )
        # get primitive configs
        data_args.history_length = train_dataset.history_length
        data_args.future_length = train_dataset.future_length
        data_args.num_primitive = train_dataset.num_primitive
        data_args.feature_dim = 0
        for k in train_dataset.motion_repr:
            data_args.feature_dim += train_dataset.motion_repr[k]

        mvae_checkpoint_dir = Path(denoiser_args.mvae_path).parent
        arg_path = mvae_checkpoint_dir / "args.yaml"
        with open(arg_path, "r") as f:
            mvae_args = tyro.extras.from_yaml(MVAEArgs, yaml.safe_load(f))

        denoiser_model_args = args.denoiser_args.model_args
        assert mvae_args.data_args.history_length == data_args.history_length
        assert mvae_args.data_args.future_length == data_args.future_length
        assert mvae_args.data_args.feature_dim == data_args.feature_dim
        denoiser_model_args.history_shape = (data_args.history_length, data_args.feature_dim)
        denoiser_model_args.noise_shape = mvae_args.model_args.latent_dim

        run_name = f"{args.exp_name}__seed{args.seed}__{int(time.time())}"
        if args.track:
            import wandb

            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=True,
                config=vars(args),
                name=run_name,
                save_code=True,
                # settings=wandb.Settings(code_dir="./mld"),
            )
            def include_fn(path, root):
                rel_path = os.path.relpath(path, root)
                flag = (rel_path.startswith("mld/") and len(Path(rel_path).parents) <= 2) or rel_path.startswith("model/")
                return flag
            wandb.run.log_code(root=".",
                               include_fn=include_fn
                               )
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )

        # load mvae model and freeze
        print('vae model args:', asdict(mvae_args.model_args))
        vae_model = AutoMldVae(
            **asdict(mvae_args.model_args),
        ).to(device)
        checkpoint = torch.load(denoiser_args.mvae_path)
        model_state_dict = checkpoint['model_state_dict']
        if 'latent_mean' not in model_state_dict:
            model_state_dict['latent_mean'] = torch.tensor(0)
        if 'latent_std' not in model_state_dict:
            model_state_dict['latent_std'] = torch.tensor(1)
        vae_model.load_state_dict(model_state_dict)
        vae_model.latent_mean = model_state_dict['latent_mean']  # register buffer seems to be not loaded by load_state_dict
        vae_model.latent_std = model_state_dict['latent_std']
        print(f"Loading vae checkpoint from {denoiser_args.mvae_path}")
        print(f"latent_mean: {vae_model.latent_mean}")
        print(f"latent_std: {vae_model.latent_std}")
        for param in vae_model.parameters():
            param.requires_grad = False
        vae_model.eval()

        denoiser_class = DenoiserMLP if isinstance(denoiser_model_args, DenoiserMLPArgs) else DenoiserTransformer
        denoiser_args.model_type = "mlp" if isinstance(denoiser_model_args, DenoiserMLPArgs) else "transformer"
        denoiser_model = denoiser_class(
            **asdict(denoiser_model_args),
        ).to(device)
        print('denoiser model type:', denoiser_args.model_type)
        print('denoiser model args:', asdict(denoiser_model_args))
        optimizer = optim.AdamW(denoiser_model.parameters(), lr=train_args.learning_rate)
        start_step = 1
        if args.train_args.resume_checkpoint is not None:
            checkpoint = torch.load(args.train_args.resume_checkpoint)
            model_state_dict = checkpoint['model_state_dict']
            denoiser_model.load_state_dict(model_state_dict)
            # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_step = checkpoint['num_steps'] + 1
            print(f"Loading checkpoint from {args.train_args.resume_checkpoint} at step {start_step}")
        self.denoiser_model_avg = None
        if train_args.ema_decay > 0:
            self.denoiser_model_avg = copy.deepcopy(denoiser_model)
            self.denoiser_model_avg.eval()

        with open(args.save_dir / "args.yaml", "w") as f:
            yaml.dump(tyro.extras.to_yaml(args), f)
        with open(args.save_dir / "args_read.yaml", "w") as f:
            yaml.dump(asdict(args), f)

        self.diffusion = create_gaussian_diffusion(args.denoiser_args.diffusion_args, enable_ddim=False)
        self.schedule_sampler_type = 'uniform'
        self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_type, self.diffusion)

        self.vae_model = vae_model
        self.denoiser_model = denoiser_model
        self.optimizer = optimizer
        self.writer = writer
        self.start_step = start_step
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = device
        self.batch_size = train_args.batch_size
        self.step = start_step

        self.rec_criterion = torch.nn.HuberLoss(reduction='mean', delta=1.0)
        self.transf_rotmat = torch.eye(3, device=self.device).unsqueeze(0)
        self.transf_transl = torch.zeros(3, device=self.device).reshape(1, 1, 3)

    def calc_loss(self, motion, cond, history_motion, future_motion_gt, future_motion_pred, latent_gt, latent_pred, weights):
        train_args = self.args.train_args
        model_kwargs = cond
        future_length = self.train_dataset.future_length
        history_length = self.train_dataset.history_length
        num_primitive = self.train_dataset.num_primitive

        terms = {}
        # feature reconstruction loss
        feature_rec_loss = self.rec_criterion(future_motion_pred, future_motion_gt)
        terms['feature_rec'] = feature_rec_loss

        # latent rec loss
        latent_rec_loss = self.rec_criterion(latent_pred, latent_gt)
        terms['latent_rec'] = latent_rec_loss

        """smplx consistency losses"""
        dataset = self.train_dataset
        primitive_utility = dataset.primitive_utility
        if train_args.weight_joints_consistency > 0 or train_args.weight_smpl_joints_rec > 0:
            gt_motion_tensor = future_motion_gt
            pred_motion_tensor = future_motion_pred
            genders = model_kwargs['y']['gender']
            betas = model_kwargs['y']['betas']
            def get_smpl_body(motion_tensor, genders, betas):
                batch_size, num_frames, _ = motion_tensor.shape
                device = motion_tensor.device
                smpl_joints = []
                smpl_vertices = []
                joints = []
                for gender_name in ['female', 'male']:
                    # body_model = body_model_male if gender_name == 'male' else body_model_female
                    body_model = primitive_utility.get_smpl_model(gender=gender_name)
                    gender_idx = [idx for idx in range(len(genders)) if genders[idx] == gender_name]
                    sub_batch_size = len(gender_idx)
                    if len(gender_idx) == 0:
                        continue
                    # gender_betas = betas[gender_idx].unsqueeze(1).repeat(1, num_frames, 1).reshape(
                    #     sub_batch_size * num_frames, -1)
                    gender_betas = betas[gender_idx, history_length:, :].reshape(
                        sub_batch_size * num_frames, 10)
                    gender_motion_tensor = motion_tensor[gender_idx, :, :]
                    gender_motion_tensor = dataset.denormalize(gender_motion_tensor).reshape(
                        sub_batch_size * num_frames, -1)

                    motion_dict = dataset.tensor_to_dict(gender_motion_tensor)
                    motion_dict.update({'betas': gender_betas})
                    joints.append(motion_dict['joints'].reshape(sub_batch_size, num_frames, 22, 3))
                    smplx_param = get_smplx_param_from_6d(motion_dict)
                    smplxout = body_model(return_verts=False, **smplx_param)
                    smpl_joints.append(smplxout.joints[:, :22, :].reshape(sub_batch_size, num_frames, 22,
                                                                          3))  # [bs, nframes, 22, 3]
                    # smpl_vertices.append(
                    #     smplxout.vertices.reshape(sub_batch_size, num_frames, -1, 3))  # [bs, nframes, V, 3]

                smpl_joints = torch.cat(smpl_joints, dim=0)
                # smpl_vertices = torch.cat(smpl_vertices, dim=0)
                smpl_vertices = None
                joints = torch.cat(joints, dim=0)
                return {'smpl_joints': smpl_joints, 'smpl_vertices': smpl_vertices, 'joints': joints}

            with torch.no_grad():
                gt_result_dict = get_smpl_body(gt_motion_tensor, genders,
                                               betas)  # note that each batch is reordered according to gender. we assume the input batch is already sorted by gender, so the actual order does not change after this operation
            pred_result_dict = get_smpl_body(pred_motion_tensor, genders, betas)
            terms['smpl_joints_rec'] = self.rec_criterion(pred_result_dict['smpl_joints'], gt_result_dict['smpl_joints'])
            terms['joints_consistency'] = self.rec_criterion(pred_result_dict['joints'], pred_result_dict['smpl_joints'])
            # terms['smpl_vertices_rec'] = torch.zeros_like(terms['smpl_joints_rec'])
        else:
            terms['smpl_joints_rec'] = torch.zeros(1, device=self.device)
            terms['joints_consistency'] = torch.zeros(1, device=self.device)

        """temporal delta loss"""
        pred_motion_tensor = torch.cat([history_motion[:, -1:, :], future_motion_pred], dim=1)  # [B, F+1, D]
        pred_motion_tensor = dataset.denormalize(pred_motion_tensor)
        pred_feature_dict = dataset.tensor_to_dict(pred_motion_tensor)
        pred_joints_delta = pred_feature_dict['joints_delta'][:, :-1, :]
        pred_transl_delta = pred_feature_dict['transl_delta'][:, :-1, :]
        pred_orient_delta = pred_feature_dict['global_orient_delta_6d'][:, :-1, :]
        calc_joints_delta = pred_feature_dict['joints'][:, 1:, :] - pred_feature_dict['joints'][:, :-1, :]
        calc_transl_delta = pred_feature_dict['transl'][:, 1:, :] - pred_feature_dict['transl'][:, :-1, :]
        pred_orient = transforms.rotation_6d_to_matrix(pred_feature_dict['poses_6d'][:, :, :6])  # [B, T, 3, 3]
        calc_orient_delta_matrix = torch.matmul(pred_orient[:, 1:],
                                                pred_orient[:, :-1].permute(0, 1, 3, 2))
        calc_orient_delta_6d = transforms.matrix_to_rotation_6d(calc_orient_delta_matrix)
        terms["joints_delta"] = self.rec_criterion(calc_joints_delta, pred_joints_delta)
        terms["transl_delta"] = self.rec_criterion(calc_transl_delta, pred_transl_delta)
        terms["orient_delta"] = self.rec_criterion(calc_orient_delta_6d, pred_orient_delta)

        loss = train_args.weight_latent_rec * latent_rec_loss + train_args.weight_feature_rec * feature_rec_loss + \
               train_args.weight_smpl_joints_rec * terms['smpl_joints_rec'] + \
               train_args.weight_joints_consistency * terms['joints_consistency'] + \
               train_args.weight_joints_delta * terms["joints_delta"] + \
               train_args.weight_transl_delta * terms["transl_delta"] + \
               train_args.weight_orient_delta * terms["orient_delta"]
        terms['loss'] = loss
        # for key in terms:
        #     print(f"{key}: {terms[key].item()}")
        return terms

    def common_step(self, motion, cond, last_primitive):
        denoiser_args = self.args.denoiser_args
        train_args = self.args.train_args
        future_length = self.train_dataset.future_length
        history_length = self.train_dataset.history_length
        num_primitive = self.train_dataset.num_primitive

        motion_tensor = motion.squeeze(2).permute(0, 2, 1)  # [B, T, D]
        future_motion_gt = motion_tensor[:, -future_length:, :]
        history_motion_gt = motion_tensor[:, :history_length, :]
        if last_primitive is not None:
            rollout_history = self.get_rollout_history(last_primitive, cond)
            history_motion = rollout_history  # [B, H, D]
        else:
            history_motion = history_motion_gt
        latent_gt, _ = self.vae_model.encode(future_motion=future_motion_gt,
                                             history_motion=history_motion_gt if denoiser_args.train_rollout_history == "gt" else history_motion,
                                             scale_latent=denoiser_args.rescale_latent)  # [T=1, B, D]
        # print('latent_gt:', latent_gt)
        # pdb.set_trace()

        t, weights = self.schedule_sampler.sample(self.batch_size, device=self.device)  # weights always 1
        # print('t:', t, 'weights:', weights)

        # forward diffusion
        x_start = latent_gt.permute(1, 0, 2)  # [B, T=1, D]
        x_t = self.diffusion.q_sample(x_start=x_start, t=t, noise=torch.randn_like(x_start))
        # denoise
        y = {
            'text_embedding': cond['y']['text_embedding'],
            'history_motion_normalized': history_motion,
        }
        x_start_pred = self.denoiser_model(x_t=x_t, timesteps=self.diffusion._scale_timesteps(t), y=y)  # [B, T=1, D]
        latent_pred = x_start_pred.permute(1, 0, 2)  # [T=1, B, D]

        future_motion_pred = self.vae_model.decode(latent_pred, history_motion, nfuture=future_length,
                                                   scale_latent=denoiser_args.rescale_latent)  # [B, F, D], normalized

        loss_dict = self.calc_loss(motion, cond, history_motion, future_motion_gt, future_motion_pred, latent_gt, latent_pred, weights)

        if self.step > train_args.stage1_steps and self.args.denoiser_args.train_rollout_type == "full":  # sample with full ddpm loop to get rollout history
            sample_fn = self.diffusion.p_sample_loop
            with torch.no_grad():
                with amp.autocast(enabled=bool(train_args.use_amp), dtype=torch.float16):
                    x_start_pred = sample_fn(
                        self.denoiser_model,
                        x_start.shape,
                        clip_denoised=False,
                        model_kwargs={'y': y},
                        skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                        init_image=x_start,
                        progress=False,
                        dump_steps=None,
                        noise=None,
                        const_noise=False,
                    )
                    latent_pred = x_start_pred.permute(1, 0, 2)  # [T=1, B, D]
                    # if torch.isnan(latent_pred).any() or torch.isinf(latent_pred).any():
                    #     print('latent_pred numerical error')
                    #     pdb.set_trace()
                    future_motion_pred = self.vae_model.decode(latent_pred, history_motion, nfuture=future_length,
                                                               scale_latent=denoiser_args.rescale_latent)  # [B, F, D], normalized

        return loss_dict, future_motion_pred

    def train(self):
        denoiser_model = self.denoiser_model
        optimizer = self.optimizer
        train_args = self.args.train_args
        writer = self.writer
        future_length = self.train_dataset.future_length
        history_length = self.train_dataset.history_length
        num_primitive = self.train_dataset.num_primitive

        denoiser_model.train()
        total_steps = train_args.stage1_steps + train_args.stage2_steps + train_args.stage3_steps
        rest_steps = (total_steps - self.start_step) // self.train_dataset.num_primitive + 1
        rest_steps = rest_steps * self.train_dataset.num_primitive
        progress_bar = iter(tqdm(range(rest_steps)))
        self.step = self.start_step

        # self.validate()

        while self.step <= total_steps:
            # Annealing the rate if instructed to do so.
            if train_args.anneal_lr:
                frac = 1.0 - (self.step - 1.0) / total_steps
                lrnow = frac * train_args.learning_rate
                optimizer.param_groups[0]["lr"] = lrnow

            # t1 = time.time()
            # print('amp:', bool(train_args.use_amp))
            with amp.autocast(enabled=bool(train_args.use_amp), dtype=torch.float16):
                batch = self.train_dataset.get_batch(self.batch_size)
            # t2 = time.time()
            last_primitive = None
            for primitive_idx in range(num_primitive):
                with amp.autocast(enabled=bool(train_args.use_amp), dtype=torch.float16):
                    motion, cond = self.get_primitive_batch(batch, primitive_idx)
                    loss_dict, future_motion_pred = self.common_step(motion, cond, last_primitive)
                    loss = loss_dict['loss']

                # optimize
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(denoiser_model.parameters(), train_args.grad_clip)
                optimizer.step()

                # update the average model using exponential moving average
                if train_args.ema_decay > 0:
                    for param, avg_param in zip(self.denoiser_model.parameters(), self.denoiser_model_avg.parameters()):
                        avg_param.data.mul_(train_args.ema_decay).add_(
                            param.data, alpha=1 - train_args.ema_decay)

                last_primitive = None
                if self.step > train_args.stage1_steps:
                    rollout_prob = min(1.0, (self.step - train_args.stage1_steps) / max(
                        float(train_args.stage2_steps), 1e-6))
                    if torch.rand(1).item() < rollout_prob:
                        last_primitive = future_motion_pred.detach()  # assume future length >= history length

                if self.step % train_args.log_interval == 0:
                    for key in loss_dict:
                        writer.add_scalar(f"loss/{key}", loss_dict[key].item(), self.step)
                    writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], self.step)

                if self.step % train_args.save_interval == 0 or self.step == total_steps:
                    self.save()

                if self.step % train_args.val_interval == 0 or self.step == total_steps:
                    self.validate()

                self.step += 1
                next(progress_bar)
            # t3 = time.time()
            # print(f"get data time: {t2 - t1}, percent:{(t2 - t1) / (t3 - t1)}, step time: {t3 - t2}")

    def get_primitive_batch(self, batch, primitive_idx):
        motion = batch[primitive_idx]['motion_tensor_normalized']  # [bs, D, 1, T]
        cond = {'y': {'text': batch[primitive_idx]['texts'],
                      'text_embedding': batch[primitive_idx]['text_embedding'],  # [bs, 512]
                      'gender': batch[primitive_idx]['gender'],
                      'betas': batch[primitive_idx]['betas'],  # [bs, T, 10]
                      'history_motion': batch[primitive_idx]['history_motion'],  # [bs, D, 1, T]
                      'history_mask': batch[primitive_idx]['history_mask'],
                      'history_length': batch[primitive_idx]['history_length'],
                      'future_length': batch[primitive_idx]['future_length']
                      }
                }
        return motion, cond

    def get_rollout_history(self, last_primitive, cond,
                            return_transform=False,
                            transf_rotmat=None, transf_transl=None
                            ):
        """update history motion seed, update global transform"""
        motion_tensor = last_primitive[:, -self.train_dataset.history_length:, :]  # [B, T, D]
        new_history_frames = self.train_dataset.denormalize(motion_tensor)
        primitive_utility = self.train_dataset.primitive_utility
        rollout_history = []
        genders = cond['y']['gender']
        new_transf_rotmat, new_transf_transl = [], []
        for gender_name in ['female', 'male']:
            gender_idx = [idx for idx in range(len(genders)) if genders[idx] == gender_name]
            if len(gender_idx) == 0:
                continue
            history_feature_dict = primitive_utility.tensor_to_dict(new_history_frames[gender_idx])
            history_feature_dict.update(
                {
                    'transf_rotmat': self.transf_rotmat.repeat(len(gender_idx), 1, 1) if transf_rotmat is None else transf_rotmat[gender_idx],
                    'transf_transl': self.transf_transl.repeat(len(gender_idx), 1, 1) if transf_transl is None else transf_transl[gender_idx],
                    'gender': gender_name,
                    'betas': cond['y']['betas'][gender_idx, -self.train_dataset.history_length:, :],
                }
            )
            pelvis_delta = primitive_utility.calc_calibrate_offset({
                'betas': history_feature_dict['betas'][:, 0, :],  # [B, 10]
                'gender': gender_name,
            })
            history_feature_dict['pelvis_delta'] = pelvis_delta
            use_predicted_joints = getattr(self.args.train_args, 'use_predicted_joints', False)
            canonicalized_history_primitive_dict, blended_feature_dict = primitive_utility.get_blended_feature(
                history_feature_dict, use_predicted_joints=use_predicted_joints)
            new_transf_rotmat.append(canonicalized_history_primitive_dict['transf_rotmat'])
            new_transf_transl.append(canonicalized_history_primitive_dict['transf_transl'])
            history_motion_tensor = primitive_utility.dict_to_tensor(blended_feature_dict)
            rollout_history.append(history_motion_tensor)

        rollout_history = torch.cat(rollout_history, dim=0)
        rollout_history = self.train_dataset.normalize(rollout_history)  # [B, T, D]
        # rollout_history = rollout_history.permute(0, 2, 1).unsqueeze(2)  # [B, D, 1, T_history]

        if return_transform:
            return rollout_history, torch.cat(new_transf_rotmat, dim=0), torch.cat(new_transf_transl, dim=0)
        else:
            return rollout_history

    def save(self):
        denoiser_model = self.denoiser_model if self.denoiser_model_avg is None else self.denoiser_model_avg
        print('save avg model:', self.denoiser_model_avg is not None)
        checkpoint_path = self.args.save_dir / f"checkpoint_{self.step}.pt"
        torch.save({
            'num_steps': self.step,
            'model_state_dict': denoiser_model.state_dict(),
            # 'optimizer_state_dict': self.optimizer.state_dict(),
        }, checkpoint_path)
        print(f"Saved checkpoint at {checkpoint_path}")

    def validate(self):
        original_mode = self.denoiser_model.training
        self.denoiser_model.eval()

        train_args = self.args.train_args
        future_length = self.train_dataset.future_length
        history_length = self.train_dataset.history_length
        num_primitive = self.train_dataset.num_primitive

        with torch.no_grad():
            losses_dict = {}
            for val_idx in tqdm(range(max(128, len(self.val_dataset) // self.batch_size))):
                batch = self.val_dataset.get_batch(self.batch_size)
                sample_primitive_seq = []
                last_primitive = None
                for primitive_idx in range(num_primitive):
                    motion, cond = self.get_primitive_batch(batch, primitive_idx)
                    loss_dict, future_motion_pred = self.common_step(motion, cond, last_primitive)

                    if (self.step == 1 or self.step % train_args.save_interval == 0) and val_idx == 0:
                        motion_tensor = motion.squeeze(2).permute(0, 2, 1)  # [B, T, D]
                        history_motion_gt = motion_tensor[:, :history_length, :]
                        sample_primitive_seq.append(torch.cat([history_motion_gt, future_motion_pred], dim=1))

                    for k, v in loss_dict.items():
                        if k not in losses_dict:
                            losses_dict[k] = []
                        losses_dict[k].append(v.detach())

                    if self.step > train_args.stage1_steps:
                        last_primitive = future_motion_pred.detach()
                    else:
                        last_primitive = None

                if (self.step == 1 or self.step % train_args.save_interval == 0) and val_idx == 0:
                    dataset = self.val_dataset
                    for idx in range(min(self.batch_size, 16)):
                        gender = batch[0]['gender'][idx]
                        betas = torch.stack([batch[i]['betas'][idx] for i in range(num_primitive)],
                                            dim=0)  # [num_primitive, T, 10]
                        texts = [batch[i]['texts'][idx] for i in range(num_primitive)]
                        gt_motion_tensor = torch.stack(
                            [batch[i]['motion_tensor_normalized'][idx] for i in range(num_primitive)],
                            dim=0)  # [num_primitive, D, 1, T]
                        gt_motion_tensor = dataset.denormalize(
                            gt_motion_tensor.squeeze(2).permute(0, 2, 1))  # [num_primitive, T, D]
                        # print('gt_motion_tensor:', gt_motion_tensor.shape)
                        # print('sample_primitive_seq:', len(sample_primitive_seq))
                        # print('sample_primitive_seq:', sample_primitive_seq[0].shape)
                        sample_motion_tensor = torch.stack([sample_primitive_seq[i][idx] for i in range(num_primitive)],
                                                           dim=0)
                        sample_motion_tensor = dataset.denormalize(sample_motion_tensor)  # [num_primitive, T, D]
                        # rollout gt and sampled primitives
                        gt_seq = self.rollout_primitive_seq(motion_tensor=gt_motion_tensor, gender=gender, betas=betas)
                        gt_seq['texts'] = texts
                        sample_seq = self.rollout_primitive_seq(motion_tensor=sample_motion_tensor, gender=gender,
                                                                betas=betas)
                        sample_seq['texts'] = texts

                        export_dir = Path(self.args.save_dir, 'samples', str(self.step), str(idx))
                        export_dir.mkdir(parents=True, exist_ok=True)
                        with open(export_dir / 'real.pkl', 'wb') as f:
                            pickle.dump(gt_seq, f)
                        with open(export_dir / 'sample.pkl', 'wb') as f:
                            pickle.dump(sample_seq, f)

        for k, v in losses_dict.items():
            losses_dict[k] = torch.stack(v).mean().item()
            self.writer.add_scalar(f"val_loss/{k}", losses_dict[k], self.step)
        self.denoiser_model.train(original_mode)

    def rollout_primitive_seq(self, motion_tensor, gender, betas):
        """
        :param motion_tensor: denormalized motion tensor, [num_primitive, T, D]
        :return:
        """
        dataset = self.train_dataset
        num_primitive = dataset.num_primitive
        history_length, future_length = dataset.history_length, dataset.future_length
        primitive_utility = dataset.primitive_utility
        transf_rotmat = self.transf_rotmat
        transf_transl = self.transf_transl
        motion_sequences = None
        for primitive_idx in range(num_primitive):
            future_frames = motion_tensor[[primitive_idx], dataset.history_length:, :] if primitive_idx > 0 else motion_tensor[[primitive_idx], :, :]
            new_history_frames = motion_tensor[[primitive_idx], -dataset.history_length:, :]

            """transform primitive to world coordinate, prepare for serialization"""
            future_feature_dict = primitive_utility.tensor_to_dict(future_frames)
            future_feature_dict.update(
                {
                    'transf_rotmat': transf_rotmat,
                    'transf_transl': transf_transl,
                    'gender': gender,
                    'betas': betas[[primitive_idx], dataset.history_length:, :] if primitive_idx > 0 else betas[[primitive_idx], :, :],
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
                    # print(key, motion_sequences[key].shape)

            """update history motion seed, update global transform"""
            history_feature_dict = primitive_utility.tensor_to_dict(new_history_frames)
            history_feature_dict.update(
                {
                    'transf_rotmat': transf_rotmat,
                    'transf_transl': transf_transl,
                    'gender': gender,
                    'betas': betas[[primitive_idx], -dataset.history_length:, :],
                }
            )
            canonicalized_history_primitive_dict, blended_feature_dict = primitive_utility.get_blended_feature(
                history_feature_dict)
            transf_rotmat, transf_transl = canonicalized_history_primitive_dict['transf_rotmat'], canonicalized_history_primitive_dict['transf_transl']

        motion_sequences = {
            'gender': motion_sequences['gender'],
            'betas': motion_sequences['betas'][0],
            'transl': motion_sequences['transl'][0],
            'global_orient': motion_sequences['global_orient'][0],
            'body_pose': motion_sequences['body_pose'][0],
            'joints': motion_sequences['joints'][0],
            'history_length': history_length,
            'future_length': future_length,
        }
        return motion_sequences

    def close(self):
        self.writer.close()


if __name__ == "__main__":
    args = tyro.cli(MLDArgs)
    trainer = Trainer(args)
    trainer.train()
    trainer.close()
