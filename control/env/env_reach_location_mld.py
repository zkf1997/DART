import smplx.joint_names
import torch
import numpy as np
from copy import deepcopy
import pickle
import random
from pytorch3d import transforms
from torch.cuda import amp

from utils.misc_util import encode_text
from utils.smpl_utils import *

class EnvReachLocationMLD:
    def __init__(self, input_args, denoiser_args, denoiser_model, vae_args, vae_model, diffusion, dataset):
        self.input_args = input_args
        self.args = input_args.env_args
        self.vae_args = vae_args
        self.vae_model = vae_model
        self.denoiser_args = denoiser_args
        self.denoiser_model = denoiser_model
        self.diffusion = diffusion
        self.init_dataset = dataset
        self.device = input_args.device
        self.max_steps = self.args.num_steps
        self.batch_size = self.args.num_envs

        self.history_length = dataset.history_length
        self.future_length = dataset.future_length
        self.primitive_length = dataset.primitive_length
        self.primitive_utility = dataset.primitive_utility
        observation_structure = {
            'goal_dir': {'shape': (3,), },
            'goal_dist': {'shape': (1,), },
            'goal_text_embedding': {'shape': (512,), },
            'motion': {'shape': (self.history_length, self.primitive_utility.feature_dim), },
            'scene': {'shape': (1,), },
        }
        start_idx = 0
        for key in observation_structure:
            observation_structure[key]['numel'] = torch.tensor(observation_structure[key]['shape']).prod().item()
            observation_structure[key]['start_idx'] = start_idx
            start_idx += observation_structure[key]['numel']
            observation_structure[key]['end_idx'] = start_idx
        self.observation_structure = observation_structure
        self.observation_shape = (start_idx, )
        self.action_structure = {
            'shape': denoiser_args.model_args.noise_shape,
            'numel': np.prod(denoiser_args.model_args.noise_shape),
        }
        self.action_shape = (self.action_structure['numel'], )

        self.texts = self.args.texts
        text_embedding = encode_text(self.init_dataset.clip_model, self.texts, force_empty_zero=True).to(dtype=torch.float32, device=self.device)
        self.text_embedding_dict = {text: text_embedding[i] for i, text in enumerate(self.args.texts)}

        self.global_iteration = 0
        self.global_step = 0
        # self.state_human = deepcopy(self.init_seed)
        self.state_human = None
        self.state_goal = {
            'goal_location': torch.zeros(self.batch_size, 3).to(self.device),  # [B, 3]
            # 'goal_texts': np.random.choice(self.texts, self.batch_size),  # use numpy array of string will leads to error when trying to assign strings contain space symbol, like a[0]='run back' leads to a[0]='run '
            'goal_texts': random.choices(self.texts, k=self.batch_size),  # [B]
        }
        self.goal_angle_max = self.args.goal_angle_init
        self.goal_dist_max = self.args.goal_dist_max_init
        self.steps = torch.zeros(self.batch_size, dtype=torch.int32, device=self.device)
        self.sequences = []
        for _ in range(self.batch_size):
            self.sequences.append({})

    def get_text_embedding(self, texts):
        unseen_texts = [text for text in texts if text not in self.text_embedding_dict]
        if len(unseen_texts) > 0:
            unseen_text_embedding = encode_text(self.init_dataset.clip_model, unseen_texts, force_empty_zero=True).to(dtype=torch.float32, device=self.device)
            self.text_embedding_dict.update({text: unseen_text_embedding[i] for i, text in enumerate(unseen_texts)})
        text_embedding = torch.stack([self.text_embedding_dict[text] for text in texts], dim=0)  # [B, text_embedding_dim]
        return text_embedding

    def curriculum_step(self):
        print('curriculum step')
        self.goal_angle_max = min(self.goal_angle_max + self.args.goal_angle_delta, 360.0)
        self.goal_dist_max = min(self.goal_dist_max + self.args.goal_dist_max_delta, self.args.goal_dist_max_clamp)
        self.args.weight_skate = min(self.args.weight_skate + self.args.weight_skate_delta, self.args.weight_skate_max)


    def reset_goal(self, batch_idx, goal_location=None, goal_texts=None, reset_text=True):
        if batch_idx is None:
            batch_idx = torch.arange(self.args.num_envs, device=self.device)

        if goal_location is None:  # random goal within some range of current pelvis location
            global_joints = self.get_global_joints()[batch_idx]  # [B, T, 22, 3]
            global_pelvis = global_joints[:, -1, 0]  # [B, 3]
            body_orient, body_location = get_new_coordinate(global_joints[:, -1])  # [B, 3, 3], [B, 1, 3]
            forward_dir = body_orient[:, :, 1]  # [B, 3]
            forward_dir[:, 2] = 0  # enforce z to be 0
            random_angle = (torch.rand(len(batch_idx), 1, device=self.device) - 0.5) * self.goal_angle_max  # [B, 1]
            random_dist = torch.rand(len(batch_idx), 1, device=self.device) * (self.goal_dist_max - self.args.goal_dist_min) + self.args.goal_dist_min
            random_rotmat = transforms.euler_angles_to_matrix(
                torch.cat([torch.zeros(len(batch_idx), 2, device=self.device), random_angle * torch.pi / 180], dim=1), 'XYZ')  # [B, 3, 3]
            random_dir = torch.einsum('bij,bj->bi', random_rotmat, forward_dir)  # [B, 3]
            random_transl = random_dir * random_dist  # [B, 3]
            goal_location_xy = global_pelvis[:, :2] + random_transl[:, :2]  # [B, 2]
            goal_location = torch.cat([goal_location_xy, torch.zeros(len(batch_idx), 1, device=self.device)], dim=-1)  # [B, 3]

        self.state_goal['goal_location'][batch_idx] = goal_location
        if reset_text:
            if goal_texts is None:
                goal_texts = random.choices(self.texts, k=len(batch_idx))
            for minibatch_idx, global_idx in enumerate(batch_idx):
                self.state_goal['goal_texts'][global_idx] = goal_texts[minibatch_idx]

        if self.args.enable_export:
            for idx in batch_idx:
                self.sequences[idx]['goal_location_list'].append(self.state_goal['goal_location'][idx].cpu())
                if len(self.sequences[idx]['goal_texts_list']) == 0 or self.state_goal['goal_texts'][idx] != self.sequences[idx]['goal_texts_list'][-1]:
                    self.sequences[idx]['goal_texts_list'].append(self.state_goal['goal_texts'][idx])

    def reset(self, batch_idx=None, goal_location=None, goal_texts=None):
        if batch_idx is None:
            batch_idx = torch.arange(self.args.num_envs, device=self.device)
        self.steps[batch_idx] = 0

        # state initialization
        batch = self.init_dataset.get_batch(batch_size=len(batch_idx))[0]
        motion_tensor_normalized = batch['motion_tensor_normalized']  # [B, D, 1, T]
        motion_tensor = motion_tensor_normalized.squeeze(2).permute(0, 2, 1)  # [B, T, D]
        motion_tensor = self.init_dataset.denormalize(motion_tensor)
        history_motion_feature_dict = self.primitive_utility.tensor_to_dict(motion_tensor[:, :self.history_length, :])
        joints = history_motion_feature_dict['joints'].reshape(self.batch_size, self.history_length, 22, 3)
        pelvis_feet_height = joints[:, 0, :, 2].min(dim=1).values  # [B]

        init_seed = {
            'gender': batch['gender'][0],  # assuming same gender
            'betas': batch['betas'][:, :self.history_length, :],  # [B, history, 10]
            'transf_rotmat': torch.eye(3).unsqueeze(0).expand(self.batch_size, 3, 3).to(self.device),
            'transf_transl': torch.zeros(self.batch_size, 1, 3).to(self.device),
        }
        pelvis_delta = self.primitive_utility.calc_calibrate_offset({
            'betas': init_seed['betas'][:, 0, :],
            'gender': init_seed['gender'],
        })
        init_seed['pelvis_delta'] = pelvis_delta
        init_seed['transf_transl'][:, :, 2] = -pelvis_feet_height.unsqueeze(1)  # make init frame foot on floor
        init_seed.update(history_motion_feature_dict)
        if self.state_human is None:
            assert len(batch_idx) == self.batch_size
            self.state_human = init_seed
        else:
            for key in self.state_human:
                if key != 'gender':
                    self.state_human[key][batch_idx] = init_seed[key]

        # set up sequences for exporting
        if self.args.enable_export:
            history_feature_dict = get_dict_subset_by_batch(self.state_human, batch_idx)
            history_primitive_dict = self.primitive_utility.feature_dict_to_smpl_dict(history_feature_dict)
            history_primitive_dict = self.primitive_utility.transform_primitive_to_world(history_primitive_dict)
            for minibatch_idx, idx in enumerate(batch_idx):
                self.sequences[idx] = {
                    'goal_location_list': [],
                    'goal_texts_list': [],
                    'goal_location_idx': [0] * self.history_length,
                    'goal_texts_idx': [0] * self.history_length,
                    'transl': history_primitive_dict['transl'][minibatch_idx],
                    'global_orient': history_primitive_dict['global_orient'][minibatch_idx],
                    'body_pose': history_primitive_dict['body_pose'][minibatch_idx],
                    'betas': history_primitive_dict['betas'][minibatch_idx],
                    'joints': history_primitive_dict['joints'][minibatch_idx],
                    'gender': history_primitive_dict['gender'],
                    'action': [],
                    'obs': [],
                }

        self.reset_goal(batch_idx, goal_location=goal_location, goal_texts=goal_texts)

        info = {}
        return self.get_observation(), info

    def get_global_joints(self, history_feature_dict=None):
        """
        Get global joints from the local joints
        :return:
        global_joints: [B, T, 22, 3]
        """
        history_feature_dict = self.state_human if history_feature_dict is None else history_feature_dict
        local_joints = history_feature_dict['joints']  # [B, T, 22 * 3]
        B, T, _ = local_joints.shape
        local_joints = local_joints.view(B, T, 22, 3)
        transf_rotmat, transf_transl = history_feature_dict['transf_rotmat'], history_feature_dict['transf_transl']
        global_joints = torch.einsum('bij,btkj->btki', transf_rotmat, local_joints) + transf_transl.unsqueeze(1)

        return global_joints

    def get_global_pelvis(self, history_feature_dict=None, frame_idx=-1):
        global_joints = self.get_global_joints(history_feature_dict=history_feature_dict)
        global_pelvis = global_joints[:, frame_idx, 0]  # [B, 3]
        return global_pelvis

    def get_observation(self):
        # return the observation of the current state
        history_feature_dict = self.state_human
        goal_location = self.state_goal['goal_location']
        goal_text_embedding = self.get_text_embedding(self.state_goal['goal_texts'])
        global_joints = self.get_global_joints(history_feature_dict=history_feature_dict)  # [B, T, 22, 3]
        global_pelvis = global_joints[:, -1, 0]  # [B, 3]
        global_goal_dir = goal_location - global_pelvis  # [B, 3]
        global_goal_dir[:, 2] = 0  # enforce z to be 0
        goal_dist = torch.norm(global_goal_dir, dim=-1, keepdim=True)  # [B, 1]
        global_goal_dir = global_goal_dir / goal_dist.clip(min=1e-12)

        # clip the goal dir so that it is not too far away from current moving dir, to avoid too large rotation
        # moving_dir = global_joints[:, -1, 0] - global_joints[:, -2, 0]  # [B, 3]
        # moving_dir[:, 2] = 0  # enforce z to be 0

        body_orient, body_location = get_new_coordinate(global_joints[:, -1])  # [B, 3, 3], [B, 1, 3]
        forward_dir = body_orient[:, :, 1]  # [B, 3]
        forward_dir[:, 2] = 0  # enforce z to be 0
        moving_dir = forward_dir

        moving_dir = moving_dir / torch.norm(moving_dir, dim=-1, keepdim=True).clip(min=1e-12)


        cos_theta = torch.einsum('bi,bi->b', global_goal_dir, moving_dir)  # [B]
        cos_theta = cos_theta.clip(min=np.cos(np.deg2rad(self.args.obs_goal_angle_clip)), max=1)  # clip to avoid too large rotation
        sign = torch.sign(torch.cross(moving_dir, global_goal_dir)[:, 2])  # [B]
        theta = torch.acos(cos_theta) * sign  # [B]
        rotation_matrix = transforms.euler_angles_to_matrix(
                torch.cat([torch.zeros(self.batch_size, 2, device=self.device), theta.unsqueeze(1)], dim=1), 'XYZ')  # [B, 3, 3]
        clipped_global_goal_dir = torch.einsum('bij,bj->bi', rotation_matrix, moving_dir)  # [B, 3]
        if torch.isnan(clipped_global_goal_dir).any():
            print('nan clipped_global_goal_dir', clipped_global_goal_dir)
        global_goal_dir = clipped_global_goal_dir

        # transform goal dir to local coordinate
        transf_rotmat, transf_transl = history_feature_dict['transf_rotmat'], history_feature_dict['transf_transl']
        local_goal_dir = torch.einsum('bij,bj->bi', transf_rotmat.permute(0, 2, 1), global_goal_dir)  # [B, 3]

        motion_tensor = self.primitive_utility.dict_to_tensor(history_feature_dict)  # [B, T, D]
        floor_height = 0 - global_joints[:, 0, 0, 2]  # [B], relative to first frame pelvis
        observation = {
            'goal_dir': local_goal_dir,
            'goal_dist': goal_dist.clip(max=self.args.obs_goal_dist_clip),
            'goal_text_embedding': goal_text_embedding,
            'motion': motion_tensor,  # unnormalized motion tensor, [B, T, D]
            'scene': floor_height,
        }
        observation_vector = torch.cat([observation[key].reshape(self.batch_size, -1) for key in self.observation_structure], dim=-1)

        return observation_vector

    def get_new_state_human(self, action):
        action = action.view(self.batch_size, *self.action_structure['shape'])  # [B, 1, D]

        denoiser_args, denoiser_model, vae_args, vae_model = self.denoiser_args, self.denoiser_model, self.vae_args, self.vae_model
        diffusion = self.diffusion
        dataset = self.init_dataset
        primitive_utility = self.primitive_utility
        sample_fn = diffusion.ddim_sample_loop
        batch_size, device = self.batch_size, self.device
        history_length, future_length, primitive_length = self.history_length, self.future_length, self.primitive_length

        history_feature_dict = self.state_human
        transf_rotmat, transf_transl = history_feature_dict['transf_rotmat'], history_feature_dict['transf_transl']
        gender = history_feature_dict['gender']  # assuming all the same gender
        pelvis_delta = history_feature_dict['pelvis_delta']

        text_embedding = self.get_text_embedding(self.state_goal['goal_texts'])  # [B, 512]
        history_motion_tensor = primitive_utility.dict_to_tensor(history_feature_dict)
        history_motion_tensor = dataset.normalize(history_motion_tensor)  # [B, T, D]
        guidance_param = torch.ones(batch_size, *denoiser_args.model_args.noise_shape).to(
            device=device) * self.input_args.guidance_param
        y = {
            'text_embedding': text_embedding,
            'history_motion_normalized': history_motion_tensor,
            'scale': guidance_param,
        }

        with torch.no_grad():
            with amp.autocast(enabled=True, dtype=torch.float16):
                x_start_pred = sample_fn(
                    denoiser_model,
                    (batch_size, *denoiser_args.model_args.noise_shape),
                    clip_denoised=False,
                    model_kwargs={'y': y},
                    skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                    init_image=None,
                    progress=False,
                    dump_steps=None,
                    noise=action,
                    const_noise=False,
                )  # [B, T=1, D]
                latent_pred = x_start_pred.permute(1, 0, 2)  # [T=1, B, D]
                future_motion_pred = vae_model.decode(latent_pred, history_motion_tensor, nfuture=future_length,
                                                      scale_latent=denoiser_args.rescale_latent)  # [B, F, D], normalized

        future_frames = dataset.denormalize(future_motion_pred)
        new_history_frames = future_frames[:, -history_length:, :]

        """transform primitive to world coordinate, prepare for serialization"""
        future_feature_dict = primitive_utility.tensor_to_dict(future_frames)
        future_feature_dict.update(
            {
                'transf_rotmat': transf_rotmat,
                'transf_transl': transf_transl,
                'gender': gender,
                'betas': self.state_human['betas'][:, [0], :].repeat(1, self.future_length, 1),
                'pelvis_delta': pelvis_delta,
            }
        )
        future_primitive_dict = primitive_utility.feature_dict_to_smpl_dict(future_feature_dict)
        future_primitive_dict = primitive_utility.transform_primitive_to_world(future_primitive_dict)

        if self.args.enable_export:
            # update sequences for exporting
            current_obs = self.get_observation()
            for idx in range(self.batch_size):
                self.sequences[idx]['goal_location_idx'] += [len(
                    self.sequences[idx]['goal_location_list']) - 1] * self.future_length
                self.sequences[idx]['goal_texts_idx'] += [len(
                    self.sequences[idx]['goal_texts_list']) - 1] * self.future_length
                for key in ['transl', 'global_orient', 'body_pose', 'betas', 'joints']:
                    self.sequences[idx][key] = torch.cat([self.sequences[idx][key], future_primitive_dict[key][idx]],
                                                         dim=0)  # [T, ...]
                self.sequences[idx]['action'].append(action[idx].cpu())
                self.sequences[idx]['obs'].append(current_obs[idx].cpu())

        """update history motion seed, update global transform"""
        history_feature_dict = primitive_utility.tensor_to_dict(new_history_frames)
        history_feature_dict.update(
            {
                'transf_rotmat': transf_rotmat,
                'transf_transl': transf_transl,
                'gender': gender,
                'betas': self.state_human['betas'],
                'pelvis_delta': pelvis_delta,
            }
        )
        canonicalized_history_primitive_dict, blended_feature_dict = primitive_utility.get_blended_feature(
            history_feature_dict,
            use_predicted_joints=self.args.use_predicted_joints)

        return blended_feature_dict, future_feature_dict, future_primitive_dict

    def get_reward(self, new_state_human, future_feature_dict, global_future_primitive_dict):
        old_state_human = self.state_human
        old_global_pelvis = self.get_global_pelvis(history_feature_dict=old_state_human)
        new_global_pelvis = self.get_global_pelvis(history_feature_dict=new_state_human)
        goal_location = self.state_goal['goal_location']
        new_goal_dist = torch.norm((goal_location - new_global_pelvis)[:, :2], dim=-1)  # xy distance [B]
        old_goal_dist = torch.norm((goal_location - old_global_pelvis)[:, :2], dim=-1)
        reward_dist = old_goal_dist - new_goal_dist

        terminated = new_goal_dist > self.args.terminate_threshold

        future_global_joints = self.get_global_joints(history_feature_dict=future_feature_dict)
        history_global_joints = self.get_global_joints(history_feature_dict=old_state_human)  # [B, H, 22, 3]
        all_global_joints = torch.cat([history_global_joints, future_global_joints], dim=1)  # [B, H+F, 22, 3]
        future_global_pelvis = future_global_joints[:, :, 0]  # [B, T, 3]
        goal_dist = torch.norm((goal_location.unsqueeze(1) - future_global_pelvis)[:, :, :2], dim=-1)  # [B, T]
        min_goal_dist = goal_dist.min(dim=1).values
        success = min_goal_dist < self.args.success_threshold
        reward_success = success.float()

        # add reward for foot floor contact
        fps = self.init_dataset.target_fps
        assert fps == 30
        floor_height = 0
        foot_joints = future_global_joints[:, :, FOOT_JOINTS_IDX]  # [B, T, foot, 3]
        foot_joints_height = foot_joints[:, :, :, 2]  # [B, T, foot]
        # floor contact reward for locomotion where human do not jump off floor
        foot_floor_thresh = 0.03
        clamped_dist_floor = (torch.abs(foot_joints_height.amin(dim=-1) - floor_height) - foot_floor_thresh).clamp(min=0)  # [B, F]
        reward_foot_floor = -clamped_dist_floor.mean(dim=-1)
        # reward for hopping, only apply floor contact reward when foot do not move vertically
        goal_texts = self.state_goal['goal_texts']
        is_hop = np.array(['hop' in goal_text for goal_text in goal_texts])
        is_run = np.array(['run' in goal_text for goal_text in goal_texts])
        reward_foot_floor[is_hop] *= 0.1  # reduce reward for hopping
        reward_foot_floor[is_run] *= 0.1  # reduce reward for running


        all_foot_joints = torch.cat([history_global_joints[:, :, FOOT_JOINTS_IDX], foot_joints], dim=1)  # [B, H+F, foot, 3]
        all_foot_joints_height = all_foot_joints[:, :, :, 2]  # [B, H+F, foot]
        foot_joints_diff = torch.norm(all_foot_joints[:, self.history_length:] - all_foot_joints[:, self.history_length - 1:-1],
                                     dim=-1)  # [B, F, foot]
        foot_joints_height_consecutive_max = torch.maximum(all_foot_joints_height[:, self.history_length - 1:-1],
                                                           all_foot_joints_height[:, self.history_length:])  # maximum height of current or previous frame
        skate = foot_joints_diff * (2 - 2 ** (foot_joints_height_consecutive_max / foot_floor_thresh).clamp(min=0, max=1))  # [B, F, foot]
        reward_skate = -skate.mean(dim=[1, 2])
        reward_skate_rigid = -skate.amax(dim=[1, 2])

        # moving orientation should align with goal orientation
        moving_orientation = (new_global_pelvis - old_global_pelvis)[:, :2]
        moving_orientation = moving_orientation / torch.norm(moving_orientation, dim=-1, keepdim=True).clip(min=1e-12)
        old_goal_orientation = (goal_location - old_global_pelvis)[:, :2]
        old_goal_orientation = old_goal_orientation / torch.norm(old_goal_orientation, dim=-1, keepdim=True).clip(min=1e-12)
        reward_orient = ((torch.einsum('bi,bi->b', moving_orientation, old_goal_orientation) + 1) / 2.0)[0]

        # penalize body rotation
        l_hips = all_global_joints[:, :, 1]  # [B, H+F, 3]
        r_hips = all_global_joints[:, :, 2]  # [B, H+F, 3]
        x_axis = r_hips - l_hips  # [B, H+F, 3]
        x_axis[:, :, 2] = 0
        x_axis = x_axis / torch.norm(x_axis, dim=-1, keepdim=True)  # [B, H+F, 3]
        dot_product = torch.einsum('bij,bij->bi', x_axis[:, self.history_length:], x_axis[:, self.history_length - 1:-1])  # [B, F]
        reward_rotation = dot_product.mean(dim=-1) - 1

        # jerk reward
        vel = all_global_joints[:, 1:] - all_global_joints[:, :-1]  # --> B x T-1 x 22 x 3
        acc = vel[:, 1:] - vel[:, :-1]  # --> B x T-2 x 22 x 3
        jerk = acc[:, 1:] - acc[:, :-1]  # --> B x T-3 x 22 x 3
        jerk = torch.abs(jerk).sum(dim=-1)  # --> B x T-3 x 22, compute L1 norm of jerk
        jerk = jerk.amax(dim=[1, 2])  # --> B, Get the max of the jerk across all joints and frames
        reward_jerk = -jerk

        # delta reward
        pred_feature_dict = {}
        for key in ['joints', 'transl', 'joints_delta', 'transl_delta', 'global_orient_delta_6d', 'poses_6d']:
            pred_feature_dict[key] = torch.cat([old_state_human[key][:, [-1]], future_feature_dict[key]],
                                               dim=1)  # [B, 1+F, D]
        pred_joints_delta = pred_feature_dict['joints_delta'][:, :-1, :]
        pred_transl_delta = pred_feature_dict['transl_delta'][:, :-1, :]
        pred_orient_delta = pred_feature_dict['global_orient_delta_6d'][:, :-1, :]
        calc_joints_delta = pred_feature_dict['joints'][:, 1:, :] - pred_feature_dict['joints'][:, :-1, :]
        calc_transl_delta = pred_feature_dict['transl'][:, 1:, :] - pred_feature_dict['transl'][:, :-1, :]
        pred_orient = transforms.rotation_6d_to_matrix(pred_feature_dict['poses_6d'][:, :, :6])  # [B, 1+F, 3, 3]
        calc_orient_delta_matrix = torch.matmul(pred_orient[:, 1:],
                                                pred_orient[:, :-1].permute(0, 1, 3, 2))
        calc_orient_delta_6d = transforms.matrix_to_rotation_6d(calc_orient_delta_matrix)
        joints_delta_diff = ((pred_joints_delta - calc_joints_delta).abs()).mean(dim=-1).amax(dim=-1)  # [B], mean over feature dim and max over frames, often first frame has most problem
        transl_delta_diff = ((pred_transl_delta - calc_transl_delta).abs()).mean(dim=-1).amax(dim=-1)  # [B]
        orient_delta_diff = ((pred_orient_delta - calc_orient_delta_6d).abs()).mean(dim=-1).amax(dim=-1)  # [B]
        reward_delta = -(joints_delta_diff + transl_delta_diff + orient_delta_diff)

        reward_dict = {
            'reward_dist': reward_dist,
            'reward_success': reward_success,
            'reward_foot_floor': reward_foot_floor,
            'reward_skate': reward_skate,
            'reward_skate_rigid': reward_skate_rigid,
            'reward_orient': reward_orient,
            'reward_rotation': reward_rotation,
            'reward_jerk': reward_jerk,
            'reward_delta': reward_delta,
        }
        reward = (reward_dist * self.args.weight_dist + reward_success * self.args.weight_success +
                  reward_foot_floor * self.args.weight_foot_floor +
                  reward_skate * self.args.weight_skate + reward_skate_rigid * self.args.weight_skate_rigid +
                  reward_orient * self.args.weight_orient + reward_rotation * self.args.weight_rotation +
                  reward_jerk * self.args.weight_jerk + reward_delta * self.args.weight_delta
                  )

        return reward, success, terminated, reward_dict

    def step(self, action, next_goal_location=None, next_goal_texts=None, reset_text=True):
        # if terminate or truncate, call reset and return the initial state observation as next observation
        new_state_human, future_feature_dict, global_future_primitive_dict = self.get_new_state_human(action)
        reward, success, terminated, reward_dict = self.get_reward(new_state_human, future_feature_dict, global_future_primitive_dict)
        self.state_human = new_state_human

        self.global_step += self.args.num_envs
        self.steps = self.steps + 1
        truncated = self.steps >= self.max_steps
        # terminated = torch.zeros_like(truncated)
        done = truncated | terminated
        info = {
            'num_success': success.sum().item(),
            'num_truncated': truncated.sum().item(),
            'num_terminated': terminated.sum().item(),
            'reward_dict': reward_dict,
        }

        if success.any():
            success_idx = torch.nonzero(success, as_tuple=True)[0]
            if next_goal_location is not None and next_goal_texts is not None:  # reset goal using inputs, for testing
                self.reset_goal(success_idx, goal_location=next_goal_location[success_idx], goal_texts=next_goal_texts[success_idx.cpu().numpy()], reset_text=reset_text)
            else:
                self.reset_goal(success_idx, reset_text=False)

        if done.any():
            reset_idx = torch.nonzero(done, as_tuple=True)[0]
            if self.args.enable_export and self.global_iteration % self.args.export_interval == 0:
                self.save_rollouts(reset_idx)
            next_observation, _ = self.reset(reset_idx)
        else:
            next_observation = self.get_observation()
        return next_observation, reward, success, terminated, truncated, info

    def save_rollouts(self, batch_idx):
        for idx in batch_idx[:self.args.max_export]:  # limit max number of exporting at the same time, to save storage
            sequence = self.sequences[idx]
            save_path = self.args.save_dir / f'iter{self.global_iteration}' / f'step{self.global_step}_{idx}.pkl'
            save_path.parent.mkdir(parents=True, exist_ok=True)
            sequence['texts'] = sequence['goal_texts_list']
            sequence['text_idx'] = sequence['goal_texts_idx']  # for backward compatibility
            sequence['action'] = torch.stack(sequence['action'], dim=0)  # [num_rollout, future_len, D]
            sequence['obs'] = torch.stack(sequence['obs'], dim=0)
            sequence = tensor_dict_to_device(sequence, 'cpu')
            with open(save_path, 'wb') as f:
                pickle.dump(sequence, f)

    def close(self):
        pass

