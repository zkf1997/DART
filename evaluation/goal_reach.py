import pdb

import numpy as np
import torch
import pickle
import json
from utils.smpl_utils import *
from evaluation.inbetween import get_metric_statistics, calc_skate

debug = 0
success_thresh = 0.3
num_seq = 32

from scipy.spatial.transform import Rotation as R
hml3d_to_canonical = R.from_euler('xyz', [90, 0, 180], degrees=True).as_matrix()

def eval_gamma(gamma_dir, cfg_file=None):
    gamma_dir = Path(gamma_dir)
    metrics = {
        'time': [],
        'dist': [],
        'skate': [],
        'success': [],
        'floor_dist': [],
    }

    fps = 40
    body_model = body_model_dict['male'].to('cuda')
    if cfg_file is None:
        cfg_file = './data/test_locomotion/test_walk_long.json'
    with open(cfg_file, 'r') as f:
        path_cfg = json.load(f)
    for seq_idx in range(num_seq):
        time_list = []
        dist_list = []
        skate_list = []
        success_list = []
        floor_dist_list = []
        for path_idx, path in enumerate(path_cfg):
            goal_location = torch.tensor(path['goal_location'])
            seq_path = gamma_dir / f'scene000_path{path_idx:03d}_seq{seq_idx:03d}' / 'results_ssm2_67_condi_marker_0.pkl.pkl'
            with open(seq_path, 'rb') as f:
                smpl_param = pickle.load(f)
            num_frames = smpl_param['transl'].shape[1]
            body_pose = torch.tensor(smpl_param['body_pose'][0]).to('cuda')
            body_pose = transforms.axis_angle_to_matrix(body_pose.reshape(-1, 3)).view(-1, 21, 3, 3)
            global_orient = torch.tensor(smpl_param['global_orient'][0]).to('cuda')
            global_orient = transforms.axis_angle_to_matrix(global_orient).view(-1, 3, 3)
            betas = torch.tensor(smpl_param['betas']).repeat(num_frames, 1).to('cuda')
            transl = torch.tensor(smpl_param['transl'][0]).to('cuda')
            # print(transl.shape, body_pose.shape, global_orient.shape, betas.shape)
            smplx_output = body_model(
                betas=betas,
                transl=transl,
                body_pose=body_pose,
                global_orient=global_orient,
                return_vertices=False,
            )
            seq_joints = smplx_output.joints[:, :22, :].cpu()  # [T, 22, 3]

            skate = calc_skate(seq_joints[None], fps=fps)
            time = num_frames / fps
            # dist = torch.norm(seq_joints[-1, 0, :2] - goal_location[-1, :2], p=2)
            dist = torch.norm(seq_joints[-8:, 0, :2] - goal_location[[-1], :2], p=2, dim=-1).min()
            success = dist < success_thresh
            floor_dist = seq_joints[:, FOOT_JOINTS_IDX, 2].amin(dim=1).abs().mean()
            time_list.append(time)
            dist_list.append(dist.item())
            skate_list.append(skate.item())
            success_list.append(success)
            floor_dist_list.append(floor_dist.item())
        metrics['time'].append(np.array(time_list).mean())
        metrics['dist'].append(np.array(dist_list).mean())
        metrics['skate'].append(np.array(skate_list).mean())
        metrics['success'].append(np.array(success_list).mean())
        metrics['floor_dist'].append(np.array(floor_dist_list).mean())

    for key in metrics:
        metrics[key] = get_metric_statistics(metrics[key], num_seq)
    return metrics

def eval_primitive(res_dir, cfg_file=None):
    res_dir = Path(res_dir)
    metrics = {
        'time': [],
        'dist': [],
        'skate': [],
        'success': [],
        'floor_dist': [],
    }
    num_seq = 32
    fps = 30
    if cfg_file is None:
        cfg_file = './data/test_locomotion/test_walk_long.json'
    with open(cfg_file, 'r') as f:
        path_cfg = json.load(f)
    for seq_idx in range(num_seq):
        time_list = []
        dist_list = []
        skate_list = []
        success_list = []
        floor_dist_list = []
        for path_idx, path in enumerate(path_cfg):
            goal_location = torch.tensor(path['goal_location'])
            seq_path = res_dir / f'{Path(cfg_file).stem}_path{path_idx}' / f'{seq_idx}.pkl'
            with open(seq_path, 'rb') as f:
                seq_joints = pickle.load(f)['joints'].reshape(-1, 22, 3)

            num_frames = seq_joints.shape[0]
            skate = calc_skate(seq_joints[None], fps=fps)
            time = num_frames / fps
            dist = torch.norm(seq_joints[-8:, 0, :2] - goal_location[[-1], :2], p=2, dim=-1).min()
            floor_dist = seq_joints[:, FOOT_JOINTS_IDX, 2].amin(dim=1).abs().mean()
            success = dist < success_thresh
            if success == False:
                print(seq_path, dist)
            time_list.append(time)
            dist_list.append(dist.item())
            skate_list.append(skate.item())
            success_list.append(success)
            floor_dist_list.append(floor_dist.item())
        metrics['time'].append(np.array(time_list).mean())
        metrics['dist'].append(np.array(dist_list).mean())
        metrics['skate'].append(np.array(skate_list).mean())
        metrics['success'].append(np.array(success_list).mean())
        metrics['floor_dist'].append(np.array(floor_dist_list).mean())

    for key in metrics:
        metrics[key] = get_metric_statistics(metrics[key], num_seq)
    return metrics

if __name__ == '__main__':
    results_dict = {
        # 'gamma': eval_gamma('./eval_results/goal_reach/gamma',
        #                     cfg_file='./data/test_locomotion/test_walk_long.json'),

        'walk_long_fixtext_repeat': eval_primitive('./policy_train/reach_location_mld/fixtext_repeat_floor100_hop10_skate100/env_test/',
                                                   cfg_file='./data/test_locomotion/test_walk_long.json'),

        'run_long_fixtext_repeat': eval_primitive('./policy_train/reach_location_mld/fixtext_repeat_floor100_hop10_skate100/env_test/',
                                                  cfg_file='./data/test_locomotion/test_run_long.json'),

        'hop_long_fixtext_repeat': eval_primitive('./policy_train/reach_location_mld/fixtext_repeat_floor100_hop10_skate100/env_test/',
                                                  cfg_file='./data/test_locomotion/test_hop_long.json'),
    }
    print(results_dict)
    export_path = Path('./eval_results/goal_reach/goal_reach_results_fps.json')
    with open(export_path, 'w') as f:
        json.dump(results_dict, f, indent=4)
    print('save results at:', export_path)
