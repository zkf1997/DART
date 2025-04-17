import pdb

import numpy as np
import torch
import pickle
import json
from utils.smpl_utils import *
from scipy.spatial.transform import Rotation as R

debug = 0
zup_to_yup = np.array([[1, 0, 0],
                           [0, 0, 1],
                           [0, 1, 0]])

yup_to_zup = np.array([[1, 0, 0],
                        [0, 0, 1],
                        [0, 1, 0]])

hml3d_to_canonical = R.from_euler('xyz', [90, 0, 180], degrees=True).as_matrix()

def get_metric_statistics(values, replication_times):
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    conf_interval = 1.96 * std / np.sqrt(replication_times)
    return float(mean), float(conf_interval)

def calc_jerk(joints, fps=30):
    vel = joints[:, 1:] - joints[:, :-1]  # --> B x T-1 x 22 x 3
    vel = vel * fps
    acc = vel[:, 1:] - vel[:, :-1]  # --> B x T-2 x 22 x 3
    acc = acc * fps
    jerk = acc[:, 1:] - acc[:, :-1]  # --> B x T-3 x 22 x 3
    jerk = jerk * fps
    jerk = torch.sqrt((jerk ** 2).sum(dim=-1))  # --> B x T-3 x 22, compute L2 norm of jerk

    return jerk.mean()

def calc_skate(joints, foot_floor_thresh=0.03, fps=30):
    foot_joints = joints[:, :, FOOT_JOINTS_IDX, :]
    foot_joints_height = foot_joints[:, :, :, 2]  # [B, T, foot]
    foot_joints_diff = torch.norm(foot_joints[:, 1:] - foot_joints[:, :-1], dim=-1, p=2)  # [B, T-1, foot]
    foot_joints_height_consecutive_max = torch.maximum(foot_joints_height[:, :-1],
                                                       foot_joints_height[:, 1:])  # maximum height of current or previous frame
    skate = foot_joints_diff * (
                2 - 2 ** (foot_joints_height_consecutive_max / foot_floor_thresh).clamp(min=0, max=1))  # [B, F, foot]
    return skate.mean() * fps

def eval_mdm(mdm_path, fps=20, history_length=1, num_try=8):
    base_dir = Path(mdm_path)
    metrics = {
        'jerk': [],
        'skate': [],
        # 'gt_jerk': [],
        # 'gt_skate': [],
        'history_error': [],
        'future_error': [],
    }
    for try_idx in range(num_try):
        jerk_list = []
        skate_list = []
        # gt_skate_list = []
        # gt_jerk_list = []
        history_error_list = []
        future_error_list = []
        for seq_dir in base_dir.glob('./*'):
            gt_seq_path = list(seq_dir.glob('./*_hml3d.npy'))[0]
            res_seq = list(seq_dir.glob('./*/joints.npy'))[0]
            gt_joints = np.load(gt_seq_path)
            num_frames = gt_joints.shape[0]
            gt_joints_zup = gt_joints @ hml3d_to_canonical.T
            res_joints = np.load(res_seq)[[try_idx], :num_frames]

            res_joints_zup = res_joints @ hml3d_to_canonical.T
            gt_joints_zup = torch.tensor(gt_joints_zup).float().unsqueeze(0)  # --> 1 x T x 22 x 3
            res_joints_zup = torch.tensor(res_joints_zup).float()  # --> B x T x 22 x 3

            # check foot joints
            history_error = torch.norm(gt_joints_zup[:, :history_length] - res_joints_zup[:, :history_length], dim=-1, p=2).mean()
            future_error = torch.norm(gt_joints_zup[:, -1] - res_joints_zup[:, -1], dim=-1, p=2).mean()
            # first frame foot on floor
            floor_height = res_joints_zup[:, :, FOOT_JOINTS_IDX, 2].min().item()
            res_joints_zup[:, :, :, 2] = res_joints_zup[:, :, :, 2] - floor_height
            floor_height = gt_joints_zup[:, :, FOOT_JOINTS_IDX, 2].min().item()
            gt_joints_zup[:, :, :, 2] = gt_joints_zup[:, :, :, 2] - floor_height
            jerk = calc_jerk(res_joints_zup, fps=fps)
            skate = calc_skate(res_joints_zup, fps=fps)

            # gt_jerk = calc_jerk(gt_joints_zup, fps=fps)
            # gt_skate = calc_skate(gt_joints_zup, fps=fps)

            jerk_list.append(jerk)
            if 'crawl' not in gt_seq_path.name and 'climb' not in gt_seq_path.name:
                print('add mdm:', gt_seq_path.name)
                skate_list.append(skate)
            #     gt_skate_list.append(gt_skate)
            # gt_jerk_list.append(gt_jerk)
            history_error_list.append(history_error)
            future_error_list.append(future_error)

        metrics['jerk'].append(torch.stack(jerk_list).mean())
        metrics['skate'].append(torch.stack(skate_list).mean())
        # metrics['gt_skate'].append(torch.stack(gt_skate_list).mean())
        # metrics['gt_jerk'].append(torch.stack(gt_jerk_list).mean())
        metrics['history_error'].append(torch.stack(history_error_list).mean())
        metrics['future_error'].append(torch.stack(future_error_list).mean())

    for key in metrics:
        metrics[key] = get_metric_statistics(metrics[key], num_try)

    return metrics


def eval_smpl(smpl_path, fps=30, history_length=1, num_try = 8):
    base_dir = Path(smpl_path)
    metrics = {
        'jerk': [],
        'skate': [],
        # 'gt_jerk': [],
        # 'gt_skate': [],
        'history_error': [],
        'future_error': [],
    }
    for try_idx in range(num_try):
        jerk_list = []
        skate_list = []
        # gt_skate_list = []
        # gt_jerk_list = []
        history_error_list = []
        future_error_list = []
        for seq_dir in base_dir.glob('./*'):
            gt_seq_path = seq_dir / 'input.pkl'
            with open(gt_seq_path, 'rb') as f:
                gt_joints = pickle.load(f)['joints'].unsqueeze(0)


            res_seq_path = seq_dir / f'sample_{try_idx}.pkl'
            with open(res_seq_path, 'rb') as f:
                res_joints = torch.tensor(pickle.load(f)['joints']).unsqueeze(0)
            gt_joints_zup = gt_joints
            res_joints_zup = res_joints

            history_error = torch.norm(gt_joints_zup[:, :history_length] - res_joints_zup[:, :history_length], dim=-1,
                                       p=2).mean()
            future_error = torch.norm(gt_joints_zup[:, -1] - res_joints_zup[:, -1], dim=-1, p=2).mean()
            floor_height = gt_joints_zup[:, :, FOOT_JOINTS_IDX, 2].min().item()
            gt_joints_zup[:, :, :, 2] -= floor_height
            floor_height = res_joints_zup[:, :, FOOT_JOINTS_IDX, 2].min().item()
            res_joints_zup[:, :, :, 2] -= floor_height
            jerk = calc_jerk(res_joints_zup, fps=fps)
            skate = calc_skate(res_joints_zup, fps=fps)
            # gt_jerk = calc_jerk(gt_joints_zup, fps=fps)
            # gt_skate = calc_skate(gt_joints_zup, fps=fps)

            jerk_list.append(jerk)
            if 'crawl' not in gt_seq_path.parent.name and 'climb' not in gt_seq_path.parent.name:
                print('add our:', gt_seq_path.parent.name)
                skate_list.append(skate)
            #     gt_skate_list.append(gt_skate)
            # gt_jerk_list.append(gt_jerk)
            history_error_list.append(history_error)
            future_error_list.append(future_error)

        metrics['jerk'].append(torch.stack(jerk_list).mean())
        metrics['skate'].append(torch.stack(skate_list).mean())
        # metrics['gt_jerk'].append(torch.stack(gt_jerk_list).mean())
        # metrics['gt_skate'].append(torch.stack(gt_skate_list).mean())
        metrics['history_error'].append(torch.stack(history_error_list).mean())
        metrics['future_error'].append(torch.stack(future_error_list).mean())

    for key in metrics:
        metrics[key] = get_metric_statistics(metrics[key], num_try)

    return metrics

if __name__ == "__main__":
    eval_results = {
        # 'dno': eval_mdm('./eval_results/inbetween/smplh_20fps_1f/dno', fps=20),
        # 'omnicontrol': eval_mdm('./eval_results/inbetween/smplh_20fps_1f/omnicontrol', fps=20),
        'ours': eval_smpl('./mld_denoiser/smplh_hml3d_2_8_4/checkpoint_300000/optim/inbetween/repeatseed/', fps=20),
    }
    print(eval_results)
    export_path = Path(f'./eval_results/inbetween/smplh_20fps_1f/smplh_inbetween_results_20fps_1f.json')
    with open(export_path, 'w') as f:
        json.dump(eval_results, f, indent=4)
    print('results saved at:', export_path)