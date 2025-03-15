import numpy as np
import os
import sys
import pickle
import torch
from pytorch3d import transforms
from pathlib import Path

def export_smpl(sequence, out_path):
    if len(sequence['body_pose'].shape) == 2:
        poses = torch.cat([torch.tensor(sequence['global_orient']), torch.tensor(sequence['body_pose'])], dim=1)
        betas = sequence['betas'][0, :10]
        transl = sequence['transl']
    elif sequence['body_pose'].shape[-1] == 63:  # gamma results
        poses = torch.cat([torch.tensor(sequence['global_orient'][0]), torch.tensor(sequence['body_pose'][0])], dim=1)
        betas = sequence['betas'][0, :10]
        transl = sequence['transl'][0]
    else:
        poses = transforms.matrix_to_axis_angle(
            torch.cat([sequence['global_orient'].reshape(-1, 1, 3, 3), sequence['body_pose']], dim=1)
        ).reshape(-1, 22 * 3)
        betas = sequence['betas'][0, :10].detach().cpu().numpy()
        transl = sequence['transl'].detach().cpu().numpy()
    poses = torch.cat([poses, torch.zeros(poses.shape[0], 99).to(dtype=poses.dtype, device=poses.device)],
                      dim=1)
    data_dict = {
        'mocap_framerate': 30,  # 30
        'gender': sequence['gender'] if 'gender' in sequence else 'male',
        'betas': betas,
        'poses': poses.detach().cpu().numpy(),
        'trans': transl,
    }
    with open(out_path, 'wb') as f:
        np.savez(f, **data_dict)

# seq_path_list = [
# # '/home/kaizhao/projects/multiskill/eval_results/inbetween/ours/floor0.0_jerk0.0_use_pred_joints_ddim10_pace_in_circles*15_guidance5.0_seed0/sample_0.pkl',
#     '/home/kaizhao/Desktop/video/goal_reach/run_10.pkl'
#
# ]
seq_path_list = list(Path('/home/kaizhao/Desktop/video/camera ready/').glob('*.pkl'))
# seq_path_list = ['/media/kaizhao/hdd/dataset/proxe/PROXD/MPH11_00151_01/results.pkl']
for seq_path in seq_path_list:
    seq_path = Path(seq_path)
    with open(seq_path, 'rb') as f:
        sequence = pickle.load(f)
    out_path = seq_path.with_suffix('.npz')
    export_smpl(sequence, out_path)
