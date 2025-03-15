import torch
import numpy as np
import pickle
from pathlib import Path
from pytorch3d import transforms

dataset_path = Path('data/mp_data/Canonicalized_h2_f8_num1_fps30/')
dataset = []
for split in ['train', 'val']:
    data_path = dataset_path / f'{split}.pkl'
    with open(data_path, 'rb') as f:
        dataset += pickle.load(f)

mean_std = {}
frame_features = {}

# keys = ['transl', 'poses', 'joints', 'joints_delta', 'transl_delta', 'global_orient_delta']
# for key in keys:
#     frame_features[key] = torch.from_numpy(np.concatenate([data[key] for data in dataset], axis=0)).to(dtype=torch.float32)
# frame_features['joints'] = frame_features['joints'].reshape(-1, 22 * 3)
# frame_features['joints_delta'] = frame_features['joints_delta'].reshape(-1, 22 * 3)
# frame_features['poses_6d'] = transforms.matrix_to_rotation_6d(transforms.axis_angle_to_matrix(frame_features['poses'].reshape(-1, 22, 3))).reshape(-1, 22 * 6)
# frame_features['global_orient_delta_6d'] = transforms.matrix_to_rotation_6d(transforms.axis_angle_to_matrix(frame_features['global_orient_delta'].reshape(-1, 3)))

keys = ['transl', 'poses_6d', 'joints', 'joints_delta', 'transl_delta', 'global_orient_delta_6d']
for key in keys:
    frame_features[key] = torch.from_numpy(np.concatenate([data[key] for data in dataset], axis=0)).to(dtype=torch.float32)

for key in frame_features:
    mean_std[key] = {}
    print(key, frame_features[key].shape)
    mean_std[key]['std'], mean_std[key]['mean'] = torch.std_mean(frame_features[key], dim=0, keepdim=True)

out_path = dataset_path / 'mean_std.pkl'
with open(out_path, 'wb') as f:
    pickle.dump(mean_std, f)
