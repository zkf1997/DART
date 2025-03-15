import torch
import pickle
from pytorch3d import transforms
from data_scripts.process_motion_primitive_babel import *

yup_to_zup = torch.eye(4)
yup_to_zup[:3, :3] = torch.tensor([[1, 0, 0],
                                   [0, 0, -1],
                                   [0, 1, 0]])
with open('data/rest_pose.pkl', 'rb') as f:
    rest_pose = pickle.load(f)
rest_pose = rest_pose.view(1, 63)
rest_motion = torch.cat([transforms.matrix_to_axis_angle(yup_to_zup), rest_pose], dim=-1)