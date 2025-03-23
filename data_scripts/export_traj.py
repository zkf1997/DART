import numpy as np
import torch
import math
import pdb
import pickle

from os.path import join as pjoin

from utils.smpl_utils import *
from pytorch3d import transforms

device = 'cuda'
primitive_utility = PrimitiveUtility(device=device)
def get_joints(motion, gender='male'):
    body_pose = torch.tensor(motion['poses'][:, 3:], dtype=torch.float32)
    body_pose = transforms.axis_angle_to_matrix(body_pose.reshape(-1, 3)).reshape(-1, 21, 3, 3).unsqueeze(
        0)  # [1, T, 21, 3, 3]
    global_orient = torch.tensor(motion['poses'][:, :3], dtype=torch.float32)
    global_orient = transforms.axis_angle_to_matrix(global_orient.reshape(-1, 3)).reshape(-1, 3, 3).unsqueeze(
        0)  # [1, T, 3, 3]
    transl = torch.tensor(motion['trans'], dtype=torch.float32).unsqueeze(0)  # [1, T, 3]
    seq_length = transl.shape[1]
    betas = torch.tensor(motion['betas'],
                         dtype=torch.float32)
    betas = betas.expand(1, seq_length, 10)  # [1, T, 10]
    seq_dict = {
        'gender': gender,
        'betas': betas,
        'transl': transl,
        'body_pose': body_pose,
        'global_orient': global_orient,
        'transf_rotmat': torch.eye(3).unsqueeze(0),
        'transf_transl': torch.zeros(1, 1, 3),
    }
    seq_dict = tensor_dict_to_device(seq_dict, device)
    _, _, canonicalized_primitive_dict = primitive_utility.canonicalize(seq_dict)
    body_model = primitive_utility.get_smpl_model(gender)
    joints = body_model(return_verts=False,
                        betas=canonicalized_primitive_dict['betas'][0],
                        body_pose=canonicalized_primitive_dict['body_pose'][0],
                        global_orient=canonicalized_primitive_dict['global_orient'][0],
                        transl=canonicalized_primitive_dict['transl'][0]
                        ).joints[:, :22, :]  # [T, 22, 3]
    return joints.detach().cpu().numpy()

# joints:
# 0: pelvis:
# 10: l_foot: kick something
# 11: r_foot: kick something
# 15: head: walk into a tunnel or something
# 20: l_wrist: raise a toolbox, or touch something, or hold something
# 21: r_wrist: raise a toolbox, or touch something, or hold something, or walk with one hand on the handrail


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def circle(n_frames=120, r=0.8, indices=[0, 2], x_offset=0.5, y_offset=0.9, z_offset=0.5):
    hint = np.zeros((n_frames, 3))
    points = sample_points_circle(n_frames, r)
    hint[:, indices] = points
    hint[:, 0] += x_offset
    hint[:, 1] += y_offset
    hint[:, 2] += z_offset
    return hint


def straight(n_frames=120, indices=[1, 2], scale=1.0, x_offset=0.5, y_offset=0.5, z_offset=0.5):
    hint = np.zeros((n_frames, 3))
    points = sample_points_forward(n_frames, scale=scale)
    hint[:, indices] = points
    hint[:, 0] += x_offset
    hint[:, 1] += y_offset
    hint[:, 2] += z_offset
    return hint


def sample_points_on_unit_square_edges(n):
    """
    Samples n points uniformly from the edges of a unit square [0,1] x [0,1] in order.

    Args:
        n (int): The total number of points to sample.

    Returns:
        np.ndarray: An array of shape (n, 2) where each row represents a point (x, y).
    """
    # Divide the points equally among the 4 edges
    n_per_edge = n // 4
    remainder = n % 4

    # Sample points for each edge

    left_edge = np.column_stack((np.zeros(n_per_edge + (1 if remainder > 1 else 0)),
                                    np.linspace(0, 1, n_per_edge + (1 if remainder > 1 else 0))))
    top_edge = np.column_stack((np.linspace(0, 1, n_per_edge + (1 if remainder > 2 else 0)),
                                np.ones(n_per_edge + (1 if remainder > 2 else 0))))
    right_edge = np.column_stack((np.ones(n_per_edge + (1 if remainder > 3 else 0)),
                                    np.linspace(1, 0, n_per_edge + (1 if remainder > 3 else 0))))
    bottom_edge = np.column_stack((np.linspace(1, 0, n_per_edge),
                                    np.zeros(n_per_edge)))


    # Combine the points from all edges
    points = np.vstack((left_edge, top_edge, right_edge, bottom_edge))[:n]

    return points

# writ a function to sample a square path
def square(n_frames=120, indices=[0, 2], scale=1.0, x_offset=0.5, y_offset=0.5, z_offset=0.5):
    hint = np.zeros((n_frames, 3))
    points = sample_points_on_unit_square_edges(n_frames) * scale
    hint[:, indices] = points
    hint[:, 0] += x_offset
    hint[:, 1] += y_offset
    hint[:, 2] += z_offset
    return hint



def specify_points(n_frames=120, points=[[50, 1, 1, 1]]):
    hint = np.zeros((n_frames, 3))
    for point in points:
        hint[point[0]] = point[1:]
    # points = sample_points_forward_uniform(n_frames, scale=scale)
    # hint[:, indices] = np.array(points)[..., np.newaxis]
    # hint[:, 0] += x_offset
    # hint[:, 1] += y_offset
    # hint[:, 2] += z_offset
    return hint


def spiral_forward(n_frames=120, points=[[50, 1, 1, 1]]):
    hint = np.zeros((n_frames, 3))
    # how many times:
    n = 3
    # radius
    r = 0.3
    # offset
    o = 0.9
    # angle step size
    angle_step = 2 * np.pi / (n_frames / n)

    points = []

    start_from = - np.pi / 2

    for i in range(n_frames):
        theta = i * angle_step + start_from

        x = r * np.cos(theta)
        y = r * np.sin(theta) + o
        z = i * 0.02

        points.append((x, y, z))

    hint = np.stack(points)
    return hint


def straight_diagonal_uniform(n_frames=120, indices=[0, 2], scale=1.0, x_offset=0.5, y_offset=0.5, z_offset=0.5):
    hint = np.zeros((n_frames, 3))
    points = sample_points_forward_uniform(n_frames, scale=scale)
    hint[:, indices] = np.array(points)[..., np.newaxis]
    hint[:, 0] += x_offset
    hint[:, 1] += y_offset
    hint[:, 2] += z_offset
    return hint


def straight_forward_step_uniform(n_frames=120, indices=[0, 2], scale=1.0, x_offset=0.5, y_offset=0.5, z_offset=0.5,
                                  step_ratio=0.5):
    hint = np.zeros((n_frames, 3))
    sub_frame = int(n_frames * step_ratio)
    points = sample_points_forward_uniform(n_frames, scale=scale)
    hint[:, indices] = np.array(points)[..., np.newaxis]
    hint[sub_frame:, 1] -= 0.2
    hint[:, 0] += x_offset
    hint[:, 1] += y_offset
    hint[:, 2] += z_offset
    return hint


def straight_forward_uniform(n_frames=120, indices=[0, 2], scale=1.0, x_offset=0.5, y_offset=0.5, z_offset=0.5):
    hint = np.zeros((n_frames, 3))
    points = sample_points_forward_uniform(n_frames, scale=scale)
    hint[:, indices] = np.array(points)[..., np.newaxis]
    hint[:, 0] += x_offset
    hint[:, 1] += y_offset
    hint[:, 2] += z_offset
    return hint


def straight_forward_backward_uniform(n_frames=120, indices=[0, 2], scale=1.0, x_offset=0.5, y_offset=0.5,
                                      z_offset=0.5):
    hint = np.zeros((n_frames, 3))
    sub_frame = n_frames // 2
    points = sample_points_forward_uniform(sub_frame, scale=scale)
    hint[:sub_frame, indices] = np.array(points)[..., np.newaxis]
    hint[sub_frame:, indices] = np.array(points[::-1])[..., np.newaxis]
    hint[sub_frame:, 0] += 0.5
    hint[:, 0] += x_offset
    hint[:, 1] += y_offset
    hint[:, 2] += z_offset
    return hint


def straight_fb(n_frames=120, indices=[1, 2], scale=0.5, x_offset=0.5, y_offset=0.6, z_offset=0.5):
    hint = np.zeros((n_frames, 3))
    points = sample_points_forward_back_verticel(n_frames)
    hint[:, indices] = points
    hint[:, 0] += x_offset
    hint[:, 1] *= scale
    hint[:, 1] += y_offset
    hint[:, 2] += z_offset
    return hint


def s_line(n_frames=120, indices=[1, 2], x_offset=0.5, y_offset=0.6, z_offset=0.5):
    hint = np.zeros((n_frames, 3))
    points = sample_points_s(n_frames)
    hint[:, indices] = points
    hint[:, 0] += x_offset
    hint[:, 1] += y_offset
    hint[:, 2] += z_offset
    return hint


def s_line_long(n_frames=120, indices=[1, 2], x_offset=0.5, y_offset=0.6, z_offset=0.5, scale=1 / 3, scale1=2 / 3):
    hint = np.zeros((n_frames, 3))
    sub_frame = n_frames // 3
    for i in range(3):
        points = sample_points_s(sub_frame, scale, scale1)
        hint[sub_frame * i:sub_frame * (i + 1), indices] = points
        hint[sub_frame * i:sub_frame * (i + 1), 0] += x_offset
        hint[sub_frame * i:sub_frame * (i + 1), 1] += y_offset
        hint[sub_frame * i:sub_frame * (i + 1), 2] += z_offset + 2 * scale * i * np.pi
    return hint


def s_line_middlelong(n_frames=120, indices=[1, 2], x_offset=0.5, y_offset=0.6, z_offset=0.5, scale=1 / 3, scale1=1):
    hint = np.zeros((n_frames, 3))
    sub_frame = n_frames // 2
    for i in range(2):
        points = sample_points_s(sub_frame, scale, scale1)
        hint[sub_frame * i:sub_frame * (i + 1), indices] = points
        hint[sub_frame * i:sub_frame * (i + 1), 0] += x_offset
        hint[sub_frame * i:sub_frame * (i + 1), 1] += y_offset
        hint[sub_frame * i:sub_frame * (i + 1), 2] += z_offset + 2 * scale * i * np.pi
    return hint


def sample_points_circle(n, r=0.8):
    # number of points

    # angle step size
    angle_step = 2 * np.pi / n

    points = []

    start_from = 0

    for i in range(n):
        theta = i * angle_step + start_from

        x = r * np.cos(theta) - r
        y = r * np.sin(theta)

        points.append((x, y))
    return points


def sample_points_s(n, scale=1 / 3, scale1=1):
    # number of points

    # angle step size
    angle_step = 2 * np.pi / n

    points = []

    start_from = 0

    for i in range(n):
        theta = i * angle_step + start_from

        x = theta * scale
        y = np.sin(theta) * scale1

        points.append((x, y))
    return points


def sample_points_forward(n, scale=1.0):
    # number of points

    # angle step size
    angle_step = np.pi / n / 2

    points = []

    start_from = 0

    for i in range(n):
        theta = i * angle_step + start_from

        x = np.sin(theta) * scale
        y = 0

        points.append((x, y))
    return points


def sample_points_forward_uniform(n, scale=1.0):
    # angle step size
    step = scale / n

    points = []

    start_from = 0

    for i in range(n):
        theta = i * step + start_from

        x = theta

        points.append(x)
    return points


def sample_points_forward_back_verticel(n):
    # number of points

    # angle step size
    angle_step = 2 * np.pi / n

    points = []

    start_from = 0

    for i in range(n):
        theta = i * angle_step + start_from

        x = np.cos(theta)
        y = 0

        points.append((x, y))
    return points

import trimesh
from scipy.spatial.transform import Rotation as R
from pathlib import Path
hml3d_to_canonical = R.from_euler('xyz', [90, 0, 180], degrees=True).as_matrix()

def export(traj_hml3d, traj_canonical, text, joint_idx, frame_idx, save_dir):
    save_dir.mkdir(parents=True, exist_ok=True)
    if traj_canonical.shape[0] == 1:
        wpath_mesh = trimesh.creation.uv_sphere(radius=0.02)
        wpath_mesh.vertices += traj_canonical
    else:
        wpath_meshes = []
        for point_idx in range(traj_canonical.shape[0]):
            color = int((point_idx+1) / traj_canonical.shape[0] * 255)
            wpath_mesh = trimesh.creation.uv_sphere(radius=0.02)
            wpath_mesh.vertices += traj_canonical[point_idx]
            wpath_mesh.visual.vertex_colors = np.asarray([color, 0, 0, color])
            wpath_meshes.append(wpath_mesh)
        # for point_idx in range(traj_canonical.shape[0] - 1):
        #     color = int(point_idx / traj_canonical.shape[0] * 255)
        #     wpath_meshes.append(trimesh.creation.cylinder(radius=0.03, segment=np.stack([traj_canonical[point_idx], traj_canonical[point_idx + 1]]),
        #                                                       vertex_colors=np.asarray([color, 0, 0, color])
        #                                                       ))
        wpath_mesh = trimesh.util.concatenate(wpath_meshes)
    wpath_mesh.export(save_dir / 'traj.ply')
    wpath_mesh.export(save_dir / 'traj.obj')
    if traj_hml3d.shape[0] == 1:
        wpath_mesh = trimesh.creation.uv_sphere(radius=0.02)
        wpath_mesh.vertices += traj_hml3d
    else:
        wpath_meshes = []
        for point_idx in range(traj_hml3d.shape[0] - 1):
            color = int(point_idx / traj_hml3d.shape[0] * 255)
            wpath_meshes.append(trimesh.creation.cylinder(radius=0.03, segment=np.stack(
                [traj_hml3d[point_idx], traj_hml3d[point_idx + 1]]),
                                                          vertex_colors=np.asarray([color, 0, 0, color])
                                                          ))
        wpath_mesh = trimesh.util.concatenate(wpath_meshes)
    wpath_mesh.export(save_dir / 'traj_hml3d.ply')
    with open(save_dir / 'traj_text.pkl', 'wb') as f:
        pickle.dump({
            'joint_idx': joint_idx,
            'traj': traj_canonical,
            'frame_idx': frame_idx,
            'text': text}, f)
    with open(save_dir / 'traj_text_hml3d.pkl', 'wb') as f:
        pickle.dump({
            'joint_idx': joint_idx,
            'traj': traj_hml3d,
            'frame_idx': frame_idx,
            'text': text}, f)

fps = 30
duration = 6
n_frames = fps * duration
response_time = 1.5
start_frame = int(response_time * fps)

text = 'wave right hand'
text_nospace = text.replace(' ', '_')
save_dir = Path('data/traj_test') / f'dense_frame{n_frames}_{text_nospace}_circle'
frame_idx = np.arange(n_frames - start_frame) + start_frame
traj_hml3d = circle(n_frames, r=0.25, indices=[0, 1], x_offset=-0.2, y_offset=1.6, z_offset=0.2)
traj_hml3d = traj_hml3d[frame_idx]
traj_canonical = np.dot(traj_hml3d, hml3d_to_canonical.T)
export(traj_hml3d, traj_canonical, text, joint_idx=21, frame_idx=frame_idx, save_dir=save_dir)

text = 'punch'
text_nospace = text.replace(' ', '_')
save_dir = Path('data/traj_test') / f'sparse_{text_nospace}'
frame_idx = np.array([45])
traj_hml3d = np.array([[0, 1.5, 0.5]])
traj_canonical = np.dot(traj_hml3d, hml3d_to_canonical.T)
export(traj_hml3d, traj_canonical, text, joint_idx=20, frame_idx=frame_idx, save_dir=save_dir)

text = 'walk'
text_nospace = text.replace(' ', '_')
save_dir = Path('data/traj_test') / f'sparse_frame{n_frames}_{text_nospace}_square'
frame_idx = np.linspace(0, n_frames - 1, 5).astype(int)[1:]
frame_idx = frame_idx[frame_idx>=start_frame]
traj_hml3d = square(n_frames, scale=duration * 1 / 4, indices=[0, 2], x_offset=0.0, y_offset=0.9, z_offset=0.0)
traj_hml3d = traj_hml3d[frame_idx]
traj_canonical = np.dot(traj_hml3d, hml3d_to_canonical.T)
export(traj_hml3d, traj_canonical, text, joint_idx=0, frame_idx=frame_idx, save_dir=save_dir)

text = 'walk'
text_nospace = text.replace(' ', '_')
save_dir = Path('data/traj_test') / f'dense_frame{n_frames}_{text_nospace}_circle'
frame_idx = np.arange(n_frames)
frame_idx = frame_idx[frame_idx>=start_frame]
traj_hml3d = circle(n_frames, r=duration * 1 / 2 / np.pi, indices=[0, 2], x_offset=0.0, y_offset=0.9, z_offset=0.0)
traj_hml3d = traj_hml3d[frame_idx]
traj_canonical = np.dot(traj_hml3d, hml3d_to_canonical.T)
export(traj_hml3d, traj_canonical, text, joint_idx=0, frame_idx=frame_idx, save_dir=save_dir)


