import smplx
import torch
import pickle
import trimesh
import tqdm
import pyrender
import numpy as np
import time
import argparse
import os

from config_files.data_paths import *
from utils.smpl_utils import convert_smpl_aa_to_rotmat

# https://stackoverflow.com/a/20865751/14532053
class _Getch:
    """Gets a single character from standard input.  Does not echo to the screen."""
    def __init__(self):
        try:
            self.impl = _GetchWindows()
        except ImportError:
            self.impl = _GetchUnix()

    def __call__(self):
        char = self.impl()
        if char == '\x03':
            raise KeyboardInterrupt
        elif char == '\x04':
            raise EOFError
        return char

class _GetchUnix:
    def __init__(self):
        import tty
        import sys

    def __call__(self):
        import sys
        import tty
        import termios
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch


class _GetchWindows:
    def __init__(self):
        import msvcrt

    def __call__(self):
        import msvcrt
        return msvcrt.getch()


getch = _Getch()


def vis_primitive(primitive_data, args):
    scene = pyrender.Scene()
    viewer = pyrender.Viewer(scene, use_raymond_lighting=True, run_in_thread=True,
                             window_title=','.join(primitive_data['texts']),
                             record=False)
    axis_node = pyrender.Node(mesh=pyrender.Mesh.from_trimesh(trimesh.creation.axis(), smooth=False), name='axis')
    viewer.render_lock.acquire()
    scene.add_node(axis_node)
    viewer.render_lock.release()

    gender = str(primitive_data['gender'])
    body_model = smplx.build_layer(body_model_dir, model_type='smplx',
                      gender=gender, ext='npz',
                      num_pca_comps=12).to(device).eval()
    body_param = {}
    body_param['transl'] = torch.FloatTensor(primitive_data['transl']).cuda()
    body_param['global_orient'] = torch.FloatTensor(primitive_data['poses'][:, :3]).cuda()
    body_param['betas'] = torch.FloatTensor(primitive_data['betas'][:10]).unsqueeze(0).repeat(
        primitive_data['poses'].shape[0], 1).cuda()
    body_param['body_pose'] = torch.FloatTensor(primitive_data['poses'][:, 3:66]).cuda()
    body_param = convert_smpl_aa_to_rotmat(body_param)
    smplxout = body_model(return_verts=True, **body_param)
    joints = primitive_data['joints']  # (T, J, 3)
    vertices = smplxout.vertices.detach().cpu().numpy() # (T, V, 3)
    faces = body_model.faces  # (F, 3)
    joint_parents = body_model.parents
    num_frames = joints.shape[0]
    body_node = None
    skeleton_node = None
    frame_idx = args.start_frame
    while True:
        body_mesh = trimesh.Trimesh(vertices=vertices[frame_idx], faces=faces, process=False)
        mat = pyrender.MetallicRoughnessMaterial(alphaMode="BLEND",
                                                 baseColorFactor=(1.0, 1.0, 1.0, 0.5),
                                                 metallicFactor=0.0,
                                                 doubleSided=False)
        skeleton_mesh = []
        for joint_idx in range(joints.shape[1]):
            joint = joints[frame_idx, joint_idx]
            joint_mesh = trimesh.creation.uv_sphere(radius=0.03)
            joint_mesh.apply_transform(trimesh.transformations.translation_matrix(joint))
            joint_mesh.visual.vertex_colors = np.array([1.0, 0.0, 0.0, 1.0])
            skeleton_mesh.append(joint_mesh)
            if joint_idx == 0:
                continue
            joint_parent = joints[frame_idx, joint_parents[joint_idx]]
            bone_mesh = trimesh.creation.cylinder(radius=0.01, segment=np.stack([joint, joint_parent]))
            bone_mesh.visual.vertex_colors = np.array([0.0, 0.0, 1.0, 1.0])
            skeleton_mesh.append(bone_mesh)
        skeleton_mesh = trimesh.util.concatenate(skeleton_mesh)
        viewer.render_lock.acquire()
        if body_node is not None:
            scene.remove_node(body_node)
        body_node = pyrender.Node(mesh=pyrender.Mesh.from_trimesh(body_mesh, material=mat, smooth=False), name='body')
        scene.add_node(body_node)
        if skeleton_node is not None:
            scene.remove_node(skeleton_node)
        skeleton_node = pyrender.Node(mesh=pyrender.Mesh.from_trimesh(skeleton_mesh, smooth=False), name='skeleton')
        scene.add_node(skeleton_node)
        viewer.render_lock.release()
        time.sleep(1 / int(primitive_data['mocap_framerate']) * args.slow_rate)
        if args.interactive:
            # Wait for the user to press a key
            command = getch()
            print("You pressed:", command)
            if command == 'q':
                break
            elif command == 'd':
                frame_idx = (frame_idx + 1) % num_frames
            elif command == 'a':
                frame_idx = (frame_idx - 1) % num_frames
            else:
                try:
                    frame_idx = int(command)
                except:
                    pass
        else:
            frame_idx = frame_idx + 1
            if frame_idx >= num_frames:
                break

device = 'cuda'
data_path = 'data/mp_data/Canonicalized_h2_f8_num1_fps30/val.pkl'
parser = argparse.ArgumentParser()
parser.add_argument('--slow_rate', type=int, default=1)
parser.add_argument('--start_frame', type=int, default=0)
parser.add_argument('--interactive', type=int, default=0)
args = parser.parse_args()

with open(data_path, 'rb') as f:
    dataset = pickle.load(f)
vis_primitive(dataset[23], args)
