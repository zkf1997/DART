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
import glob
from pytorch3d import transforms
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib import cm as cmx
jet = plt.get_cmap('twilight')
cNorm = colors.Normalize(vmin=0, vmax=1)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

from config_files.data_paths import *
from utils.smpl_utils import *


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


def vis_joints(joints, args):
    scene = pyrender.Scene()
    viewer = pyrender.Viewer(scene, use_raymond_lighting=True, run_in_thread=True,
                             record=False)
    axis_node = pyrender.Node(mesh=pyrender.Mesh.from_trimesh(trimesh.creation.axis(), smooth=False), name='axis')
    viewer.render_lock.acquire()
    scene.add_node(axis_node)
    viewer.render_lock.release()

    gender = 'male'
    body_model = smplx.build_layer(body_model_dir, model_type='smplx',
                      gender=gender, ext='npz',
                      num_pca_comps=12).to(device).eval()


    joints_list = [joints]
    num_frames = joints.shape[0]
    mocap_framerate = 30
    num_sequences = len(joints_list)

    faces = body_model.faces  # (F, 3)
    joint_parents = body_model.parents
    max_frame_num = max([joints.shape[0] for joints in joints_list])


    body_node = None
    skeleton_node = None
    joint_node = None
    bone_node = None
    goal_node = None
    frame_idx = args.start_frame
    while True:
        joint_poses = []
        bones = []
        # t0 = time.time()
        for seq_idx in range(num_sequences):
            joints = joints_list[seq_idx]
            for joint_idx in range(joints.shape[1]):
                joint = joints[frame_idx, joint_idx]
                pose = np.eye(4)
                pose[:3, 3] = joint
                joint_poses.append(pose)
                if joint_idx == 0:
                    continue
                joint_parent = joints[frame_idx, joint_parents[joint_idx]]
                bones.append(pyrender.Primitive(positions=[joint, joint_parent], color_0=[0, 0, 255], mode=pyrender.constants.GLTF.LINES))
        joint_mesh = trimesh.creation.uv_sphere(radius=0.03)
        joint_mesh.visual.vertex_colors = np.array([0.0, 0.0, 1.0, 1.0])
        joint_mesh = pyrender.Mesh.from_trimesh(joint_mesh, poses=np.stack(joint_poses), smooth=False)
        # print('joint mesh time:', time.time() - t5)
        bone_mesh = pyrender.Mesh(bones)

        viewer.render_lock.acquire()
        if joint_node is not None:
            scene.remove_node(joint_node)
        joint_node = pyrender.Node(mesh=joint_mesh, name='joint')
        scene.add_node(joint_node)
        if bone_node is not None:
            scene.remove_node(bone_node)
        bone_node = pyrender.Node(mesh=bone_mesh, name='bone')
        scene.add_node(bone_node)
        viewer.render_lock.release()
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
            time.sleep(1 / mocap_framerate * args.slow_rate)
            frame_idx = (frame_idx + 1) % num_frames
            # if frame_idx >= num_frames:
            #     break

device = 'cuda'

parser = argparse.ArgumentParser()
parser.add_argument('--slow_rate', type=int, default=1)
parser.add_argument('--start_frame', type=int, default=0)
parser.add_argument('--interactive', type=int, default=0)
parser.add_argument('--max_seq', type=int, default=4)
parser.add_argument('--seq_path', type=str)

args = parser.parse_args()
args.device = device


joints = np.load(args.seq_path)
vis_joints(joints[5], args)

