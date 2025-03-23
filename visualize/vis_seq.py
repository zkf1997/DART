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

from pyrender.trackball import Trackball
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

def get_body_by_batch(primitive_data, body_model, batch_size=256):
    """
        Get SMPLX bodies in batch.
    """
    smplx_input = get_smplx_param_from_6d(primitive_data) if 'poses_6d' in primitive_data else primitive_data
    frame_num = smplx_input['transl'].shape[0]
    last_frame = 0
    vertices = []
    joints = []
    while last_frame < frame_num:
        cur_frame = min(last_frame + batch_size, frame_num)
        smplx_params = {}
        for key in smplx_input:
            if torch.is_tensor(smplx_input[key]):
                smplx_params[key] = smplx_input[key][last_frame:cur_frame, :]
        smplx_output = body_model(**smplx_params)
        vertices += [smplx_output.vertices]
        joints += [smplx_output.joints]
        last_frame = cur_frame
    vertices = torch.cat(vertices, dim=0)
    joints = torch.cat(joints, dim=0)
    return vertices, joints

def makeLookAt(position, target, up=np.array([0, 0, 1])):
    forward = np.subtract(target, position)
    forward = np.divide(forward, np.linalg.norm(forward))

    right = np.cross(forward, up)

    # if forward and up vectors are parallel, right vector is zero;
    #   fix by perturbing up vector a bit
    if np.linalg.norm(right) < 0.001:
        epsilon = np.array([0.001, 0, 0])
        right = np.cross(forward, up + epsilon)

    right = np.divide(right, np.linalg.norm(right))

    up = np.cross(right, forward)
    up = np.divide(up, np.linalg.norm(up))

    return np.array([[right[0], up[0], -forward[0], position[0]],
                     [right[1], up[1], -forward[1], position[1]],
                     [right[2], up[2], -forward[2], position[2]],
                     [0, 0, 0, 1]])

def get_camera_pose(points):
    # Compute the average position of the points (centroid)
    target = np.mean(points, axis=0)
    print('target:', target)

    # Camera position (adjust based on your scene)
    camera_position = np.array([0, 5.0, 5.0])  # Example position for the camera

    # Define the up direction (usually Y-axis is up in most scenes)
    up_direction = np.array([0, 1.0, 0])

    # Compute the direction the camera should look in (from camera to target)
    direction = target - camera_position
    direction = direction / np.linalg.norm(direction)

    # Compute the right direction by crossing the direction and up
    right_direction = np.cross(up_direction, direction)
    right_direction = right_direction / np.linalg.norm(right_direction)

    # Recalculate the true up direction
    true_up = np.cross(direction, right_direction)

    # Create the 4x4 view matrix (camera pose matrix)
    view_matrix = np.eye(4)
    view_matrix[:3, 0] = right_direction
    view_matrix[:3, 1] = true_up
    view_matrix[:3, 2] = -direction  # The camera looks in the negative Z direction
    view_matrix[:3, 3] = camera_position

    return view_matrix

center=np.array([0.0, 5., 2.0])
up=np.array([0, 0.0, 1.0])
def vis_primitive_list(primitive_data_list, args):
    num_sequences = len(primitive_data_list)
    for primitive_data in primitive_data_list:
        tensor_dict_to_device(primitive_data, device)
    primitive_data = primitive_data_list[0]
    scene = pyrender.Scene()
    history_length = primitive_data.get('history_length', 2)
    future_length = primitive_data.get('future_length', 8)
    if 'goal_location_list' not in primitive_data:
        args.vis_goal = 0
    if 'scene_path' in primitive_data:
        scene_path = primitive_data['scene_path']
        scene_mesh = trimesh.load(scene_path)
        scene.add(pyrender.Mesh.from_trimesh(scene_mesh, smooth=False))
    texts = primitive_data.get('texts', [''])
    print(texts)
    if args.follow_camera:
        camera = pyrender.camera.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.414)
        camera_pose = makeLookAt(position=center, target=np.array([0.0, 0, 0]), up=up)
        camera_node = pyrender.Node(camera=camera, name='camera', matrix=camera_pose)
        scene.add_node(camera_node)
    mocap_framerate = int(primitive_data.get('mocap_framerate', 30))
    print('fps:', mocap_framerate)
    viewer = pyrender.Viewer(scene, use_raymond_lighting=True, run_in_thread=True,
                             window_title='_'.join(texts),
                             record=False)
    axis_node = pyrender.Node(mesh=pyrender.Mesh.from_trimesh(trimesh.creation.axis(), smooth=False), name='axis')
    viewer.render_lock.acquire()
    scene.add_node(axis_node)
    viewer.render_lock.release()

    gender = str(primitive_data['gender'])
    if args.body_type == 'smplx':
        body_model = smplx.build_layer(body_model_dir, model_type='smplx',
                          gender=gender, ext='npz',
                          num_pca_comps=12).to(device).eval()
    else:
        body_model = smplx.build_layer(body_model_dir, model_type='smplh',
                                          gender='male', ext='pkl',
                                          num_pca_comps=12).to(device).eval()

    vertices_list = [get_body_by_batch(primitive_data, body_model)[0] for primitive_data in primitive_data_list]
    joints_list = []
    for idx, primitive_data in enumerate(primitive_data_list):
        if args.use_pred_joints and 'joints' in primitive_data:
            joints_list.append(primitive_data['joints'].reshape(-1, 22, 3))
        else:
            joints_list.append(get_body_by_batch(primitive_data, body_model)[1][:, :22, :])

    faces = body_model.faces  # (F, 3)
    joint_parents = body_model.parents
    max_frame_num = max([joints.shape[0] for joints in joints_list])
    # pad vertices and joints with last frame to max_frame_num
    for i in range(num_sequences):
        vertices_list[i] = torch.cat([vertices_list[i], vertices_list[i][-1].unsqueeze(0).repeat(max_frame_num - vertices_list[i].shape[0], 1, 1)], dim=0).detach().cpu().numpy()
        joints_list[i] = torch.cat([joints_list[i], joints_list[i][-1].reshape(1, -1, 3).repeat(max_frame_num - joints_list[i].shape[0], 1, 1)], dim=0).detach().cpu().numpy()
    num_frames = max_frame_num

    # add translation to vertices and joints, avoid body overlapping
    if args.translate_body:
        for seq_idx in range(num_sequences):
            vertices_list[seq_idx] = vertices_list[seq_idx] + np.array([0, 0, seq_idx * 2])
            joints_list[seq_idx] = joints_list[seq_idx] + np.array([0, 0, seq_idx * 2])

    if args.add_floor:
        floor_height = joints_list[0][0, :, 2].min()
        floor = trimesh.creation.box(extents=np.array([20, 20, 0.01]),
                                     transform=np.array([[1.0, 0.0, 0.0, 0],
                                                         [0.0, 1.0, 0.0, 0],
                                                         [0.0, 0.0, 1.0, floor_height - 0.005],
                                                         [0.0, 0.0, 0.0, 1.0],
                                                         ]),
                                     )
        floor.visual.vertex_colors = [0.8, 0.8, 0.8]
        floor_node = pyrender.Node(mesh=pyrender.Mesh.from_trimesh(floor), name='floor')
        viewer.render_lock.acquire()
        scene.add_node(floor_node)
        viewer.render_lock.release()

    if 'traj_ply' in primitive_data and args.vis_traj:
        traj_ply = trimesh.load(primitive_data['traj_ply'])
        if args.translate_body:
            poses = np.stack([np.eye(4) for _ in range(num_sequences)])
            poses[:, :3, 3] = np.array([[0, 0, seq_idx * 2] for seq_idx in range(num_sequences)])
            traj_meshes = pyrender.Mesh.from_trimesh(traj_ply, poses=poses, smooth=False)
            traj_node = pyrender.Node(mesh=traj_meshes, name='traj')
        else:
            traj_mesh = pyrender.Mesh.from_trimesh(traj_ply, smooth=False)
            traj_node = pyrender.Node(mesh=traj_mesh, name='traj')
        viewer.render_lock.acquire()
        scene.add_node(traj_node)
        viewer.render_lock.release()

    if 'goal_body' in primitive_data:
        goal_joints = primitive_data['goal_body']['joints'].reshape(22, 3).detach().cpu().numpy()
        bones = []
        joint_poses = []
        for joint_idx in range(22):
            joint = goal_joints[joint_idx]
            pose = np.eye(4)
            pose[:3, 3] = joint
            joint_poses.append(pose)
            if joint_idx == 0:
                continue
            joint_parent = goal_joints[joint_parents[joint_idx]]
            bones.append(pyrender.Primitive(positions=[joint, joint_parent], color_0=[0, 0, 255], mode=pyrender.constants.GLTF.LINES))

        joint_mesh = trimesh.creation.uv_sphere(radius=0.03)
        joint_mesh.visual.vertex_colors = np.array([1.0, 0.0, 0.0, 1.0])
        joint_mesh = pyrender.Mesh.from_trimesh(joint_mesh, poses=np.stack(joint_poses), smooth=False)
        bone_mesh = pyrender.Mesh(bones)
        joint_node = pyrender.Node(mesh=joint_mesh, name='goal_joint')
        bone_node = pyrender.Node(mesh=bone_mesh, name='goal_bone')
        viewer.render_lock.acquire()
        scene.add_node(joint_node)
        scene.add_node(bone_node)
        viewer.render_lock.release()

    if 'pelvis_traj' in primitive_data:
        pelvis_traj = primitive_data['pelvis_traj']
        pelvis_mesh = trimesh.creation.uv_sphere(radius=0.02)
        pelvis_mesh.visual.vertex_colors = np.array([0.0, 1.0, 0.0, 1.0])
        pelvis_poses = np.stack([np.eye(4) for _ in range(pelvis_traj.shape[0])])
        pelvis_poses[:, :3, 3] = pelvis_traj.detach().cpu().numpy()
        pelvis_mesh = pyrender.Mesh.from_trimesh(pelvis_mesh, poses=pelvis_poses, smooth=False)
        pelvis_node = pyrender.Node(mesh=pelvis_mesh, name='pelvis_gt')
        viewer.render_lock.acquire()
        scene.add_node(pelvis_node)
        viewer.render_lock.release()

    body_node = None
    skeleton_node = None
    joint_node = None
    bone_node = None
    goal_node = None
    frame_idx = args.start_frame
    while True:
        mp_idx = max(frame_idx - history_length, 0) // future_length
        if 'text_idx' in primitive_data:  # per frame text label
            text_idx = primitive_data['text_idx'][frame_idx]
            text = texts[text_idx]
        elif len(texts) == 1:
            text = texts[0]
        else:
            text = texts[mp_idx]
        if 'goal_location_idx' in primitive_data:
            goal_location_idx = primitive_data['goal_location_idx'][frame_idx]
            goal_location = primitive_data['goal_location_list'][goal_location_idx].numpy()
        else:
            goal_location = None
        print('frame_idx:', frame_idx, ', mp_idx:', mp_idx, ', text:', text, ', goal:', goal_location)
        body_mesh_list = []
        joint_poses = []
        bones = []
        for seq_idx in range(num_sequences):
            vertices = vertices_list[seq_idx]
            joints = joints_list[seq_idx]

            body_mesh = trimesh.Trimesh(vertices=vertices[frame_idx], faces=faces, process=False)
            body_mesh.visual.vertex_colors[:, 3] = 130
            body_mesh.visual.vertex_colors[:, :3] = np.asarray(np.asarray(scalarMap.to_rgba(seq_idx / num_sequences)[:3]) * 255, dtype=np.uint8)
            body_mesh_list.append(body_mesh)

            for joint_idx in range(joints.shape[1]):
                joint = joints[frame_idx, joint_idx]
                pose = np.eye(4)
                pose[:3, 3] = joint
                joint_poses.append(pose)
                if joint_idx == 0:
                    continue
                joint_parent = joints[frame_idx, joint_parents[joint_idx]]
                bones.append(pyrender.Primitive(positions=[joint, joint_parent], color_0=[0, 0, 255], mode=pyrender.constants.GLTF.LINES))

        body_mesh = trimesh.util.concatenate(body_mesh_list)
        joint_mesh = trimesh.creation.uv_sphere(radius=0.03)
        joint_mesh.visual.vertex_colors = np.array([0.0, 0.0, 1.0, 1.0])
        joint_mesh = pyrender.Mesh.from_trimesh(joint_mesh, poses=np.stack(joint_poses), smooth=False)
        bone_mesh = pyrender.Mesh(bones)
        if args.vis_goal:
            assert goal_location is not None
            goal_mesh = trimesh.creation.uv_sphere(radius=0.03)
            goal_mesh.visual.vertex_colors = np.array([1.0, 0.0, 0.0, 1.0])
            goal_pose = np.eye(4)
            goal_pose[:3, 3] = goal_location
            goal_mesh = pyrender.Mesh.from_trimesh(goal_mesh, poses=np.stack([goal_pose]), smooth=False)

        viewer.render_lock.acquire()
        if args.vis_goal:
            if goal_node is not None:
                scene.remove_node(goal_node)
            goal_node = pyrender.Node(mesh=goal_mesh, name='goal')
            scene.add_node(goal_node)
        if args.vis_mesh:
            if body_node is not None:
                scene.remove_node(body_node)
            body_node = pyrender.Node(mesh=pyrender.Mesh.from_trimesh(body_mesh, smooth=False), name='body')
            scene.add_node(body_node)
        if args.vis_joint:
            if joint_node is not None:
                scene.remove_node(joint_node)
            joint_node = pyrender.Node(mesh=joint_mesh, name='joint')
            scene.add_node(joint_node)
            if bone_node is not None:
                scene.remove_node(bone_node)
            bone_node = pyrender.Node(mesh=bone_mesh, name='bone')
            scene.add_node(bone_node)
        if args.follow_camera:
            camera_pose = makeLookAt(position=center, target=joints_list[0][frame_idx, 0], up=up)
            camera_pose_current = viewer._camera_node.matrix
            camera_pose_current[:, :] = camera_pose
            # camera_pose_current[:3, 3] = joints_list[0][frame_idx][0] + np.array([0, 0, 5])
            viewer._trackball = Trackball(camera_pose_current, viewer.viewport_size, 1.0)
            # not sure why _scale value of 1500.0 but panning is much smaller if not set to this ?!?
            # your values may be different based on scale and world coordinates
            viewer._trackball._scale = 1500.0
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--slow_rate', type=int, default=1, help='slow down the visualization by this factor')
    parser.add_argument('--start_frame', type=int, default=0, help='start playing animation from this frame index')
    parser.add_argument('--interactive', type=int, default=0, help='whether to enable the interactive mode. If not, the animation will loop automatically. If set to 1, you can press "a" to go back, "d" to go forward, and "q" to quit.')
    parser.add_argument('--max_seq', type=int, default=4, help='maximum number of sequences to visualize')
    parser.add_argument('--seq_path', type=str, help='path to the sequence pkl files, support regular expression')
    parser.add_argument('--vis_joint', type=int, default=1, help='whether to visualize the joints')
    parser.add_argument('--vis_mesh', type=int, default=1, help='whether to visualize the body mesh')
    parser.add_argument('--vis_traj', type=int, default=1, help='whether to visualize the traj mesh')
    parser.add_argument('--vis_goal', type=int, default=1, help='whether to visualize the goals')
    parser.add_argument('--add_floor', type=int, default=0, help='whether to add a floor aligned with the first frame lower foot')
    parser.add_argument('--use_pred_joints', type=int, default=0, help='whether to use the predicted joints instead of SMPL model regression')
    parser.add_argument('--translate_body', type=int, default=0, help='if set to 1, translate the different sequences vertically to avoid overlapping')
    parser.add_argument('--follow_camera', type=int, default=0, help='if set to 1, the camera will follow the body automatically')
    parser.add_argument('--body_type', type=str, default='smplx', help='the type of body to use. For generations using HML3D checkpoints, please select smplh.', choices=['smplx', 'smplh'])
    args = parser.parse_args()
    args.device = device

    primitive_data_list = []
    for seq_path in sorted(list(glob.glob(args.seq_path)))[:args.max_seq]:
        print(seq_path)
        with open(seq_path, 'rb') as f:
            data = pickle.load(f)
        primitive_data_list.append(data)
    vis_primitive_list(primitive_data_list, args)

