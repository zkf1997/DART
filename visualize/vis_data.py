import time

import smplx
import torch
import pickle
import trimesh
import tqdm
import pyrender
import argparse
import numpy as np
import pytorch3d
import glob
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', default="/home/kaizhao/dataset/samp/armchair_stageII.pkl")
args = parser.parse_args()

model_path = "/home/kaizhao/dataset/models_smplx_v1_1/models"
gender = "male"
body_model = smplx.create(model_path=model_path,
                             model_type='smplx',
                             gender=gender,
                             use_pca=False,
                             batch_size=1)

def visualize_samp_seq(input_path, noise_mode=None):
    with open(input_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
        print(data.keys())
        framerate = data['mocap_framerate']
        print(framerate)
        full_poses = torch.tensor(data['pose_est_fullposes'], dtype=torch.float32)
        betas = torch.tensor(data['shape_est_betas'][:10], dtype=torch.float32).reshape(1, 10)
        full_trans = torch.tensor(data['pose_est_trans'], dtype=torch.float32)
        print("Number of frames is {}".format(full_poses.shape[0]))

    scene = pyrender.Scene()
    viewer = pyrender.Viewer(scene, use_raymond_lighting=True, run_in_thread=True)
    m_node = None
    for i in tqdm.tqdm(range(0, full_poses.shape[0], 4)):
    # for i in tqdm.tqdm(range(1120, 1480)):
        global_orient = full_poses[i, 0:3].reshape(1, -1)
        body_pose = full_poses[i, 3:66].reshape(1, -1)
        transl = full_trans[i, :].reshape(1, -1)
        output = body_model(global_orient=global_orient, body_pose=body_pose, betas=betas, transl=transl,
                            return_verts=True)
        m = trimesh.Trimesh(vertices=output.vertices.detach().numpy().squeeze(), faces=body_model.faces, process=False)
        viewer.render_lock.acquire()
        if m_node is not None:
            scene.remove_node(m_node)
        m_node = pyrender.Node(mesh=pyrender.Mesh.from_trimesh(m))
        scene.add_node(m_node)
        viewer.render_lock.release()
        time.sleep(1 / 30)


def export_samp(input_path):
    input_path = Path(input_path)
    with open(input_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
        print(data.keys())
        framerate = data['mocap_framerate']
        print(framerate)
        full_poses = torch.tensor(data['pose_est_fullposes'], dtype=torch.float32)
        betas = torch.tensor(data['shape_est_betas'][:10], dtype=torch.float32).reshape(1, 10)
        full_trans = torch.tensor(data['pose_est_trans'], dtype=torch.float32)
        print("Number of frames is {}".format(full_poses.shape[0]))

    smpl_params = {
        'betas': betas,
        'gender': 'male',
        'global_orient': full_poses[::4, 0:3],
        'body_pose': full_poses[::4, 3:66],
        'transl': full_trans[::4],
    }
    output_path = input_path.parent / 'smpl' / input_path.name
    output_path.parent.mkdir(exist_ok=True, parents=True)
    with open(output_path, 'wb') as f:
        pickle.dump(smpl_params, f)

import pytorch3d.transforms
def apply_rot_noise(rot, noise):
    t, d = rot.shape
    rot = pytorch3d.transforms.axis_angle_to_matrix(rot.reshape(-1, 3))
    noise = pytorch3d.transforms.axis_angle_to_matrix(noise.reshape(-1, 3))
    result = torch.matmul(noise, rot)
    return pytorch3d.transforms.matrix_to_axis_angle(result).reshape(t, d)

def visualize_primitive(input_path, noise_mode=None):
    data = np.load(input_path, allow_pickle=True)
    print(data.keys())
    framerate = data['mocap_framerate']
    print(framerate)
    full_poses = torch.tensor(data['poses'], dtype=torch.float32)
    betas = torch.tensor(data['betas'][:10], dtype=torch.float32).reshape(1, 10)
    full_trans = torch.tensor(data['trans'], dtype=torch.float32)
    print("Number of frames is {}".format(full_poses.shape[0]))
    if noise_mode == 'frame':
        rot_noise = torch.normal(mean=0, std=0.1, size=full_poses.shape, dtype=full_poses.dtype)
        full_poses = apply_rot_noise(full_poses, rot_noise)
    elif noise_mode == 'primitive':
        rot_noise = torch.normal(mean=0, std=0.1, size=full_poses[:1, :].shape, dtype=full_poses.dtype).expand(full_poses.shape)
        full_poses = apply_rot_noise(full_poses, rot_noise)

    scene = pyrender.Scene()
    viewer = pyrender.Viewer(scene, use_raymond_lighting=True, run_in_thread=True)
    m_node = None
    for i in tqdm.tqdm(range(full_poses.shape[0])):
    # for i in tqdm.tqdm(range(1120, 1480)):
        global_orient = full_poses[i, 0:3].reshape(1, -1)
        body_pose = full_poses[i, 3:66].reshape(1, -1)
        transl = full_trans[i, :].reshape(1, -1)
        output = body_model(global_orient=global_orient, body_pose=body_pose, betas=betas, transl=transl,
                            return_verts=True)
        m = trimesh.Trimesh(vertices=output.vertices.detach().numpy().squeeze(), faces=body_model.faces, process=False)
        viewer.render_lock.acquire()
        if m_node is not None:
            scene.remove_node(m_node)
        m_node = pyrender.Node(mesh=pyrender.Mesh.from_trimesh(m))
        scene.add_node(m_node)
        viewer.render_lock.release()
        time.sleep(0.025)

def visualize_babel_seq(data):
    fps = 30
    print(data.keys())
    full_poses = torch.tensor(data['poses'], dtype=torch.float32)
    betas = torch.tensor(data['betas'][:10], dtype=torch.float32).reshape(1, 10)
    full_trans = torch.tensor(data['trans'], dtype=torch.float32)
    print("Number of frames is {}".format(full_poses.shape[0]))

    scene = pyrender.Scene()
    viewer = pyrender.Viewer(scene, use_raymond_lighting=True, run_in_thread=True)
    m_node = None
    for i in tqdm.tqdm(range(0, full_poses.shape[0], 1)):
        global_orient = full_poses[i, 0:3].reshape(1, -1)
        body_pose = full_poses[i, 3:66].reshape(1, -1)
        transl = full_trans[i, :].reshape(1, -1)
        output = body_model(global_orient=global_orient, body_pose=body_pose, betas=betas, transl=transl,
                            return_verts=True)
        m = trimesh.Trimesh(vertices=output.vertices.detach().numpy().squeeze(), faces=body_model.faces, process=False)
        viewer.render_lock.acquire()
        if m_node is not None:
            scene.remove_node(m_node)
        m_node = pyrender.Node(mesh=pyrender.Mesh.from_trimesh(m))
        scene.add_node(m_node)
        viewer.render_lock.release()
        time.sleep(1 / fps)
    # viewer.close_external()

def visualize_gamma(data):
    fps = 30
    print(data.keys())
    betas = torch.tensor(data['betas'], dtype=torch.float32).reshape(1, 10)
    transl = torch.tensor(data['transl'], dtype=torch.float32).squeeze(0)
    body_pose = torch.tensor(data['body_pose'], dtype=torch.float32).squeeze(0)  # (T, 63)
    global_orient = torch.tensor(data['global_orient'], dtype=torch.float32).squeeze(0)  # (T, 3)
    num_frames = body_pose.shape[0]

    scene = pyrender.Scene()
    scene.add(pyrender.Mesh.from_trimesh(trimesh.creation.axis(origin_size=0.1, axis_radius=0.001, axis_length=0.5), smooth=False))
    floor = trimesh.creation.box(extents=np.array([20, 20, 0.01]),
                                 transform=np.array([[1.0, 0.0, 0.0, 0],
                                                     [0.0, 1.0, 0.0, 0],
                                                     [0.0, 0.0, 1.0, - 0.005],
                                                     [0.0, 0.0, 0.0, 1.0],
                                                     ]),
                                 )
    floor_node = pyrender.Node(mesh=pyrender.Mesh.from_trimesh(floor), name='floor')
    scene.add_node(floor_node)
    viewer = pyrender.Viewer(scene, use_raymond_lighting=True, run_in_thread=True)
    m_node = None
    for i in tqdm.tqdm(range(0, num_frames)):
        output = body_model(global_orient=global_orient[[i]], body_pose=body_pose[[i]], betas=betas, transl=transl[[i]],
                            return_verts=True)
        m = trimesh.Trimesh(vertices=output.vertices.detach().numpy().squeeze(), faces=body_model.faces, process=False)
        viewer.render_lock.acquire()
        if m_node is not None:
            scene.remove_node(m_node)
        m_node = pyrender.Node(mesh=pyrender.Mesh.from_trimesh(m))
        scene.add_node(m_node)
        viewer.render_lock.release()
        time.sleep(1 / fps)

def visualize_motionx_seq(seq_path):
    data = np.load(seq_path, allow_pickle=True)
    fps = 30
    betas = torch.tensor(data[:, 312:], dtype=torch.float32)
    global_orient = torch.tensor(data[:, :3], dtype=torch.float32)
    body_pose = torch.tensor(data[:, 3:66], dtype=torch.float32)
    transl = torch.tensor(data[:, 309:312], dtype=torch.float32)
    num_frames = body_pose.shape[0]

    scene = pyrender.Scene()
    viewer = pyrender.Viewer(scene, use_raymond_lighting=True, run_in_thread=True)
    m_node = None
    for i in tqdm.tqdm(range(0, num_frames, 1)):
        output = body_model(global_orient=global_orient[[i]], body_pose=body_pose[[i]], betas=betas[[i]], transl=transl[[i]],
                            return_verts=True)
        m = trimesh.Trimesh(vertices=output.vertices.detach().numpy().squeeze(), faces=body_model.faces, process=False)
        viewer.render_lock.acquire()
        if m_node is not None:
            scene.remove_node(m_node)
        m_node = pyrender.Node(mesh=pyrender.Mesh.from_trimesh(m))
        scene.add_node(m_node)
        viewer.render_lock.release()
        time.sleep(1 / fps)

def convert_motionx(input_dir):
    for seq_path in Path(input_dir).glob('*.npy'):
        data = np.load(seq_path, allow_pickle=True)
        output_path = seq_path.parent / 'smpl' / (seq_path.stem + '.npz')
        output_path.parent.mkdir(exist_ok=True, parents=True)
        poses = data[:, :66]
        poses = np.concatenate([poses, np.zeros((poses.shape[0], 99))], axis=1)

        data_dict = {
            'mocap_framerate': 30,  # has to be greater than 30 to be loaded by blender
            'gender': 'male',
            'betas': data[0, 312:],
            'poses': poses,
            'trans': data[:, 309:312],
        }
        with open(output_path, 'wb') as f:
            np.savez(f, **data_dict)

def convert_amass(input_path, output_path):
    data = np.load(input_path, allow_pickle=True)
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    poses = data['poses'][::4, :]
    # poses = np.concatenate([poses, np.zeros((poses.shape[0], 99))], axis=1)

    data_dict = {
        'mocap_framerate': 30,  # has to be greater than 30 to be loaded by blender
        'gender': data['gender'],
        'betas': data['betas'],
        'poses': poses,
        'trans': data['trans'][::4],
    }
    with open(output_path, 'wb') as f:
        np.savez(f, **data_dict)

# input_path = "/home/kaizhao/dataset/amass/gamma/Canonicalized-MPx10/data/sit/subseq_00444.npz"
# print(input_path)
# visualize_samp_seq(args.input_path)
# visualize_primitive(input_path, args.noise_mode)

# export_samp(args.input_path)


# with open('/mnt/atlas_root/vlg-nfs/scratch/genli/neurips2024/scene000_path004_seq000/results_ssm2_67_condi_marker_0.pkl.pkl', 'rb') as f:
#     data = pickle.load(f)
# visualize_gamma(data)

# seq_idx = 639
# with open('./data/seq_data_zero_male/no_slerp/val.pkl', 'rb') as f:
#     dataset = pickle.load(f)
# visualize_babel_seq(dataset[seq_idx]['motion'])
# with open('./data/seq_data_zero_male/val.pkl', 'rb') as f:
#     dataset = pickle.load(f)
# visualize_babel_seq(dataset[seq_idx]['motion'])

# data = np.load('/home/kaizhao/dataset/amass/smplh_g/SSM_synced/20160930_50032/special_move_sync_poses.npz')
# visualize_babel_seq(data)

# with open('/home/kaizhao/projects/multiskill/data/hml3d/seq_data_zero_male/test.pkl', 'rb') as f:
#     test_data = pickle.load(f)
# visualize_babel_seq(test_data[79]['motion'])

# visualize_motionx_seq(args.input_path)
# convert_motionx(args.input_path)

convert_amass('/home/kaizhao/dataset/amass/smplx_g/BMLmovi/Subject_38_F_MoSh/Subject_38_F_21_stageii.npz', '/home/kaizhao/Desktop/video/iclr rebuttal/amass/crawl.npz')
