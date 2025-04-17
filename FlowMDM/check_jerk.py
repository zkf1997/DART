import numpy as np
import smplx
from pathlib import Path
import pickle
import torch

model_path = "/home/kaizhao/dataset/models_smplx_v1_1/models"
gender = "male"


def calculate_jerk(poses, lengths=None):
    """
    Compute the jerk of a movement represented by a sequence of poses.

    Args:
    - poses (np.ndarray): An ...(BS)xNx22x3 array representing N poses of 22 joints with 3 coordinates each.
    - lengths (np.ndarray): An ...(BS) array representing the length of each pose sequence.

    Returns:
    - float: The average jerk across all joints and coordinates.
    """
    # Calculate the third derivative of the poses with respect to time to get the jerk
    vel = poses[:, 1:] - poses[:, :-1]  # --> ... x N-1 x 22 x 3
    acc = vel[:, 1:] - vel[:, :-1]  # --> ... x N-2 x 22 x 3
    jerk = acc[:, 1:] - acc[:, :-1]  # --> ... x N-3 x 22 x 3

    # for i in range(jerk.shape[0]):
    #     # remove last frames of each batch element according to its length (they have wrong values for jerk)
    #     vel[i, lengths[i] - 1:] = 0
    #     acc[i, lengths[i] - 2:] = 0
    #     jerk[i, lengths[i] - 3:] = 0

    # compute L1 norm of jerk
    jerk = np.sum(np.abs(jerk), axis=-1)  # --> ... x N x 22

    # Get the max of the jerk across all joints
    jerk = np.max(np.abs(jerk), axis=(-1))  # --> ...

    return jerk

feat_p = 'BMLmovi/BMLmovi/Subject_83_F_MoSh/Subject_83_F_9_poses.npz'
start_frame = 30
end_frame = 150

with open('/home/kaizhao/projects/multiskill/data/seq_data/babel_train.pkl', 'rb') as f:
    dataset = pickle.load(f)
data = [data for data in dataset if data['feat_p'] == feat_p][0]['motion']
full_poses = torch.tensor(data['poses'], dtype=torch.float32)
num_frames = full_poses.shape[0]
betas = torch.tensor(data['betas'][:10], dtype=torch.float32).expand(num_frames, 10)
full_trans = torch.tensor(data['trans'], dtype=torch.float32)
global_orient = full_poses[:, 0:3]
body_pose = full_poses[:, 3:66]
transl = full_trans[:, :]
body_model = smplx.create(model_path=model_path,
                             model_type='smplx',
                             gender=gender,
                             use_pca=False,
                             batch_size=num_frames)
output = body_model(global_orient=global_orient, body_pose=body_pose, betas=betas, transl=transl,
                    return_verts=True)
joints = output.joints[start_frame:end_frame, :22].detach().numpy()  # --> [T, 22, 3]
joints = joints.reshape(-1, 30, 22, 3)  # split to chunks of 30 frames

GT_jerk = 0.016383045
batch_jerk = calculate_jerk(joints) # --> [BS, SEQ]
seq_jerks = batch_jerk.mean(axis=0) # --> [SEQ] --> mean jerk per frame in the seq
diff = seq_jerks - GT_jerk
auj = np.sum(np.abs(diff))  # Area Under Jerk Curve
jerk_max = seq_jerks.max()  # Jerk --> max jerk along the sequence


print(feat_p, 'Auj:', auj, 'Jerk:', jerk_max)