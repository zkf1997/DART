from pytorch3d import transforms
from copy import deepcopy
import torch
import smplx
from typing import Tuple
from typing import Union

from config_files.data_paths import *

body_model_dict = {
    'male': smplx.build_layer(body_model_dir, model_type='smplx',
                              gender='male', ext='npz',
                              num_pca_comps=12),
    'female': smplx.build_layer(body_model_dir, model_type='smplx',
                                gender='female', ext='npz',
                                num_pca_comps=12
                                )
}

def tensor_dict_to_device(tensor_dict, device, dtype=torch.float32):
    for k in tensor_dict:
        if isinstance(tensor_dict[k], torch.Tensor):
            tensor_dict[k] = tensor_dict[k].to(device=device)
    return tensor_dict


def convert_smpl_aa_to_rotmat(smplx_param):
    smplx_param = deepcopy(smplx_param)
    smplx_param['global_orient'] = transforms.axis_angle_to_matrix(smplx_param['global_orient'])
    smplx_param['body_pose'] = transforms.axis_angle_to_matrix(smplx_param['body_pose'].reshape(-1, 3)).reshape(-1, 21,
                                                                                                                3, 3)
    return smplx_param


def get_smplx_param_from_6d(primitive_data):
    body_param = {}
    if 'gender' in primitive_data:
        body_param['gender'] = primitive_data['gender']
    batch_size = primitive_data['transl'].shape[0]
    body_param['transl'] = primitive_data['transl']
    if len(primitive_data['betas'].shape) == 1:
        body_param['betas'] = primitive_data['betas'][:10].unsqueeze(0).repeat(batch_size, 1)
    else:
        body_param['betas'] = primitive_data['betas'][:, :10]
    global_orient = primitive_data['poses_6d'][:, :6]
    global_orient = transforms.rotation_6d_to_matrix(global_orient).reshape(-1, 3, 3)
    body_pose = primitive_data['poses_6d'][:, 6:132]
    body_pose = transforms.rotation_6d_to_matrix(body_pose.reshape(-1, 6)).reshape(-1, 21, 3, 3)
    body_param['global_orient'] = global_orient
    body_param['body_pose'] = body_pose
    return body_param


def get_new_coordinate(jts: torch.Tensor):
    x_axis = jts[:, 2, :] - jts[:, 1, :]  # [b,3]
    x_axis[:, -1] = 0
    x_axis = x_axis / torch.norm(x_axis, dim=-1, keepdim=True)
    z_axis = torch.FloatTensor([[0, 0, 1]]).to(jts.device).repeat(x_axis.shape[0], 1)
    y_axis = torch.cross(z_axis, x_axis, dim=-1)
    y_axis = y_axis / torch.norm(y_axis, dim=-1, keepdim=True)
    new_rotmat = torch.stack([x_axis, y_axis, z_axis], dim=-1)  # [b,3,3]
    new_transl = jts[:, :1]  # [b,1,3]
    return new_rotmat, new_transl


def update_global_transform(body_param_dict, new_rotmat, new_transl):
    """update the global human to world transform using the transformation from new coord axis to old coord axis"""
    old_rotmat = body_param_dict['transf_rotmat']
    old_transl = body_param_dict['transf_transl']
    body_param_dict['transf_rotmat'] = torch.einsum('bij,bjk->bik', old_rotmat, new_rotmat)  # [b,3,3]
    body_param_dict['transf_transl'] = torch.einsum('bij,btj->bti', old_rotmat, new_transl) + old_transl  # [b,1,3]
    return body_param_dict


def transform_local_points_to_global(local_points, transf_rotmat, transf_transl):
    """
    :param local_points: [B, N, 3]
    :param transf_rotmat: [B, 3, 3]
    :param transf_transl: [B, 1, 3]
    :return:
    """
    global_points = torch.einsum('bij,bkj->bki', transf_rotmat, local_points) + transf_transl
    return global_points


def transform_global_points_to_local(global_points, transf_rotmat, transf_transl):
    """
    :param global_points: [B, N, 3]
    :param transf_rotmat: [B, 3, 3]
    :param transf_transl: [B, 1, 3]
    :return:
    """
    local_points = torch.einsum('bij,bki->bkj', transf_rotmat, global_points - transf_transl)
    # local_points = torch.einsum('bij,bkj->bki', transf_rotmat.permute(0, 2, 1), global_points - transf_transl)
    return local_points


def get_dict_subset_by_batch(dict_data, batch_idx):
    new_dict = {}
    for key in dict_data:
        if key == 'gender':
            new_dict[key] = dict_data[key]
        else:
            new_dict[key] = dict_data[key][batch_idx]
    return new_dict

class PrimitiveUtility:
    def __init__(self, device='cpu', dtype=torch.float32, motion_repr=None, body_type='smplx'):
        self.device = device
        self.dtype = dtype
        self.motion_repr = {
            'transl': 3,
            'poses_6d': 22 * 6,
            'transl_delta': 3,
            'global_orient_delta_6d': 6,
            'joints': 22 * 3,
            'joints_delta': 22 * 3,
        }
        feature_dim = 0
        for k in self.motion_repr:
            feature_dim += self.motion_repr[k]
        self.feature_dim = feature_dim
        self.body_type = body_type
        if body_type == 'smplx':
            self.bm_male = body_model_dict['male'].to(self.device).eval()
            self.bm_female = body_model_dict['female'].to(self.device).eval()
        else:
            smplh_body_model_dict = {
                'male': smplx.build_layer(body_model_dir, model_type='smplh',
                                          gender='male', ext='pkl',
                                          num_pca_comps=12),
                'female': smplx.build_layer(body_model_dir, model_type='smplh',
                                            gender='female', ext='pkl',
                                            num_pca_comps=12
                                            )
            }
            self.bm_male = smplh_body_model_dict['male'].to(self.device).eval()
            self.bm_female = smplh_body_model_dict['female'].to(self.device).eval()

    def get_smpl_model(self, gender):
        return self.bm_male if gender == 'male' else self.bm_female

    def dict_to_tensor(self, data_dict):
        tensors = [data_dict[key] for key in self.motion_repr]
        merged_tensor = torch.cat(tensors, dim=-1)  # (B, [T], 22*3+22*3+3+3+6+22*6)
        return merged_tensor

    def tensor_to_dict(self, tensor):
        data_dict = {}
        start = 0
        for key in self.motion_repr:
            end = start + self.motion_repr[key]
            data_dict[key] = tensor[..., start:end]
            start = end
        return data_dict

    def feature_dict_to_smpl_dict(self, feature_dict):
        body_param = {
            'gender': feature_dict['gender'],
            'betas': feature_dict['betas'],
            'transf_rotmat': feature_dict['transf_rotmat'],
            'transf_transl': feature_dict['transf_transl'],
            'transl': feature_dict['transl'],
            'joints': feature_dict['joints'],  # network predicted joints
        }
        if 'pelvis_delta' in feature_dict:
            body_param['pelvis_delta'] = feature_dict['pelvis_delta']

        # print(feature_dict['poses_6d'].shape, feature_dict['transl'].shape)
        prefix_shape = feature_dict['poses_6d'].shape[:-1]
        global_orient = feature_dict['poses_6d'][..., :6]
        global_orient = transforms.rotation_6d_to_matrix(global_orient)
        body_pose = feature_dict['poses_6d'][..., 6:132].reshape(*prefix_shape, 21, 6)
        body_pose = transforms.rotation_6d_to_matrix(body_pose).reshape(*prefix_shape, 21, 3, 3)
        body_param['global_orient'] = global_orient
        body_param['body_pose'] = body_pose
        return body_param

    def smpl_dict_to_vertices(self, body_param):
        gender = body_param['gender']
        body_model = self.bm_male if gender == 'male' else self.bm_female
        assert len(body_param['transl'].shape) == 3  # [B, T, 3]
        B, T, _ = body_param['transl'].shape
        vertices = body_model(betas=body_param['betas'].reshape(B * T, 10),
                              global_orient=body_param['global_orient'].reshape(B * T, 3, 3),
                              body_pose=body_param['body_pose'].reshape(B * T, 21, 3, 3),
                              transl=body_param['transl'].reshape(B * T, 3)
                              ).vertices
        vertices = vertices.reshape(B, T, -1, 3)
        return vertices

    def smpl_dict_inference(self, body_param, return_vertices=False, batch_size=512):
        # input body_param: T x D, no batch dimension
        body_model = self.bm_male if body_param['gender'] == 'male' else self.bm_female
        T, _ = body_param['transl'].shape
        vertices = []
        joints = []
        batch_start_idx = 0
        while batch_start_idx < T:
            batch_end_idx = min(batch_start_idx + batch_size, T)
            smplx_out = body_model(betas=body_param['betas'][batch_start_idx:batch_end_idx],
                                   global_orient=body_param['global_orient'][batch_start_idx:batch_end_idx],
                                   body_pose=body_param['body_pose'][batch_start_idx:batch_end_idx],
                                   transl=body_param['transl'][batch_start_idx:batch_end_idx],
                                   return_vertices=return_vertices
                                   )
            joints.append(smplx_out.joints[:, :22])
            if return_vertices:
                vertices.append(smplx_out.vertices)
            batch_start_idx = batch_end_idx

        joints = torch.cat(joints, dim=0)
        if return_vertices:
            vertices = torch.cat(vertices, dim=0)
            return joints, vertices
        else:
            return joints


    def get_new_coordinate(self, body_param_dict, use_predicted_joints=False, pred_joints=None):
        if use_predicted_joints:
            joints = pred_joints
        else:
            body_model = self.bm_male if body_param_dict['gender'] == 'male' else self.bm_female
            joints = body_model(**body_param_dict).joints  # [b,J,3]

        new_rotmat, new_transl = get_new_coordinate(joints)  # transformation from new coord axis to old coord axis

        return new_rotmat, new_transl

    def calc_calibrate_offset(self, body_param_dict):
        body_model = self.bm_male if body_param_dict['gender'] == 'male' else self.bm_female
        smplx_out = body_model(betas=body_param_dict['betas'],
                               # body_pose=body_param_dict['body_pose'],
                               )
        delta_T = smplx_out.joints[:, 0, :]  # [b, 3], we output all pelvis locations

        return delta_T

    def canonicalize(self, primitive_dict, use_predicted_joints=False):
        """inplace canonicalize
        primitive_dict:{
        'transf_rotmat', 'transf_transl': [B, 3, 3], [B, 1, 3]
        'gender': 'male' or 'female',
        'betas': [B, T, 10],
        'transl', 'global_orient', 'body_pose': [B, T, 3], [B, T, 3, 3], [B, T, 21, 3, 3]
        'joints': optional, [B, T, 22*3],
        }
        """
        body_param_dict = {
            'gender': primitive_dict['gender'],
            'betas': primitive_dict['betas'][:, 0, :],
            'transl': primitive_dict['transl'][:, 0, :],
            'body_pose': primitive_dict['body_pose'][:, 0, :, :, :],
            'global_orient': primitive_dict['global_orient'][:, 0, :, :],
        }   # first frame bodies
        # delta_T = self.calc_calibrate_offset(body_param_dict)  # [b,3]
        delta_T = primitive_dict['pelvis_delta'] if 'pelvis_delta' in primitive_dict else self.calc_calibrate_offset(body_param_dict)  # [b,3]
        transf_rotmat, transf_transl = self.get_new_coordinate(body_param_dict,
                                                               use_predicted_joints=use_predicted_joints,
                                                               pred_joints=primitive_dict['joints'][:, 0, :].reshape(-1, 22, 3) if 'joints' in primitive_dict else None
                                                               )  # [b,3,3], [b,1,3]

        transl = primitive_dict['transl']  # [b, T, 3]
        global_ori = primitive_dict['global_orient']  # [b, T, 3, 3]
        global_ori_new = torch.einsum('bij,btjk->btik', transf_rotmat.permute(0, 2, 1), global_ori)
        transl = torch.einsum('bij,btj->bti', transf_rotmat.permute(0, 2, 1),
                              transl + delta_T.unsqueeze(1) - transf_transl) - delta_T.unsqueeze(1)
        primitive_dict['global_orient'] = global_ori_new
        primitive_dict['transl'] = transl

        if 'joints' in primitive_dict:
            B, T, _ = primitive_dict['transl'].shape
            joints = primitive_dict['joints'].reshape(B, T, 22, 3)  # [b, T, 22*3] -> [b, T, 22, 3]
            joints = torch.einsum('bij,btkj->btki', transf_rotmat.permute(0, 2, 1), joints - transf_transl.unsqueeze(1))
            primitive_dict['joints'] = joints.reshape(B, T, 22 * 3)  # [b, T, 22*3]

        update_global_transform(primitive_dict, transf_rotmat, transf_transl)
        return transf_rotmat, transf_transl, primitive_dict

    def calc_features(self, primitive_dict, use_predicted_joints=False):
        """calculate the redundant features from the smplx sequences"""
        motion_features = {
            # 'gender': primitive_dict['gender'],
            # 'betas': primitive_dict['betas'],
            # 'transf_rotmat': primitive_dict['transf_rotmat'],
            # 'transf_transl': primitive_dict['transf_transl'],
        }
        B, T, _ = primitive_dict['transl'].shape
        if use_predicted_joints:
            output_joints = primitive_dict['joints'].reshape(B, T, 22, 3)
        else:
            body_model = self.bm_male if primitive_dict['gender'] == 'male' else self.bm_female
            smplx_out = body_model(betas=primitive_dict['betas'].reshape(B * T, 10),
                                   global_orient=primitive_dict['global_orient'].reshape(B * T, 3, 3),
                                   body_pose=primitive_dict['body_pose'].reshape(B * T, 21, 3, 3),
                                   transl=primitive_dict['transl'].reshape(B * T, 3)
                                   )
            output_joints = smplx_out.joints[:, :22].reshape(B, T, 22, 3)
        motion_features['transl'] = primitive_dict['transl']  # [b, t,3]
        motion_features['transl_delta'] = primitive_dict['transl'][:, 1:, :] - primitive_dict['transl'][:, :-1, :]  # [b, t-1,3]
        motion_features['joints'] = output_joints[:, :, :22].reshape(B, T, 22 * 3)
        motion_features['joints_delta'] = (output_joints[:, 1:, :22, :] - output_joints[:, :-1, :22, :]).reshape(B, T - 1, 22 * 3)
        global_orient_delta_rotmat = torch.matmul(primitive_dict['global_orient'][:, 1:],
                                                  primitive_dict['global_orient'][:, :-1].permute(0, 1, 3, 2))
        motion_features['global_orient_delta_6d'] = transforms.matrix_to_rotation_6d(global_orient_delta_rotmat)  # [B, t-1, 6]
        motion_features['poses_6d'] = transforms.matrix_to_rotation_6d(
            torch.cat([primitive_dict['global_orient'].unsqueeze(2), primitive_dict['body_pose']], dim=2)
            # [B, t, 22, 3, 3]
        ).reshape(B, T, 22 * 6)  # [B, t, 22 * 6]

        return motion_features

    def get_blended_feature(self, feature_dict, use_predicted_joints=False):
        primitive_dict = self.feature_dict_to_smpl_dict(feature_dict)
        transf_rotmat, transf_transl, primitive_dict = self.canonicalize(primitive_dict,
                                                                         use_predicted_joints=use_predicted_joints)
        # print('relative transform:', transf_rotmat, transf_transl)
        if use_predicted_joints:  #  directly use the predicted joints, no blending
            # transf_rotmat, transf_transl: [b,3,3], [b,1,3], transformation from new coord axis to old coord axis
            B, T, _ = feature_dict['transl'].shape
            poses_6d = feature_dict['poses_6d']  # [b, T, 22*6], not change
            global_orient_6d = poses_6d[:, :, :6]  # [b, T, 6]
            global_orient_rotmat = transforms.rotation_6d_to_matrix(global_orient_6d)  # [b, T, 3, 3]
            global_orient_rotmat = torch.matmul(transf_rotmat.permute(0, 2, 1).unsqueeze(1), global_orient_rotmat)
            global_orient_6d = transforms.matrix_to_rotation_6d(global_orient_rotmat)  # [b, T, 6]
            new_poses_6d = torch.cat([global_orient_6d, poses_6d[:, :, 6:]], dim=-1)  # [b, T, 22*6]
            global_orient_delta_6d = feature_dict['global_orient_delta_6d']  # [b, T, 6], not change
            global_orient_delta_rotmat = transforms.rotation_6d_to_matrix(global_orient_delta_6d)  # [b, T, 3, 3]
            global_orient_delta_rotmat = torch.matmul(
                torch.matmul(transf_rotmat.permute(0, 2, 1).unsqueeze(1), global_orient_delta_rotmat),
                transf_rotmat.unsqueeze(1)
            )
            global_orient_delta_6d = transforms.matrix_to_rotation_6d(global_orient_delta_rotmat)  # [b, T, 6]
            transl = primitive_dict['transl']  # [b, T, 3], from canonicalized primitive dict
            joints = primitive_dict['joints']  # [b, T, 22*3], from canonicalized primitive dict
            transl_delta = feature_dict['transl_delta']  # [b, T, 3]
            joints_delta = feature_dict['joints_delta'].reshape(B, T, 22, 3)  # [b, T, 22*3]
            transl_delta = torch.einsum('bij,btj->bti', transf_rotmat.permute(0, 2, 1), transl_delta)  # [b,3]
            joints_delta = torch.einsum('bij,btkj->btki', transf_rotmat.permute(0, 2, 1), joints_delta).reshape(B, T, 22 * 3)
            smpl_features = {
                'transl': transl,
                'transl_delta': transl_delta,
                'joints': joints,
                'joints_delta': joints_delta,
                'global_orient_delta_6d': global_orient_delta_6d,
                'poses_6d': new_poses_6d,
            }
        else:  # use smplx to infer joint location from rotation, and blend with last frame

            smpl_features = self.calc_features(primitive_dict)
            last_transl_delta = feature_dict['transl_delta'][:, -1, :]  # [b,3]
            last_joints_delta = feature_dict['joints_delta'][:, -1, :]  # [b,22*3]
            last_global_orient_delta_6d = feature_dict['global_orient_delta_6d'][:, -1, :]  # [b,6], not change
            last_global_orient_delta_rotmat = transforms.rotation_6d_to_matrix(last_global_orient_delta_6d)  # [b,3,3]
            last_global_orient_delta_rotmat = torch.matmul(torch.matmul(transf_rotmat.permute(0, 2, 1), last_global_orient_delta_rotmat), transf_rotmat)  # [b,3,3]
            last_global_orient_delta_6d = transforms.matrix_to_rotation_6d(last_global_orient_delta_rotmat)  # [b,6]
            # transform the last frame delta features
            last_transl_delta = torch.einsum('bij,bj->bi', transf_rotmat.permute(0, 2, 1), last_transl_delta) # [b,3]
            last_joints_delta = torch.einsum('bij,bkj->bki', transf_rotmat.permute(0, 2, 1),
                                             last_joints_delta.reshape(-1, 22, 3)
                                             ).reshape(-1, 22 * 3) # [b,22*3]

            smpl_features['transl_delta'] = torch.cat([smpl_features['transl_delta'], last_transl_delta.unsqueeze(1)], dim=1)
            smpl_features['joints_delta'] = torch.cat([smpl_features['joints_delta'], last_joints_delta.unsqueeze(1)], dim=1)
            smpl_features['global_orient_delta_6d'] = torch.cat([smpl_features['global_orient_delta_6d'],
                                                                last_global_orient_delta_6d.unsqueeze(1)], dim=1)

        smpl_features['transf_rotmat'] = primitive_dict['transf_rotmat']
        smpl_features['transf_transl'] = primitive_dict['transf_transl']
        smpl_features['gender'] = primitive_dict['gender']
        smpl_features['betas'] = primitive_dict['betas']
        if 'pelvis_delta' in primitive_dict:
            smpl_features['pelvis_delta'] = primitive_dict['pelvis_delta']
        new_feature_dict = smpl_features
        return primitive_dict, new_feature_dict

    def transform_feature_to_world(self, feature_dict, use_predicted_joints=True):
        transf_rotmat, transf_transl = feature_dict['transf_rotmat'], feature_dict['transf_transl']
        device = transf_rotmat.device
        batch_size = transf_rotmat.shape[0]
        dtype = transf_rotmat.dtype
        delta_T = feature_dict['pelvis_delta']

        B, T, _ = feature_dict['transl'].shape
        poses_6d = feature_dict['poses_6d']  # [b, T, 22*6], not change
        global_orient_6d = poses_6d[:, :, :6]  # [b, T, 6]
        global_orient_rotmat = transforms.rotation_6d_to_matrix(global_orient_6d)  # [b, T, 3, 3]
        global_orient_rotmat = torch.matmul(transf_rotmat.unsqueeze(1), global_orient_rotmat)
        global_orient_6d = transforms.matrix_to_rotation_6d(global_orient_rotmat)  # [b, T, 6]
        new_poses_6d = torch.cat([global_orient_6d, poses_6d[:, :, 6:]], dim=-1)  # [b, T, 22*6]
        global_orient_delta_6d = feature_dict['global_orient_delta_6d']  # [b, T, 6], not change
        global_orient_delta_rotmat = transforms.rotation_6d_to_matrix(global_orient_delta_6d)  # [b, T, 3, 3]
        global_orient_delta_rotmat = torch.matmul(
            torch.matmul(transf_rotmat.unsqueeze(1), global_orient_delta_rotmat),
            transf_rotmat.permute(0, 2, 1).unsqueeze(1)
        )
        global_orient_delta_6d = transforms.matrix_to_rotation_6d(global_orient_delta_rotmat)  # [b, T, 6]
        transl = feature_dict['transl']  # [b, T, 3]
        joints = feature_dict['joints'].reshape(B, T, 22, 3)  # [b, T, 22*3]
        transl = torch.einsum('bij,btj->bti', transf_rotmat,
                              transl + delta_T.unsqueeze(1)) - delta_T.unsqueeze(1) + transf_transl
        joints = torch.einsum('bij,btkj->btki', transf_rotmat, joints) + transf_transl.unsqueeze(1)
        joints = joints.reshape(B, T, 22 * 3)
        transl_delta = feature_dict['transl_delta']  # [b, T, 3]
        joints_delta = feature_dict['joints_delta'].reshape(B, T, 22, 3)  # [b, T, 22*3]
        transl_delta = torch.einsum('bij,btj->bti', transf_rotmat, transl_delta)  # [b,3]
        joints_delta = torch.einsum('bij,btkj->btki', transf_rotmat, joints_delta).reshape(B, T,
                                                                                                            22 * 3)

        world_feature_dict = {
            'transf_rotmat': torch.eye(3, device=device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1, 1),
            'transf_transl': torch.zeros(3, device=device, dtype=dtype).reshape(1, 1, 3).repeat(batch_size, 1, 1),
            'gender': feature_dict['gender'],
            'betas': feature_dict['betas'],
            'pelvis_delta': feature_dict['pelvis_delta'],
            'transl': transl,
            'transl_delta': transl_delta,
            'joints': joints,
            'joints_delta': joints_delta,
            'global_orient_delta_6d': global_orient_delta_6d,
            'poses_6d': new_poses_6d,
        }
        return world_feature_dict

    def transform_primitive_to_world(self, primitive_dict):
        # body_param_dict = {
        #     'gender': primitive_dict['gender'],
        #     'betas': primitive_dict['betas'][:, 0, :],
        #     'transl': primitive_dict['transl'][:, 0, :],
        #     'body_pose': primitive_dict['body_pose'][:, 0, :, :, :],
        #     'global_orient': primitive_dict['global_orient'][:, 0, :, :],
        # }  # first frame bodies
        # delta_T = self.calc_calibrate_offset(body_param_dict)  # [b,3]
        delta_T = primitive_dict['pelvis_delta'] if 'pelvis_delta' in primitive_dict else self.calc_calibrate_offset({
            'gender': primitive_dict['gender'],
            'betas': primitive_dict['betas'][:, 0, :],
        })  # [b,3]
        transf_rotmat, transf_transl = primitive_dict['transf_rotmat'], primitive_dict['transf_transl']

        B, T, _ = primitive_dict['transl'].shape
        transl = primitive_dict['transl']  # [b, T, 3]
        joints = primitive_dict['joints'].reshape(B, T, 22, 3)  # [b, T, 22*3] -> [b, T, 22, 3]
        global_ori = primitive_dict['global_orient']  # [b, T, 3, 3]
        global_ori_new = torch.einsum('bij,btjk->btik', transf_rotmat, global_ori)
        transl = torch.einsum('bij,btj->bti', transf_rotmat,
                              transl + delta_T.unsqueeze(1)) - delta_T.unsqueeze(1) + transf_transl
        joints = torch.einsum('bij,btkj->btki', transf_rotmat, joints) + transf_transl.unsqueeze(1)

        primitive_dict['global_orient'] = global_ori_new
        primitive_dict['transl'] = transl
        primitive_dict['joints'] = joints
        primitive_dict['transf_rotmat'] = torch.eye(3).unsqueeze(0).repeat(transf_rotmat.shape[0], 1, 1).to(
            device=self.device, dtype=self.dtype)
        primitive_dict['transf_transl'] = torch.zeros(transf_transl.shape).to(device=self.device, dtype=self.dtype)

        return primitive_dict


JOINT_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "jaw",
    "left_eye_smplhf",
    "right_eye_smplhf",
    "left_index1",
    "left_index2",
    "left_index3",
    "left_middle1",
    "left_middle2",
    "left_middle3",
    "left_pinky1",
    "left_pinky2",
    "left_pinky3",
    "left_ring1",
    "left_ring2",
    "left_ring3",
    "left_thumb1",
    "left_thumb2",
    "left_thumb3",
    "right_index1",
    "right_index2",
    "right_index3",
    "right_middle1",
    "right_middle2",
    "right_middle3",
    "right_pinky1",
    "right_pinky2",
    "right_pinky3",
    "right_ring1",
    "right_ring2",
    "right_ring3",
    "right_thumb1",
    "right_thumb2",
    "right_thumb3",
    "nose",
    "right_eye",
    "left_eye",
    "right_ear",
    "left_ear",
    "left_big_toe",
    "left_small_toe",
    "left_heel",
    "right_big_toe",
    "right_small_toe",
    "right_heel",
    "left_thumb",
    "left_index",
    "left_middle",
    "left_ring",
    "left_pinky",
    "right_thumb",
    "right_index",
    "right_middle",
    "right_ring",
    "right_pinky",
    "right_eye_brow1",
    "right_eye_brow2",
    "right_eye_brow3",
    "right_eye_brow4",
    "right_eye_brow5",
    "left_eye_brow5",
    "left_eye_brow4",
    "left_eye_brow3",
    "left_eye_brow2",
    "left_eye_brow1",
    "nose1",
    "nose2",
    "nose3",
    "nose4",
    "right_nose_2",
    "right_nose_1",
    "nose_middle",
    "left_nose_1",
    "left_nose_2",
    "right_eye1",
    "right_eye2",
    "right_eye3",
    "right_eye4",
    "right_eye5",
    "right_eye6",
    "left_eye4",
    "left_eye3",
    "left_eye2",
    "left_eye1",
    "left_eye6",
    "left_eye5",
    "right_mouth_1",
    "right_mouth_2",
    "right_mouth_3",
    "mouth_top",
    "left_mouth_3",
    "left_mouth_2",
    "left_mouth_1",
    "left_mouth_5",  # 59 in OpenPose output
    "left_mouth_4",  # 58 in OpenPose output
    "mouth_bottom",
    "right_mouth_4",
    "right_mouth_5",
    "right_lip_1",
    "right_lip_2",
    "lip_top",
    "left_lip_2",
    "left_lip_1",
    "left_lip_3",
    "lip_bottom",
    "right_lip_3",
    # Face contour
    "right_contour_1",
    "right_contour_2",
    "right_contour_3",
    "right_contour_4",
    "right_contour_5",
    "right_contour_6",
    "right_contour_7",
    "right_contour_8",
    "contour_middle",
    "left_contour_8",
    "left_contour_7",
    "left_contour_6",
    "left_contour_5",
    "left_contour_4",
    "left_contour_3",
    "left_contour_2",
    "left_contour_1",
]
FOOT_JOINTS_IDX = [JOINT_NAMES.index(joint_name) for joint_name in ['left_foot', 'right_foot']]
