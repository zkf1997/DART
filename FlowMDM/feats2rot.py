from data_loaders.amass.transforms import SlimSMPLTransform
from pathlib import Path
import torch
import numpy as np
import pickle
import json
from utils.rotation_conversions import matrix_to_axis_angle, rotation_6d_to_matrix

def convert_feat(feat_path):
    config_path = feat_path.parent / feat_path.name.replace('.pt', '_kwargs.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    npz_path = feat_path.parent / feat_path.name.replace('.pt', '_smpl.npz')
    pkl_path = feat_path.parent / feat_path.name.replace('.pt', '.pkl')
    with open(feat_path, 'rb') as f:
        feature = torch.load(f)
    feature = feature.squeeze().permute(1, 0)
    # print(feature)

    smpl_param = transform.rots2rfeats.inverse(feature.cpu())
    num_frames = smpl_param.trans.shape[0]
    # print(smpl_param.rots.shape, smpl_param.trans.shape)
    thetas = smpl_param.rots
    thetas = matrix_to_axis_angle(thetas).reshape(-1, 66)  # [T, 22*3]
    thetas = torch.cat([thetas, torch.zeros(num_frames, 99).to(dtype=thetas.dtype)], dim=1)
    trans = smpl_param.trans  # [T, 3]
    data_dict = {
        'mocap_framerate': 30,  # actually 20, set 30 to be compatible with blender add on
        'gender': 'male',
        'betas': np.zeros((16, )),
        'poses': thetas.detach().cpu().numpy(),
        'trans': trans.detach().cpu().numpy(),
    }
    np.savez(npz_path, **data_dict)

    # text_idx = []
    # for idx, text in enumerate(config['text']):
    #     text_idx += [idx] * config['lengths'][idx]
    # data_dict = {
    #     'mocap_framerate': 30,  # actually 20, set 30 to be compatible with blender add on
    #     'gender': 'male',
    #     'betas': torch.zeros((num_frames, 10)),
    #     'transl': smpl_param.trans.detach().cpu(),
    #     'body_pose': smpl_param.rots[:, 1:].detach().cpu(),
    #     'global_orient': smpl_param.rots[:, 0].detach().cpu(),
    #     'texts': config['text'],
    #     'text_idx': text_idx,
    # }
    # with open(pkl_path, 'wb') as f:
    #     pickle.dump(data_dict, f)


transform = SlimSMPLTransform(batch_size=1, name='SlimSMPLTransform', ename='smplnh', normalization=True)

# base_dir = Path('/home/kaizhao/projects/flowmdm/results/babel/FlowMDM/evaluation_precomputed/Motion_FlowMDM_001300000_gscale1.5_final_s10')
# base_dir = Path('/home/kaizhao/projects/flowmdm/results/babel/Motion_FlowMDM_001300000_gscale1.5_fastbabel_random_s10/DoubleTake_eval_samples_Babel_TrasnEmb_GeoLoss_000850000_seed10_handshake_6_blend_2_skipSteps_100/evaluation_precomputed')
# for feat_path in base_dir.glob('./*/*.pt'):
#     print('process:', feat_path)
#     convert_feat(feat_path)

base_dir = Path('/home/kaizhao/projects/flowmdm/results/babel/Motion_FlowMDM_001300000_gscale1.5_fastbabel_rebuttal_sub_s10/DoubleTake_eval_samples_Babel_TrasnEmb_GeoLoss_000850000_seed10_handshake_6_blend_2_skipSteps_100/evaluation_precomputed/00')
for feat_path in base_dir.glob('./*.pt'):
    print('process:', feat_path)
    convert_feat(feat_path)