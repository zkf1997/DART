import numpy
from pathlib import Path
import pickle
import os
import numpy as np
from pytorch3d import transforms
from scipy.spatial.transform import Rotation as R

# AMASS dataset names from website are slightly different from what used in BABEL
amass_dataset_rename_dict = {
    'ACCAD': 'ACCAD',
    'BMLmovi': 'BMLmovi',
    'BioMotionLab_NTroje': 'BMLrub',
    'MPI_HDM05': 'HDM05',
    'CMU': 'CMU',
    'Eyes_Japan_Dataset': 'EyesJapanDataset/Eyes_Japan_Dataset',
    'HumanEva': 'HumanEva',
    'TCD_handMocap': 'TCDHands',
    'KIT': 'KIT',
    'Transitions_mocap': 'Transitions',
    'DFaust_67': 'DFaust',
    'MPI_Limits': 'PosePrior',
    'SSM_synced': 'SSM',
    'MPI_mosh': 'MoSh',
}
raw_dataset_path = '/home/kaizhao/dataset/amass/smplx_g'

def export(feat_p, start_frame, end_frame, texts, output_path, output_mode='default', target_fps=30):
    file_path = os.path.join(*(feat_p.split(os.path.sep)[1:]))
    dataset_name = file_path.split(os.path.sep)[0]
    if dataset_name in amass_dataset_rename_dict:
        file_path = file_path.replace(dataset_name, amass_dataset_rename_dict[dataset_name])
    file_path = file_path.replace('poses.npz',
                                  'stageii.npz')  # file naming suffix changed in different amass versions
    # replace space
    file_path = file_path.replace(" ",
                                  "_")  # set replace count to string length, so all will be replaced
    seq = os.path.join(raw_dataset_path, file_path)
    data = dict(np.load(seq, allow_pickle=True))
    downsample_rate = 120 // target_fps
    ## read data and downsample
    transl_all = data['trans'][::downsample_rate]
    pose_all = data['poses'][::downsample_rate]
    betas = data['betas'][:10]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_mode == 'default':
        primitive_dict = {
            'gender': str(data['gender'].item()),
            'betas': betas,
            'transl': transl_all[start_frame:end_frame, :],
            'global_orient': pose_all[start_frame:end_frame, :3],
            'body_pose': pose_all[start_frame:end_frame, 3:66],
            'texts': texts,
        }
        # save the primitive
        with open(output_path, 'wb') as f:
            pickle.dump(primitive_dict, f)
        print(f'Primitive saved to {output_path}')
    elif output_mode == 'flowmdm':
        transl = transl_all[start_frame:end_frame, :]
        poses = pose_all[start_frame:end_frame, :66].reshape(-1, 22, 3)
        poses = R.from_rotvec(poses).as_matrix()  # [T, 22, 3, 3]
        with open(output_path, 'wb') as f:
            np.save(f, {'transl': transl, 'rots': poses})

# # sit down primitive
# feat_p = 'BMLrub/BioMotionLab_NTroje/rub018/0016_sitting2_poses.npz'
# start_frame = 54
# end_frame = 75
# texts = ['sit down']
# file_name = '+'.join(texts)
# output_path = f'/home/kaizhao/projects/multiskill/data/{file_name}.pkl'
# export(feat_p, start_frame, end_frame, texts, output_path)
#
# # sit still primitive
# feat_p = 'BMLrub/BioMotionLab_NTroje/rub018/0016_sitting2_poses.npz'
# start_frame = 138
# end_frame = 159
# texts = ['sit on chair']
# file_name = '+'.join(texts)
# output_path = f'/home/kaizhao/projects/multiskill/data/{file_name}.pkl'
# export(feat_p, start_frame, end_frame, texts, output_path)
#
# # stand up primitive
# feat_p = 'BMLrub/BioMotionLab_NTroje/rub018/0016_sitting2_poses.npz'
# start_frame = 190
# end_frame = 211
# texts = ['stand up']
# file_name = '+'.join(texts)
# output_path = f'/home/kaizhao/projects/multiskill/data/{file_name}.pkl'
# export(feat_p, start_frame, end_frame, texts, output_path)
#
# feat_p = 'BMLmovi/BMLmovi/Subject_15_F_MoSh/Subject_15_F_1_poses.npz'
# start_frame = 0
# end_frame = 21
# texts = ['stand']
# file_name = '+'.join(texts)
# output_path = f'/home/kaizhao/projects/multiskill/data/{file_name}.pkl'
# export(feat_p, start_frame, end_frame, texts, output_path)
#
# feat_p = 'BMLrub/BioMotionLab_NTroje/rub018/0016_sitting2_poses.npz'
# start_frame = 0
# end_frame = 21
# texts = ['walk']
# file_name = '+'.join(texts)
# output_path = f'/home/kaizhao/projects/multiskill/data/{file_name}.pkl'
# export(feat_p, start_frame, end_frame, texts, output_path)
#
# feat_p = 'KIT/KIT/348/walking_run08_poses.npz'
# start_frame = 73
# end_frame = 94
# texts = ['run']
# file_name = '+'.join(texts)
# output_path = f'/home/kaizhao/projects/multiskill/data/{file_name}.pkl'
# export(feat_p, start_frame, end_frame, texts, output_path)
#
# feat_p = 'CMU/CMU/105/105_45_poses.npz'
# start_frame = 26
# end_frame = 47
# texts = ['jump forward']
# file_name = '+'.join(texts)
# output_path = f'/home/kaizhao/projects/multiskill/data/{file_name}.pkl'
# export(feat_p, start_frame, end_frame, texts, output_path)

# feat_p = 'KIT/KIT/441/conversation01_poses.npz'
# start_frame = 0
# end_frame = 21
# texts = ['stand still']
# file_name = '+'.join(texts)
# output_path = f'/home/kaizhao/projects/multiskill/data/{file_name}.pkl'
# export(feat_p, start_frame, end_frame, texts, output_path)

# feat_p = 'EyesJapanDataset/Eyes_Japan_Dataset/aita/walk-07-moonwalk-aita_poses.npz'
# start_frame = 0
# end_frame = 21
# texts = ['t pose']
# file_name = '+'.join(texts)
# output_path = f'/home/kaizhao/projects/multiskill/data/{file_name}.pkl'
# export(feat_p, start_frame, end_frame, texts, output_path)

# feat_p = 'BMLrub/BioMotionLab_NTroje/rub114/0026_circle_walk_poses.npz'
# start_frame = 0
# end_frame = 82
# texts = ['walk in circle']
# file_name = '+'.join(texts)
# output_path = f'/home/kaizhao/projects/multiskill/data/{file_name}.pkl'
# export(feat_p, start_frame, end_frame, texts, output_path)

# feat_p = 'EyesJapanDataset/Eyes_Japan_Dataset/hamada/gesture_etc-06-jiggle knee-hamada_poses.npz'
# start_frame = 100
# end_frame = 241
# texts = ['sit down']
# file_name = '+'.join(texts)
# output_path = f'/home/kaizhao/projects/multiskill/data/opt/{file_name}.pkl'
# export(feat_p, start_frame, end_frame, texts, output_path)

# feat_p = "CMU/CMU/60/60_08_poses.npz"
# start_frame = 100
# end_frame = 182
# texts = ['dance']
# file_name = '+'.join(texts)
# output_path = f'/home/kaizhao/projects/multiskill/data/opt/{file_name}.pkl'
# export(feat_p, start_frame, end_frame, texts, output_path)

# feat_p = "BMLmovi/BMLmovi/Subject_38_F_MoSh/Subject_38_F_21_poses.npz"
# start_frame = int(9 * 30)
# end_frame = int(13 * 30)
# texts = ['crawl']
# file_name = '+'.join(texts)
# output_path = f'/home/kaizhao/projects/multiskill/data/opt/{file_name}.pkl'
# export(feat_p, start_frame, end_frame, texts, output_path)

# feat_p = "CMU/CMU/90/90_04_poses.npz"
# start_frame = 0
# end_frame = int(5.5 * 30)
# texts = ['cartwheel']
# file_name = '+'.join(texts)
# output_path = f'/home/kaizhao/projects/multiskill/data/rec_test/{file_name}.pkl'
# export(feat_p, start_frame, end_frame, texts, output_path)

# feat_p = "EyesJapanDataset/Eyes_Japan_Dataset/shiono/pose-19-funny-shiono_poses.npz"
# start_frame = 25 * 30
# end_frame = 30 * 30
# texts = ['t pose']
# file_name = '+'.join(texts)
# output_path = f'/home/kaizhao/projects/multiskill/data/rec_test/{file_name}.pkl'
# export(feat_p, start_frame, end_frame, texts, output_path)

# feat_p = "Transitionsmocap/Transitions_mocap/mazen_c3d/LOB_turntwist180_poses.npz"
# start_frame = 0 * 30
# end_frame = 9 * 30
# texts = ['lie down']
# file_name = '+'.join(texts)
# output_path = f'/home/kaizhao/projects/multiskill/data/rec_test/{file_name}.pkl'
# export(feat_p, start_frame, end_frame, texts, output_path)
#
# feat_p = 'EyesJapanDataset/Eyes_Japan_Dataset/kudo/gesture_etc-19-fold legs-kudo_poses.npz'
# start_frame = 0 * 30
# end_frame = 33 * 30
# texts = ['sit and move leg']
# file_name = '+'.join(texts)
# output_path = f'/home/kaizhao/projects/multiskill/data/rec_test/{file_name}.pkl'
# export(feat_p, start_frame, end_frame, texts, output_path)

# feat_p = "BMLrub/BioMotionLab_NTroje/rub022/0014_sitting1_poses.npz"
# start_frame = int(2.555 * 30)
# end_frame = int(5.992 * 30)
# texts = ['sit on chair']
# file_name = '+'.join(texts)
# output_path = f'/home/kaizhao/projects/multiskill/data/rec_test/{file_name}.pkl'
# export(feat_p, start_frame, end_frame, texts, output_path)

# feat_p = "CMU/CMU/105/105_45_poses.npz"
# start_frame = int(0 * 30)
# end_frame = int(3.1 * 30)
# texts = ['jump forward']
# file_name = '+'.join(texts)
# output_path = f'/home/kaizhao/projects/multiskill/data/rec_test/{file_name}.pkl'
# export(feat_p, start_frame, end_frame, texts, output_path)

# feat_p = "EyesJapanDataset/Eyes_Japan_Dataset/hamada/jump-02-leap-hamada_poses.npz"
# start_frame = int(0 * 30)
# end_frame = int(30 * 30)
# texts = ['long jump']
# file_name = '+'.join(texts)
# output_path = f'/home/kaizhao/projects/multiskill/data/rec_test/{file_name}.pkl'
# export(feat_p, start_frame, end_frame, texts, output_path)

# feat_p = "BMLrub/BioMotionLab_NTroje/rub076/0031_rom_poses.npz"
# start_frame = int(0 * 30)
# end_frame = int(123.7 * 30)
# texts = ['arm swing']
# file_name = '+'.join(texts)
# output_path = f'/home/kaizhao/projects/multiskill/data/rec_test/{file_name}.pkl'
# export(feat_p, start_frame, end_frame, texts, output_path)

# feat_p = "BMLrub/BioMotionLab_NTroje/rub076/0031_rom_poses.npz"
# start_frame = int(17 * 30)
# end_frame = int(18 * 30)
# texts = ['arm swing']
# file_name = '+'.join(texts)
# output_path = f'/home/kaizhao/projects/multiskill/data/seed/{file_name}.pkl'
# export(feat_p, start_frame, end_frame, texts, output_path)

feat_p = 'BMLmovi/BMLmovi/Subject_15_F_MoSh/Subject_15_F_1_poses.npz'
start_frame = 0
end_frame = 21
texts = ['stand']
file_name = '+'.join(texts)
output_path = f'/home/kaizhao/projects/multiskill/data/{file_name}_20fps.pkl'
export(feat_p, start_frame, end_frame, texts, output_path, target_fps=20)

feat_p = 'BMLrub/BioMotionLab_NTroje/rub018/0016_sitting2_poses.npz'
start_frame = int(20 / 30 * 20)
end_frame = start_frame + 21
texts = ['walk']
file_name = '+'.join(texts)
output_path = f'/home/kaizhao/projects/multiskill/data/{file_name}_20fps.pkl'
export(feat_p, start_frame, end_frame, texts, output_path, target_fps=20)

