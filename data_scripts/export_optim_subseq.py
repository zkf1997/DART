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

def export(feat_p, start_frame, end_frame, texts, output_path, output_mode='default', target_fps=30, body_type='smplx'):
    print(f'Exporting {feat_p} from frame {start_frame} to {end_frame} to {output_path}')
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
    frame_rate = data['mocap_frame_rate'].item()
    downsample_rate = int(frame_rate // target_fps)
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
    elif output_mode == '1f':
        primitive_dict = {
            'gender': str(data['gender'].item()),
            'betas': betas,
            'transl': transl_all[start_frame:end_frame, :],
            'global_orient': pose_all[start_frame:end_frame, :3],
            'body_pose': pose_all[start_frame:end_frame, 3:66],
            'texts': texts,
        }
        primitive_dict['transl'][1:-1] = primitive_dict['transl'][0]
        primitive_dict['global_orient'][1:-1] = primitive_dict['global_orient'][0]
        primitive_dict['body_pose'][1:-1] = primitive_dict['body_pose'][0]
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

# target_fps = 30
target_fps = 20
# output_mode = 'default'
output_mode = '1f'
export_path = Path(f'./data/inbetween/opt_eval_{target_fps}fps_{output_mode}')
raw_dataset_path = './data/amass/smplx_g'
seq_list = []

feat_p = 'KIT/KIT/9/LeftTurn07_poses.npz'
# start_frame = 30
# end_frame = 120
start_frame = int(1 * target_fps)
end_frame = int(4 * target_fps)
texts = ['walk']
seq_list.append(
    {'feat_p': feat_p, 'start_frame': start_frame, 'end_frame': end_frame, 'texts': texts}
)

feat_p = 'EyesJapanDataset/Eyes_Japan_Dataset/aita/walk-04-fast-aita_poses.npz'
start_frame = int(7.28 * target_fps)
end_frame = int(9.38 * target_fps)
texts = ['run forward']
seq_list.append(
    {'feat_p': feat_p, 'start_frame': start_frame, 'end_frame': end_frame, 'texts': texts}
)

feat_p = 'CMU/CMU/105/105_45_poses.npz'
start_frame = int(0.3 * target_fps)
end_frame = int(2.31 * target_fps)
texts = ['jump forward']
seq_list.append(
    {'feat_p': feat_p, 'start_frame': start_frame, 'end_frame': end_frame, 'texts': texts}
)

feat_p = 'BMLrub/BioMotionLab_NTroje/rub114/0026_circle_walk_poses.npz'
# start_frame = 30
# end_frame = 150
start_frame = int(1 * target_fps)
end_frame = int(5 * target_fps)
texts = ['pace in circles']
seq_list.append(
    {'feat_p': feat_p, 'start_frame': start_frame, 'end_frame': end_frame, 'texts': texts}
)

feat_p = "BMLmovi/BMLmovi/Subject_38_F_MoSh/Subject_38_F_21_poses.npz"
start_frame = int(9 * target_fps)
end_frame = int(13 * target_fps)
texts = ['crawl']
seq_list.append(
    {'feat_p': feat_p, 'start_frame': start_frame, 'end_frame': end_frame, 'texts': texts}
)

feat_p = "CMU/CMU/60/60_08_poses.npz"
# start_frame = 150
# end_frame = 270
start_frame = int(5 * target_fps)
end_frame = int(9 * target_fps)
texts = ['dance']
seq_list.append(
    {'feat_p': feat_p, 'start_frame': start_frame, 'end_frame': end_frame, 'texts': texts}
)

feat_p = "EyesJapanDataset/Eyes_Japan_Dataset/hamada/walk-05-backward-hamada_poses.npz"
start_frame = int(23 * target_fps)
end_frame = int(26 * target_fps)
texts = ['walk backwards']
seq_list.append(
    {'feat_p': feat_p, 'start_frame': start_frame, 'end_frame': end_frame, 'texts': texts}
)

feat_p = "KIT/KIT/423/downstairs03_poses.npz"
start_frame = int(0.564 * target_fps)
end_frame = int(3.81 * target_fps)
texts = ['climb down stairs']
seq_list.append(
    {'feat_p': feat_p, 'start_frame': start_frame, 'end_frame': end_frame, 'texts': texts}
)

feat_p = 'EyesJapanDataset/Eyes_Japan_Dataset/hamada/gesture_etc-06-jiggle knee-hamada_poses.npz'
# start_frame = 120
# end_frame = 210
start_frame = int(4 * target_fps)
end_frame = int(7 * target_fps)
texts = ['sit down']
file_name = '+'.join(texts)
seq_list.append(
    {'feat_p': feat_p, 'start_frame': start_frame, 'end_frame': end_frame, 'texts': texts}
)



for idx, seq in enumerate(seq_list):
    feat_p = seq['feat_p']
    start_frame = seq['start_frame']
    end_frame = seq['end_frame']
    texts = seq['texts']
    file_name = '+'.join(texts)
    output_path = export_path / f'{idx}_{file_name}.pkl'
    export(feat_p, start_frame, end_frame, texts, output_path, target_fps=target_fps, output_mode=output_mode)
