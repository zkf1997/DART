from torch.utils.data import DataLoader
from data_loaders.tensors import collate as all_collate
from data_loaders.tensors import t2m_collate
import torch
from tqdm import tqdm

def get_dataset_class(name):
    if name == "amass":
        from .amass import AMASS
        return AMASS
    elif name == "uestc":
        from .a2m.uestc import UESTC
        return UESTC
    elif name == "humanact12":
        from .a2m.humanact12poses import HumanAct12Poses
        return HumanAct12Poses
    elif name == "humanml":
        from data_loaders.humanml.data.dataset import HumanML3D
        return HumanML3D
    elif name == "kit":
        from data_loaders.humanml.data.dataset import KIT
        return KIT
    else:
        raise ValueError(f'Unsupported dataset name [{name}]')

def get_collate_fn(name, hml_mode='train'):
    if hml_mode == 'gt':
        from data_loaders.humanml.data.dataset import collate_fn as t2m_eval_collate
        return t2m_eval_collate
    if name in ["humanml", "kit"]:
        return t2m_collate
    else:
        return all_collate


def get_dataset(name, num_frames, split='train', hml_mode='train'):
    DATA = get_dataset_class(name)
    if name in ["humanml", "kit"]:
        dataset = DATA(split=split, num_frames=num_frames, mode=hml_mode)
    else:
        dataset = DATA(split=split, num_frames=num_frames)
    return dataset


def get_dataset_loader(name, batch_size, num_frames, split='train', hml_mode='train'):
    dataset = get_dataset(name, num_frames, split, hml_mode)
    collate = get_collate_fn(name, hml_mode)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=8, drop_last=True, collate_fn=collate
    )

    return loader

def mp_collate(batch):
    # sort batch by gender
    # batch = sorted(batch, key=lambda x: x['gender'])
    new_idx = []
    for gender in ['female', 'male']:
        new_idx = new_idx + [idx for idx in range(len(batch)) if batch[idx]['gender'] == gender]
    batch = [batch[i] for i in new_idx]

    text_batch = [b['text'] for b in batch]
    gender_batch = [b['gender'] for b in batch]
    betas_batch = torch.stack([b['betas'] for b in batch], dim=0)  # (B, T, 10)
    motion_batch = torch.stack([b['motion_tensor_normalized'] for b in batch], dim=0)  # (B, D, 1, T)
    history_mask_batch = torch.stack([b['history_mask'] for b in batch], dim=0)
    history_motion_batch = torch.stack([b['history_motion'] for b in batch], dim=0)

    motion = motion_batch
    cond = {'y': {'text': text_batch, 'gender': gender_batch, 'betas': betas_batch,
                  'history_motion': history_motion_batch, 'history_mask': history_mask_batch,
                  'history_length': batch[0]['history_length'], 'future_length': batch[0]['future_length']
                  }
            }
    return motion, cond

# def mp_seq_collate(batch):
#     # sort batch by gender
#     # batch = sorted(batch, key=lambda x: x['gender'])
#     new_idx = []
#     for gender in ['female', 'male']:
#         new_idx = new_idx + [idx for idx in range(len(batch)) if batch[idx][0]['gender'] == gender]
#     batch = [batch[i] for i in new_idx]
#     return batch

def get_dataset_loader_mp(dataset_path, batch_size, split='train'):
    from data_loaders.humanml.data.dataset import Text2MotionPrimitiveDataset
    dataset = Text2MotionPrimitiveDataset(dataset_path=dataset_path, split=split, load_data=True)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True if split == 'train' else False,
        num_workers=8, drop_last=True, collate_fn=mp_collate
    )

    return loader

# def get_dataset_loader_mp_seq(dataset_path, batch_size, split='train'):
#     from data_loaders.humanml.data.dataset import PrimitiveSequenceDataset
#     dataset = PrimitiveSequenceDataset(dataset_path=dataset_path, split=split, load_data=True)
#
#     loader = DataLoader(
#         dataset, batch_size=batch_size, shuffle=True if split == 'train' else False,
#         num_workers=8, drop_last=True, collate_fn=mp_seq_collate
#     )
#
#     return loader

if __name__ == "__main__":
    # trian_loader = get_dataset_loader_mp(dataset_path='./data/mp_data/Canonicalized_h2_f8_num1_fps30/', batch_size=2, split='train')
    # for i, batch in enumerate(trian_loader):
    #     print(i)
    #     print(batch[0], batch[1])
    #     break

    from data_loaders.humanml.data.dataset import PrimitiveSequenceDataset
    dataset = PrimitiveSequenceDataset(dataset_path='./data/mp_data/Canonicalized_h2_f8_num1_fps30/',
                                         split='train')
    for _ in tqdm(range(10)):
        batch = dataset.get_batch(batch_size=64)