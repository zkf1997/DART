import numpy
from pathlib import Path
import pickle
import os
import numpy as np
import json
from os.path import join as ospj
from config_files.data_paths import *
from utils.misc_util import have_overlap, encode_text, load_and_freeze_clip
from tqdm import tqdm
import time
import smplx
import torch
import pickle
import trimesh
import pyrender


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

dataset = {
    'train': [],
    'val': [],
}
# load babel data
raw_dataset_path = amass_dir / 'smplx_g/'
d_folder = babel_dir
splits = ['train', 'val']
babel = {}
for spl in splits:
    babel[spl] = json.load(open(ospj(d_folder, spl + '.json')))
    for sid in tqdm(babel[spl]):
        if 'frame_ann' in babel[spl][sid] and babel[spl][sid]['frame_ann'] is not None:
            frame_labels = babel[spl][sid]['frame_ann']['labels']
        else:  # the sequence has only sequence label, which means the sequence has only one action
            frame_labels = babel[spl][sid]['seq_ann']['labels']  # onle one element
        for seg in frame_labels:
            dataset[spl].append(seg['proc_label'])
for spl in splits:
    dataset[spl] = list(set(dataset[spl]))

val_only_texts = [text for text in dataset['val'] if text not in dataset['train']]
# print('val_only_texts:', val_only_texts)

clip_model = load_and_freeze_clip(clip_version='ViT-B/32', device='cuda')
batch_size = 256
batch_start_idx = 0
num_texts = len(dataset['train'])
print('num_train_texts:', num_texts)
train_embeddings = []
while batch_start_idx < num_texts:
    batch_end_idx = min(batch_start_idx + batch_size, num_texts)
    train_embeddings.append(encode_text(clip_model, dataset['train'][batch_start_idx:batch_end_idx]))
    batch_start_idx = batch_end_idx
train_embeddings = torch.cat(train_embeddings, dim=0)

val_only_retrieval = {}
for text in val_only_texts:
    text_embedding = encode_text(clip_model, [text])
    similarity_list = []
    batch_start_idx = 0
    while batch_start_idx < num_texts:
        batch_end_idx = min(batch_start_idx + batch_size, num_texts)
        # print('batch_start_idx:', batch_start_idx)
        # print('batch_end_idx:', batch_end_idx)
        # print(text_embedding.shape)
        similarity = torch.nn.functional.cosine_similarity(train_embeddings[batch_start_idx:batch_end_idx], text_embedding)
        similarity_list.append(similarity)
        batch_start_idx = batch_end_idx
    similarity = torch.cat(similarity_list, dim=0)
    # print('similarity:', similarity.shape)
    val_only_retrieval[text] = dataset['train'][torch.argmax(similarity).item()]

export_path = './data/val_only_retrieval.json'
with open(export_path, 'w') as f:
    json.dump(val_only_retrieval, f)

