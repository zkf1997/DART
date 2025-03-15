import pickle
import json
from pathlib import Path
import numpy as np
from utils.misc_util import *

dataset_dir = Path('/home/kaizhao/projects/mdm/hml3d_filter')
with open(dataset_dir / 'test_dataset.pkl', 'rb') as f:
    dataset = pickle.load(f)

with open('/home/kaizhao/projects/multiskill/data/hml3d_smplh/seq_data_zero_male/test.pkl', 'rb') as f:
    hml_smpl_dataset = pickle.load(f)

labels = []
for seq in hml_smpl_dataset:
    for label_idx in range(len(seq['frame_labels'])):
        labels.append((seq['frame_labels'][label_idx]['proc_label'], seq, label_idx))
clip_model = load_and_freeze_clip(clip_version='ViT-B/32', device='cuda')
text_embeddings = []
batch_start_idx = 0
num_texts = len(labels)
raw_texts = [label[0] for label in labels]
batch_size = 256
while batch_start_idx < num_texts:
    batch_end_idx = min(batch_start_idx + batch_size, num_texts)
    text_embeddings.append(encode_text(clip_model, raw_texts[batch_start_idx:batch_end_idx]))
    batch_start_idx = batch_end_idx
text_embeddings = torch.cat(text_embeddings, dim=0)  # [num_texts, 512]

matched_label = []
matched_motions = []
for data in dataset:
    text, length = data[2], data[-2]
    found_flag = False
    for label in labels:
        if text == label[0]:
            matched_label.append(label)
            motion = label[1]['motion']
            frame_label = label[1]['frame_labels'][label[2]]
            start_t, end_t = frame_label['start_t'], frame_label['end_t']
            start_frame, end_frame = int(start_t * 20), int(end_t * 20)
            for key in ['poses', 'trans', 'joints']:
                motion[key] = motion[key][start_frame:end_frame]
            matched_motions.append(motion)
            found_flag = True
            break
    if found_flag:
        # print(f'Found: {seq_name} {text} {start_t} {end_t}')
        pass
    else:
        print(f'Not found: {text}')
        text_embedding = encode_text(clip_model, [text])  # 1,512,
        # find nearest embedding
        similarity_list = []
        batch_start_idx = 0
        while batch_start_idx < num_texts:
            batch_end_idx = min(batch_start_idx + batch_size, num_texts)
            similarity = torch.nn.functional.cosine_similarity(text_embeddings[batch_start_idx:batch_end_idx],
                                                               text_embedding)
            similarity_list.append(similarity)
            batch_start_idx = batch_end_idx
        similarity = torch.cat(similarity_list, dim=0)
        label_idx = torch.argmax(similarity).item()
        print(f'Nearest text: {labels[label_idx][0]}')
        label = labels[label_idx]
        motion = label[1]['motion']
        frame_label = label[1]['frame_labels'][label[2]]
        start_t, end_t = frame_label['start_t'], frame_label['end_t']
        start_frame, end_frame = int(start_t * 20), int(end_t * 20)
        for key in ['poses', 'trans', 'joints']:
            motion[key] = motion[key][start_frame:end_frame]
        matched_motions.append(motion)

with open('/home/kaizhao/projects/mdm/hml3d_filter/matched_smplh.pkl', 'wb') as f:
    pickle.dump(matched_motions, f)