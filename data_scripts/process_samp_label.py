from pathlib import Path
import numpy as np
import json
import pickle

base_dir = Path('/home/kaizhao/dataset/samp/')
label_path = './data/samp_label.json'
output_path = './data/samp_label_walk.json'
with open(label_path, 'r') as f:
    label = json.load(f)
seqs = [seq for seq in label.keys() if 'locomotion' not in seq]
for seq in seqs:
    seq_path = base_dir / seq
    with open(seq_path, 'rb') as f:
        seq_label = pickle.load(f, encoding='latin1')
    num_frames = seq_label['pose_est_trans'].shape[0] // 4
    seq_labels = label[seq]
    walk_seg1 = {
        "act_cat": ["walk"],
        "proc_label": "walk",
        "start_t": 0,
        "end_t": seq_labels[0]['start_t'],
    }
    walk_seg2 = {
        "act_cat": ["walk"],
        "proc_label": "walk",
        "start_t": seq_labels[-1]['end_t'],
        "end_t": num_frames - 1,
    }
    label[seq] = [walk_seg1] + seq_labels + [walk_seg2]
with open(output_path, 'w') as f:
    json.dump(label, f, indent=4)
