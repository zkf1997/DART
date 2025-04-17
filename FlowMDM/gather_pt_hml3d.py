from pathlib import Path
import torch
import json
import numpy as np
import pickle

mean_path = './dataset/HumanML3D/Mean.npy'
std_path = './dataset/HumanML3D/Std.npy'
mean = np.load(mean_path).reshape((1, -1))
std = np.load(std_path).reshape((1, -1))

with open('./dataset/eval_hml3d_filter.json', 'r') as f:
    eval_data = json.load(f)

src_dir = Path('./results/humanml/FlowMDM/evaluation_precomputed/Motion_FlowMDM_000500000_gscale2.5_fasteval_hml3d_filter_s10')
for rep_dir in src_dir.glob('./*'):
    generation = []
    for seq_idx in range(len(eval_data)):
        seq_path = rep_dir / f'{seq_idx:02d}.pt'
        seq_data = torch.load(seq_path).cpu().squeeze(0).squeeze(1).permute(1, 0)
        seq_data = seq_data.numpy()
        seq_data = seq_data * std + mean
        # np.save(rep_dir / f'{seq_idx:02d}.npy', seq_data)
        generation.append(seq_data)
        # break
    with open(rep_dir / 'generation.pkl', 'wb') as f:
        pickle.dump(generation, f)


