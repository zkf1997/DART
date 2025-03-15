import pickle
import json
from pathlib import Path
import numpy as np

dataset_dir = Path('/home/kaizhao/projects/mdm/hml3d_filter')
with open(dataset_dir / 'test_dataset.pkl', 'rb') as f:
    dataset = pickle.load(f)

text_prompts = []
for data in dataset:
    text, length = data[2], data[-2]
    text_prompt = f'{text}*{length}'
    text_prompts.append(text_prompt)

texts = [data[2] for data in dataset]
lengths = [data[-2] for data in dataset]

with open(dataset_dir / 'raw_text_length.pkl', 'wb') as f:
    pickle.dump({'texts': texts, 'lengths': lengths, 'size':len(dataset)}, f)

with open(dataset_dir / 'raw_prompt.pkl', 'wb') as f:
    pickle.dump(text_prompts, f)


import ast
fps = 20
text_prompts=[]
for seq_idx in range(len(dataset)):
    # print(f'Processing sequence {seq_idx}')
    raw_duration = dataset[seq_idx][-2] / fps
    decompose_path = dataset_dir / f'decompose/{seq_idx}.json'
    with open(decompose_path, 'r') as f:
        decompose = json.load(f)
    action_list = ast.literal_eval(decompose['decomposed'])
    text_prompt = []
    duration_sum = 0
    for action in action_list:
        action_text = action[0]
        duration_sum += action[1]
        action_duration = int(action[1] * fps)
        text_prompt.append(f'{action_text}*{action_duration}')
    if np.abs(duration_sum - raw_duration) > 0.1:
        print(f'Warning: Duration mismatch in sequence {seq_idx} with raw duration {raw_duration} and decomposed duration {duration_sum}')
        # break
    text_prompts.append('#$'.join(text_prompt))
with open(dataset_dir / 'decomposed_prompt.pkl', 'wb') as f:
    pickle.dump(text_prompts, f)

# fps = 30
# text_prompts=[]
# for seq_idx in range(len(dataset)):
#     print(f'Processing sequence {seq_idx}')
#     cfg_path = dataset_dir / f'verb/{seq_idx}.json'
#     # if not Path(cfg_path).exists():
#     #     continue
#     with open(cfg_path, 'r') as f:
#         seq_cfg = json.load(f)
#     verb_list = seq_cfg['verb']
#     duration_list = seq_cfg['duration']
#     num_verb = len(verb_list)
#     text_prompt = []
#     for action_idx in range(num_verb):
#         action_text = verb_list[action_idx]
#         action_duration = int(duration_list[action_idx] * fps)
#         text_prompt.append(f'{action_text}*{action_duration}')
#     text_prompts.append('#$'.join(text_prompt))
# with open(dataset_dir / 'verb_prompt.pkl', 'wb') as f:
#     pickle.dump(text_prompts, f)
