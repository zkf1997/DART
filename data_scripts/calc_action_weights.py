import json
from pathlib import Path

from config_files.data_paths import *

# load babel data
raw_dataset_path = amass_dir / 'smplx_g/'
output_path = './data/seq_data'
d_folder = babel_dir
splits = ['train']
action_statistics = {}
babel = {}
for spl in splits:
    babel[spl] = json.load(open(d_folder / (spl + '.json')))
    for sid in babel[spl]:
        if 'frame_ann' in babel[spl][sid] and babel[spl][sid]['frame_ann'] is not None:
            frame_labels = babel[spl][sid]['frame_ann']['labels']
            # process the transition labels, concatenate it with the target action
            for seg in frame_labels:
                act_cat_list = seg['act_cat']
                duration = seg['end_t'] - seg['start_t']
                for act_cat in act_cat_list:
                    if act_cat not in action_statistics:
                        action_statistics[act_cat] = {'total_weight': 1, 'total_len': 0}
                    action_statistics[act_cat]['total_len'] += duration
        else:  # the sequence has only sequence label, which means the sequence has only one action
            frame_labels = babel[spl][sid]['seq_ann']['labels']  # onle one element
            duration = babel[spl][sid]['dur']
            act_cat_list = frame_labels[0]['act_cat']
            for act_cat in act_cat_list:
                if act_cat not in action_statistics:
                    action_statistics[act_cat] = {'total_weight': 1, 'total_len': 0}
                action_statistics[act_cat]['total_len'] += duration

for act_cat in action_statistics:
    action_statistics[act_cat]['weight'] = action_statistics[act_cat]['total_weight'] / action_statistics[act_cat]['total_len']

# sort according to total length
action_statistics = dict(sorted(action_statistics.items(), key=lambda item: item[1]['total_len'], reverse=True))

export_path = Path('./data/action_statistics.json')
with export_path.open('w') as f:
    json.dump(action_statistics, f, indent=4)
print(f'Action statistics exported to {export_path}')