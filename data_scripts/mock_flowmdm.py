from pathlib import Path
import json

with open('/home/kaizhao/projects/multiskill/data/long_seq_5.json', 'r') as f:
    all_configs = json.load(f)
output_dir = '/home/kaizhao/projects/flowmdm/results/babel/FlowMDM/evaluation_precomputed/long_seq_5'

num_rep = 3
for rep in range(num_rep):
    rep_dir = Path(output_dir) / f'{rep:02d}'
    rep_dir.mkdir(parents=True, exist_ok=True)

for config in all_configs:
    config_file = Path(output_dir) / '00' / (config['id'] + '_kwargs.json')
    with open(config_file, 'w') as f:
        json.dump(config, f)