import pickle
import json

with open('/home/kaizhao/projects/mdm/hml3d_filter/raw_prompt.pkl', 'rb') as f:
    raw_prompt = pickle.load(f)

json_file = []
for seq_idx, prompt in enumerate(raw_prompt):
    text, length = prompt.split('*')
    json_file.append({
        "id": str(seq_idx),
        "scenario": "short",
        "text": [text],
        "lengths": [length]
    })
with open('./dataset/eval_hml3d_filter.json', 'w') as f:
    json.dump(json_file, f, indent=4)