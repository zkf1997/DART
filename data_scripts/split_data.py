from pathlib import Path
import numpy as np
import pickle

def contain_label(data, label):
    frame_labels = data['frame_labels']
    for frame_label in frame_labels:
        if label in frame_label['act_cat']:
            return True
    return False

full_data_path = Path('./data/seq_data/train.pkl')
with open(full_data_path, 'rb') as f:
    full_data = pickle.load(f)

babel_data = [data for data in full_data if data['data_source'] == 'babel']
export_path = full_data_path.parent / 'babel_train.pkl'
with open(export_path, 'wb') as f:
    pickle.dump(babel_data, f)

babel_run_data = [data for data in babel_data if contain_label(data, 'run')]
export_path = full_data_path.parent / 'babel_run_train.pkl'
with open(export_path, 'wb') as f:
    pickle.dump(babel_run_data, f)

