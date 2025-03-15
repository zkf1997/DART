import time

import smplx
import torch
from torch.utils import data
import numpy as np
import os
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm
import spacy
from omegaconf import DictConfig, OmegaConf
import pickle

from torch.utils.data._utils.collate import default_collate
from data_loaders.humanml.utils.word_vectorizer import WordVectorizer
from data_loaders.humanml.utils.get_opt import get_opt
from pytorch3d import transforms
from pathlib import Path
import json
import multiprocessing as mp

from config_files.data_paths import *
from utils.smpl_utils import *
from utils.misc_util import have_overlap, get_overlap, load_and_freeze_clip, encode_text, compose_texts_with_and
import torch.nn.functional as F

# import spacy

def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)


'''For use of training text-2-motion generative model'''
class Text2MotionDataset(data.Dataset):
    def __init__(self, opt, mean, std, split_file, w_vectorizer):
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        min_motion_len = 40 if self.opt.dataset_name =='t2m' else 24

        joints_num = opt.joints_num

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                if (len(motion)) < min_motion_len or (len(motion) >= 200):
                    continue
                text_data = []
                flag = False
                with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag*20) : int(to_tag*20)]
                                if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
                                    continue
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                data_dict[new_name] = {'motion': n_motion,
                                                       'length': len(n_motion),
                                                       'text':[text_dict]}
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {'motion': motion,
                                       'length': len(motion),
                                       'text':text_data}
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except:
                # Some motion may not exist in KIT dataset
                pass


        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        if opt.is_train:
            # root_rot_velocity (B, seq_len, 1)
            std[0:1] = std[0:1] / opt.feat_bias
            # root_linear_velocity (B, seq_len, 2)
            std[1:3] = std[1:3] / opt.feat_bias
            # root_y (B, seq_len, 1)
            std[3:4] = std[3:4] / opt.feat_bias
            # ric_data (B, seq_len, (joint_num - 1)*3)
            std[4: 4 + (joints_num - 1) * 3] = std[4: 4 + (joints_num - 1) * 3] / 1.0
            # rot_data (B, seq_len, (joint_num - 1)*6)
            std[4 + (joints_num - 1) * 3: 4 + (joints_num - 1) * 9] = std[4 + (joints_num - 1) * 3: 4 + (
                        joints_num - 1) * 9] / 1.0
            # local_velocity (B, seq_len, joint_num*3)
            std[4 + (joints_num - 1) * 9: 4 + (joints_num - 1) * 9 + joints_num * 3] = std[
                                                                                       4 + (joints_num - 1) * 9: 4 + (
                                                                                                   joints_num - 1) * 9 + joints_num * 3] / 1.0
            # foot contact (B, seq_len, 4)
            std[4 + (joints_num - 1) * 9 + joints_num * 3:] = std[
                                                              4 + (joints_num - 1) * 9 + joints_num * 3:] / opt.feat_bias

            assert 4 + (joints_num - 1) * 9 + joints_num * 3 + 4 == mean.shape[-1]
            np.save(pjoin(opt.meta_dir, 'mean.npy'), mean)
            np.save(pjoin(opt.meta_dir, 'std.npy'), std)

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.opt.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d"%self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']

        if len(tokens) < self.opt.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.opt.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        len_gap = (m_length - self.max_length) // self.opt.unit_length

        if self.opt.is_train:
            if m_length != self.max_length:
            # print("Motion original length:%d_%d"%(m_length, len(motion)))
                if self.opt.unit_length < 10:
                    coin2 = np.random.choice(['single', 'single', 'double'])
                else:
                    coin2 = 'single'
                if len_gap == 0 or (len_gap == 1 and coin2 == 'double'):
                    m_length = self.max_length
                    idx = random.randint(0, m_length - self.max_length)
                    motion = motion[idx:idx+self.max_length]
                else:
                    if coin2 == 'single':
                        n_m_length = self.max_length + self.opt.unit_length * len_gap
                    else:
                        n_m_length = self.max_length + self.opt.unit_length * (len_gap - 1)
                    idx = random.randint(0, m_length - n_m_length)
                    motion = motion[idx:idx + self.max_length]
                    m_length = n_m_length
                # print(len_gap, idx, coin2)
        else:
            if self.opt.unit_length < 10:
                coin2 = np.random.choice(['single', 'single', 'double'])
            else:
                coin2 = 'single'

            if coin2 == 'double':
                m_length = (m_length // self.opt.unit_length - 1) * self.opt.unit_length
            elif coin2 == 'single':
                m_length = (m_length // self.opt.unit_length) * self.opt.unit_length
            idx = random.randint(0, len(motion) - m_length)
            motion = motion[idx:idx+m_length]

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length


'''For use of training text motion matching model, and evaluations'''
class Text2MotionDatasetV2(data.Dataset):
    def __init__(self, opt, mean, std, split_file, w_vectorizer):
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = opt.max_motion_length
        min_motion_len = 40 if self.opt.dataset_name =='t2m' else 24

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        # id_list = id_list[:200]

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                if (len(motion)) < min_motion_len or (len(motion) >= 200):
                    continue
                text_data = []
                flag = False
                with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag*20) : int(to_tag*20)]
                                if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
                                    continue
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                data_dict[new_name] = {'motion': n_motion,
                                                       'length': len(n_motion),
                                                       'text':[text_dict]}
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {'motion': motion,
                                       'length': len(motion),
                                       'text': text_data}
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except:
                pass

        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d"%self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']

        if len(tokens) < self.opt.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.opt.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        # Crop the motions in to times of 4, and introduce small variations
        if self.opt.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        if coin2 == 'double':
            m_length = (m_length // self.opt.unit_length - 1) * self.opt.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.opt.unit_length) * self.opt.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx+m_length]

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)
        # print(word_embeddings.shape, motion.shape)
        # print(tokens)
        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens)


'''For use of training baseline'''
class Text2MotionDatasetBaseline(data.Dataset):
    def __init__(self, opt, mean, std, split_file, w_vectorizer):
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = opt.max_motion_length
        min_motion_len = 40 if self.opt.dataset_name =='t2m' else 24

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        # id_list = id_list[:200]

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                if (len(motion)) < min_motion_len or (len(motion) >= 200):
                    continue
                text_data = []
                flag = False
                with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag*20) : int(to_tag*20)]
                                if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
                                    continue
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                data_dict[new_name] = {'motion': n_motion,
                                                       'length': len(n_motion),
                                                       'text':[text_dict]}
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {'motion': motion,
                                       'length': len(motion),
                                       'text': text_data}
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except:
                pass

        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d"%self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']

        if len(tokens) < self.opt.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.opt.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        len_gap = (m_length - self.max_length) // self.opt.unit_length

        if m_length != self.max_length:
            # print("Motion original length:%d_%d"%(m_length, len(motion)))
            if self.opt.unit_length < 10:
                coin2 = np.random.choice(['single', 'single', 'double'])
            else:
                coin2 = 'single'
            if len_gap == 0 or (len_gap == 1 and coin2 == 'double'):
                m_length = self.max_length
                s_idx = random.randint(0, m_length - self.max_length)
            else:
                if coin2 == 'single':
                    n_m_length = self.max_length + self.opt.unit_length * len_gap
                else:
                    n_m_length = self.max_length + self.opt.unit_length * (len_gap - 1)
                s_idx = random.randint(0, m_length - n_m_length)
                m_length = n_m_length
        else:
            s_idx = 0

        src_motion = motion[s_idx: s_idx + m_length]
        tgt_motion = motion[s_idx: s_idx + self.max_length]

        "Z Normalization"
        src_motion = (src_motion - self.mean) / self.std
        tgt_motion = (tgt_motion - self.mean) / self.std

        if m_length < self.max_motion_length:
            src_motion = np.concatenate([src_motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)
        # print(m_length, src_motion.shape, tgt_motion.shape)
        # print(word_embeddings.shape, motion.shape)
        # print(tokens)
        return word_embeddings, caption, sent_len, src_motion, tgt_motion, m_length


class MotionDatasetV2(data.Dataset):
    def __init__(self, opt, mean, std, split_file):
        self.opt = opt
        joints_num = opt.joints_num

        self.data = []
        self.lengths = []
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                if motion.shape[0] < opt.window_size:
                    continue
                self.lengths.append(motion.shape[0] - opt.window_size)
                self.data.append(motion)
            except:
                # Some motion may not exist in KIT dataset
                pass

        self.cumsum = np.cumsum([0] + self.lengths)

        if opt.is_train:
            # root_rot_velocity (B, seq_len, 1)
            std[0:1] = std[0:1] / opt.feat_bias
            # root_linear_velocity (B, seq_len, 2)
            std[1:3] = std[1:3] / opt.feat_bias
            # root_y (B, seq_len, 1)
            std[3:4] = std[3:4] / opt.feat_bias
            # ric_data (B, seq_len, (joint_num - 1)*3)
            std[4: 4 + (joints_num - 1) * 3] = std[4: 4 + (joints_num - 1) * 3] / 1.0
            # rot_data (B, seq_len, (joint_num - 1)*6)
            std[4 + (joints_num - 1) * 3: 4 + (joints_num - 1) * 9] = std[4 + (joints_num - 1) * 3: 4 + (
                        joints_num - 1) * 9] / 1.0
            # local_velocity (B, seq_len, joint_num*3)
            std[4 + (joints_num - 1) * 9: 4 + (joints_num - 1) * 9 + joints_num * 3] = std[
                                                                                       4 + (joints_num - 1) * 9: 4 + (
                                                                                                   joints_num - 1) * 9 + joints_num * 3] / 1.0
            # foot contact (B, seq_len, 4)
            std[4 + (joints_num - 1) * 9 + joints_num * 3:] = std[
                                                              4 + (joints_num - 1) * 9 + joints_num * 3:] / opt.feat_bias

            assert 4 + (joints_num - 1) * 9 + joints_num * 3 + 4 == mean.shape[-1]
            np.save(pjoin(opt.meta_dir, 'mean.npy'), mean)
            np.save(pjoin(opt.meta_dir, 'std.npy'), std)

        self.mean = mean
        self.std = std
        print("Total number of motions {}, snippets {}".format(len(self.data), self.cumsum[-1]))

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return self.cumsum[-1]

    def __getitem__(self, item):
        if item != 0:
            motion_id = np.searchsorted(self.cumsum, item) - 1
            idx = item - self.cumsum[motion_id] - 1
        else:
            motion_id = 0
            idx = 0
        motion = self.data[motion_id][idx:idx+self.opt.window_size]
        "Z Normalization"
        motion = (motion - self.mean) / self.std

        return motion


class RawTextDataset(data.Dataset):
    def __init__(self, opt, mean, std, text_file, w_vectorizer):
        self.mean = mean
        self.std = std
        self.opt = opt
        self.data_dict = []
        self.nlp = spacy.load('en_core_web_sm')

        with cs.open(text_file) as f:
            for line in f.readlines():
                word_list, pos_list = self.process_text(line.strip())
                tokens = ['%s/%s'%(word_list[i], pos_list[i]) for i in range(len(word_list))]
                self.data_dict.append({'caption':line.strip(), "tokens":tokens})

        self.w_vectorizer = w_vectorizer
        print("Total number of descriptions {}".format(len(self.data_dict)))


    def process_text(self, sentence):
        sentence = sentence.replace('-', '')
        doc = self.nlp(sentence)
        word_list = []
        pos_list = []
        for token in doc:
            word = token.text
            if not word.isalpha():
                continue
            if (token.pos_ == 'NOUN' or token.pos_ == 'VERB') and (word != 'left'):
                word_list.append(token.lemma_)
            else:
                word_list.append(word)
            pos_list.append(token.pos_)
        return word_list, pos_list

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        data = self.data_dict[item]
        caption, tokens = data['caption'], data['tokens']

        if len(tokens) < self.opt.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.opt.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        return word_embeddings, pos_one_hots, caption, sent_len

class TextOnlyDataset(data.Dataset):
    def __init__(self, opt, mean, std, split_file):
        self.mean = mean
        self.std = std
        self.opt = opt
        self.data_dict = []
        self.max_length = 20
        self.pointer = 0
        self.fixed_length = 120


        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        # id_list = id_list[:200]

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                text_data = []
                flag = False
                with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                data_dict[new_name] = {'text':[text_dict]}
                                new_name_list.append(new_name)
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {'text': text_data}
                    new_name_list.append(name)
            except:
                pass

        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = new_name_list

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        text_list = data['text']

        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']
        return None, None, caption, None, np.array([0]), self.fixed_length, None
        # fixed_length can be set from outside before sampling

# A wrapper class for t2m original dataset for MDM purposes
class HumanML3D(data.Dataset):
    def __init__(self, mode, datapath='./dataset/humanml_opt.txt', split="train", **kwargs):
        self.mode = mode
        
        self.dataset_name = 't2m'
        self.dataname = 't2m'

        # Configurations of T2M dataset and KIT dataset is almost the same
        abs_base_path = f'.'
        dataset_opt_path = pjoin(abs_base_path, datapath)
        device = None  # torch.device('cuda:4') # This param is not in use in this context
        opt = get_opt(dataset_opt_path, device)
        opt.meta_dir = pjoin(abs_base_path, opt.meta_dir)
        opt.motion_dir = pjoin(abs_base_path, opt.motion_dir)
        opt.text_dir = pjoin(abs_base_path, opt.text_dir)
        opt.model_dir = pjoin(abs_base_path, opt.model_dir)
        opt.checkpoints_dir = pjoin(abs_base_path, opt.checkpoints_dir)
        opt.data_root = pjoin(abs_base_path, opt.data_root)
        opt.save_root = pjoin(abs_base_path, opt.save_root)
        opt.meta_dir = './dataset'
        self.opt = opt
        print('Loading dataset %s ...' % opt.dataset_name)

        if mode == 'gt':
            # used by T2M models (including evaluators)
            self.mean = np.load(pjoin(opt.meta_dir, f'{opt.dataset_name}_mean.npy'))
            self.std = np.load(pjoin(opt.meta_dir, f'{opt.dataset_name}_std.npy'))
        elif mode in ['train', 'eval', 'text_only']:
            # used by our models
            self.mean = np.load(pjoin(opt.data_root, 'Mean.npy'))
            self.std = np.load(pjoin(opt.data_root, 'Std.npy'))

        if mode == 'eval':
            # used by T2M models (including evaluators)
            # this is to translate their norms to ours
            self.mean_for_eval = np.load(pjoin(opt.meta_dir, f'{opt.dataset_name}_mean.npy'))
            self.std_for_eval = np.load(pjoin(opt.meta_dir, f'{opt.dataset_name}_std.npy'))

        self.split_file = pjoin(opt.data_root, f'{split}.txt')
        if mode == 'text_only':
            self.t2m_dataset = TextOnlyDataset(self.opt, self.mean, self.std, self.split_file)
        else:
            self.w_vectorizer = WordVectorizer(pjoin(abs_base_path, 'glove'), 'our_vab')
            self.t2m_dataset = Text2MotionDatasetV2(self.opt, self.mean, self.std, self.split_file, self.w_vectorizer)
            self.num_actions = 1 # dummy placeholder

        assert len(self.t2m_dataset) > 1, 'You loaded an empty dataset, ' \
                                          'it is probably because your data dir has only texts and no motions.\n' \
                                          'To train and evaluate MDM you should get the FULL data as described ' \
                                          'in the README file.'

    def __getitem__(self, item):
        return self.t2m_dataset.__getitem__(item)

    def __len__(self):
        return self.t2m_dataset.__len__()

# A wrapper class for t2m original dataset for MDM purposes
class KIT(HumanML3D):
    def __init__(self, mode, datapath='./dataset/kit_opt.txt', split="train", **kwargs):
        super(KIT, self).__init__(mode, datapath, split, **kwargs)

class Text2MotionPrimitiveDataset(data.Dataset):
    def __init__(self, dataset_name='babel_mp', dataset_path='./data/mp_data/Canonicalized_h2_f8_num1_fps30/', split="train", load_data=True, **kwargs):
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.split = split

        self.primitive_utility = PrimitiveUtility(device='cpu')
        self.motion_repr = self.primitive_utility.motion_repr

        cfg_path = Path(dataset_path, 'config.yaml')
        with open(cfg_path, 'r') as f:
            self.cfg = OmegaConf.load(f)

        mean_std_path = pjoin(dataset_path, 'mean_std.pkl')
        with open(mean_std_path, 'rb') as f:
            self.mean_std = pickle.load(f)
        mean_dict = {}
        std_dict = {}
        for key in self.mean_std:
            mean_dict[key] = self.mean_std[key]['mean']
            std_dict[key] = self.mean_std[key]['std']
        self.tensor_mean = self.dict_to_tensor(mean_dict).reshape(1, 1, -1)
        self.tensor_std = self.dict_to_tensor(std_dict).reshape(1, 1, -1)
        self.tensor_mean_device_dict = {}

        if load_data:
            self.split_file = pjoin(dataset_path, f'{split}.pkl')
            with open(self.split_file, 'rb') as f:
                dataset = pickle.load(f)
            # dataset = dataset[:128]
            # dataset = [data for data in dataset if len(data['texts']) > 0]
            self.dataset = dataset
        else:
            self.dataset = []

        print('num of primitives: ', self.__len__())

    def dict_to_tensor(self, data_dict):
        return self.primitive_utility.dict_to_tensor(data_dict)


    def tensor_to_dict(self, tensor):
        return self.primitive_utility.tensor_to_dict(tensor)

    def get_mean_std_by_device(self, device):
        if device not in self.tensor_mean_device_dict:
            self.tensor_mean_device_dict[device] = (self.tensor_mean.to(device=device), self.tensor_std.to(device=device))
        return self.tensor_mean_device_dict[device]

    def normalize(self, tensor):
        tensor_mean, tensor_std = self.get_mean_std_by_device(tensor.device)
        return (tensor - tensor_mean) / tensor_std  # [B, T, D]

    def denormalize(self, tensor):
        tensor_mean, tensor_std = self.get_mean_std_by_device(tensor.device)
        return tensor * tensor_std + tensor_mean  # [B, T, D]

    def __getitem__(self, item):
        data = self.dataset[item]
        data_out = {}
        data_out['gender'] = data['gender'] if isinstance(data['gender'], str) else data['gender'].item()
        data_out['texts'] = data['texts']
        for key in data:
            if key in ['gender', 'texts']:
                continue
            data_out[key] = torch.tensor(data[key]).to(dtype=torch.float32)
        if len(data_out['betas'].shape) == 1:
            data_out['betas'] = data_out['betas'].unsqueeze(0).repeat(self.cfg.history_length + self.cfg.future_length, 1)  # [T, 10]
        data_out['joints'] = data_out['joints'].reshape(-1, 22 * 3)
        data_out['joints_delta'] = data_out['joints_delta'].reshape(-1, 22 * 3)
        if 'poses_6d' not in data_out:
            data_out['poses_6d'] = transforms.matrix_to_rotation_6d(
                transforms.axis_angle_to_matrix(data_out['poses'].reshape(-1, 22, 3)))
        data_out['poses_6d'] = data_out['poses_6d'].reshape(-1, 22 * 6)
        if 'global_orient_delta_6d' not in data_out:
            data_out['global_orient_delta_6d'] = transforms.matrix_to_rotation_6d(
                transforms.axis_angle_to_matrix(data_out['global_orient_delta']))
        data_out['global_orient_delta_6d'] = data_out['global_orient_delta_6d'].reshape(-1, 6)

        motion_tensor_normalized = self.normalize(self.dict_to_tensor(data_out).unsqueeze(0))  # [1, T, D]
        motion_tensor_normalized = motion_tensor_normalized.permute(2, 0, 1)  # [D, 1, T]
        # print('tensor shape: ', motion_tensor_normalized.shape)
        history_mask = torch.zeros_like(motion_tensor_normalized, dtype=torch.bool)
        history_mask[:, :, :self.cfg.history_length] = True
        history_motion = torch.zeros_like(motion_tensor_normalized, dtype=torch.float32)
        history_motion[:, :, :self.cfg.history_length] = motion_tensor_normalized[:, :, :self.cfg.history_length]

        output = {
            'text': random.choice(data_out['texts']) if len(data_out['texts']) > 0 else '',
            'gender': data_out['gender'],
            'betas': data_out['betas'],
            'motion_tensor_normalized': motion_tensor_normalized,
            'history_motion': history_motion,
            'history_mask': history_mask,
            'history_length': self.cfg.history_length,
            'future_length': self.cfg.future_length,
        }
        return output

    def __len__(self):
        return len(self.dataset)


class RestPrimitiveDataset(Text2MotionPrimitiveDataset):
    def __init__(self, dataset_path='./data/mp_data/Canonicalized_h2_f8_num1_fps30/', rest_pose_path='./data/rest_pose.pkl',
                 batch_size=16, gender='male', betas=None, texts=None,  #text: list of str lists
                 **kwargs):
        super().__init__(dataset_name='rest_mp', dataset_path=dataset_path,
                         split='test', load_data=False,
                         **kwargs)
        self.batch_size = batch_size
        with open(rest_pose_path, 'rb') as f:
            self.rest_pose = pickle.load(f)
        self.rest_pose = transforms.axis_angle_to_matrix(self.rest_pose.reshape(1, 21, 3))  # [1, 21, 3, 3]
        self.update_rest_seed(gender, betas, texts)

    def update_rest_seed(self, gender='male', betas=None, texts=None):
        batch_size = self.batch_size
        seq_length = self.cfg.history_length + self.cfg.future_length + 1
        if betas is None:
            betas = torch.zeros(seq_length, 10, dtype=torch.float32)
        else:
            betas = torch.tensor(betas, dtype=torch.float32).expand(seq_length, -1)
        yup_to_zup = torch.eye(3)
        yup_to_zup[:3, :3] = torch.tensor([[1, 0, 0],
                                           [0, 0, -1],
                                           [0, 1, 0]])
        yup_to_zup = yup_to_zup.unsqueeze(0).repeat(seq_length, 1, 1)
        transl = torch.zeros(seq_length, 3, dtype=torch.float32)
        rest_motion = torch.tensor(self.rest_pose, dtype=torch.float32).repeat(seq_length, 1, 1, 1)
        primitive_dict = {
            'transl': transl.unsqueeze(0),
            'global_orient': yup_to_zup.unsqueeze(0),
            'body_pose': rest_motion.unsqueeze(0),
            'betas': betas.unsqueeze(0),
            'gender': gender,
            'transf_rotmat': torch.eye(3).unsqueeze(0),
            'transf_transl': torch.zeros(1, 1, 3),
        }
        transf_rotmat, transf_transl, canonical_body_param_dict = self.primitive_utility.canonicalize(primitive_dict)
        rest_motion_features = self.primitive_utility.calc_features(canonical_body_param_dict)
        for key in rest_motion_features:
            rest_motion_features[key] = rest_motion_features[key][0, :seq_length - 1, :]

        self.dataset = []
        for idx in range(batch_size):
            data = deepcopy(rest_motion_features)
            data['gender'] = gender
            data['betas'] = betas[:-1, :]
            if texts is None:
                data['texts'] = ['stand']
            else:
                data['texts'] = texts[idx]
            self.dataset.append(data)

    def __len__(self):
        return len(self.dataset)


class PrimitiveSequenceDataset:
    def __init__(self, dataset_name='babel_mp_seq',
                 dataset_path='./data/mp_data/Canonicalized_h2_f8_num1_fps30/',
                 cfg_path='./config_files/config_hydra/motion_primitive/mp_2_8.yaml',
                 split="train",
                 device='cuda',
                 **kwargs):
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.split = split
        self.device = device

        self.primitive_utility = PrimitiveUtility(device=self.device)
        self.motion_repr = self.primitive_utility.motion_repr

        # cfg_path = Path(dataset_path, 'config.yaml')
        with open(cfg_path, 'r') as f:
            self.cfg = OmegaConf.load(f)
        self.target_fps = self.cfg.fps
        self.downsample_rate = 120 // self.target_fps
        self.history_length = self.cfg.history_length
        self.future_length = self.cfg.future_length
        self.primitive_length = self.history_length + self.future_length
        self.num_primitive = self.cfg.num_primitive
        self.seq_length = self.history_length + self.future_length * self.num_primitive + 1

        mean_std_path = pjoin(dataset_path, 'mean_std.pkl')
        with open(mean_std_path, 'rb') as f:
            self.mean_std = pickle.load(f)
        mean_dict = {}
        std_dict = {}
        for key in self.mean_std:
            mean_dict[key] = self.mean_std[key]['mean']
            std_dict[key] = self.mean_std[key]['std']
        self.tensor_mean = self.dict_to_tensor(mean_dict).reshape(1, 1, -1)
        self.tensor_std = self.dict_to_tensor(std_dict).reshape(1, 1, -1)
        self.tensor_mean_device_dict = {}

        cache_path = pjoin(dataset_path, f'{split}_cache.pkl')
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                dataset = pickle.load(f)
        else:
            dataset = []
            seq_info_path = pjoin(dataset_path, 'seq_info.json')
            with open(seq_info_path, 'r') as f:
                seq_info_dataset = json.load(f)
            for seq_info in tqdm(seq_info_dataset[split]):
                seq_path = seq_info['seq_path']
                if not os.path.exists(seq_path):
                    continue
                # if not 'frame_labels' in seq_info:
                #     continue

                seq_data = dict(np.load(seq_path, allow_pickle=True))
                fps = seq_data['mocap_frame_rate']
                assert fps == 120.0
                motion_data = {}
                motion_data['trans'] = torch.from_numpy(seq_data['trans'][::self.downsample_rate].astype(np.float32))
                motion_data['poses'] = torch.from_numpy(seq_data['poses'][::self.downsample_rate, :66].astype(np.float32))
                motion_data['betas'] = torch.from_numpy(seq_data['betas'][:10].astype(np.float32))
                motion_data['gender'] = str(seq_data['gender'].item())
                if len(motion_data['trans']) < self.seq_length:
                    continue

                seq_data_dict = {'motion': motion_data}
                if 'frame_labels' in seq_info:
                    seq_data_dict['frame_labels'] = seq_info['frame_labels']
                dataset.append(seq_data_dict)
            print('num of sequences: ', len(dataset))
            with open(cache_path, 'wb') as f:
                pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
        self.dataset = dataset
        self.data_index = 0
        self.reset()
        print('num of sequences: ', len(self.dataset))

    def dict_to_tensor(self, data_dict):
        return self.primitive_utility.dict_to_tensor(data_dict)


    def tensor_to_dict(self, tensor):
        return self.primitive_utility.tensor_to_dict(tensor)

    def get_mean_std_by_device(self, device):
        if device not in self.tensor_mean_device_dict:
            self.tensor_mean_device_dict[device] = (self.tensor_mean.to(device=device), self.tensor_std.to(device=device))
        return self.tensor_mean_device_dict[device]

    def normalize(self, tensor):
        tensor_mean, tensor_std = self.get_mean_std_by_device(tensor.device)
        return (tensor - tensor_mean) / tensor_std  # [B, T, D]

    def denormalize(self, tensor):
        tensor_mean, tensor_std = self.get_mean_std_by_device(tensor.device)
        return tensor * tensor_std + tensor_mean  # [B, T, D]

    def get_primitive(self, seq_data, start_frame, end_frame):
        motion_data = seq_data['motion']
        primitive_dict = {
            'gender': motion_data['gender'],
            'betas': motion_data['betas'].expand(1, self.primitive_length + 1, 10),
            'transl': motion_data['trans'][start_frame:end_frame + 1].unsqueeze(0),  # include one more frame for delta feature calculation
            'global_orient': transforms.axis_angle_to_matrix(motion_data['poses'][start_frame:end_frame + 1, :3].unsqueeze(0)),
            'body_pose': transforms.axis_angle_to_matrix(
                motion_data['poses'][start_frame:end_frame + 1, 3:66].unsqueeze(0).reshape(1, end_frame - start_frame + 1, 21, 3)
            ),
            'transf_rotmat': torch.eye(3).unsqueeze(0),
            'transf_transl': torch.zeros(1, 1, 3),
        }
        primitive_dict = tensor_dict_to_device(primitive_dict, self.device)
        # _, _, canonicalized_primitive_dict = self.primitive_utility.canonicalize(primitive_dict)
        # transf_rotmat, transf_transl = canonicalized_primitive_dict['transf_rotmat'], canonicalized_primitive_dict[
        #     'transf_transl']
        # feature_dict = self.primitive_utility.calc_features(canonicalized_primitive_dict)
        # feature_dict['transl'] = feature_dict['transl'][:, :-1, :]  # [1, T, 3]
        # feature_dict['poses_6d'] = feature_dict['poses_6d'][:, :-1, :]  # [1, T, 66]
        # feature_dict['joints'] = feature_dict['joints'][:, :-1, :]  # [1, T, 22 * 3]
        # motion_tensor_normalized = self.normalize(self.dict_to_tensor(feature_dict))  # [1, T, D]
        # motion_tensor_normalized = motion_tensor_normalized.permute(2, 0, 1)  # [D, 1, T]
        # history_mask = torch.zeros_like(motion_tensor_normalized, dtype=torch.bool, device=self.device)
        # history_mask[:, :, :self.cfg.history_length] = True
        # history_motion = torch.zeros_like(motion_tensor_normalized, dtype=torch.float32, device=self.device)
        # history_motion[:, :, :self.cfg.history_length] = motion_tensor_normalized[:, :, :self.cfg.history_length]

        texts = []
        if 'frame_labels' in seq_data:
            future_start = (start_frame + self.history_length) / self.target_fps
            future_end = (start_frame + self.history_length + self.future_length - 1) / self.target_fps
            for seg in seq_data['frame_labels']:
                if have_overlap([seg['start_t'], seg['end_t']], [future_start, future_end]):
                    texts.append(seg['proc_label'])

        output = {
            'text': random.choice(texts) if len(texts) > 0 else '',
            'primitive_dict': primitive_dict,
            # 'gender': motion_data['gender'],
            # 'betas': motion_data['betas'].expand(self.primitive_length, 10),
            # 'motion_tensor_normalized': motion_tensor_normalized, # [D, 1, T]
            # 'history_motion': history_motion,
            # 'history_mask': history_mask,
            # 'history_length': self.cfg.history_length,
            # 'future_length': self.cfg.future_length,
        }
        return output

    def get_batch(self, batch_size=8):
        seq_list = []
        for _ in range(batch_size):
            if self.data_index >= len(self.dataset):
                self.reset()
            seq_data = self.dataset[self.data_index]
            self.data_index = self.data_index + 1

            num_frames = len(seq_data['motion']['trans'])
            start_frame = random.randint(0, num_frames - self.seq_length)  # [0, num_frames - seq_length], right end inclusive
            primitive_data_list = []
            for frame_idx in range(start_frame, start_frame + self.seq_length - self.future_length, self.future_length):
                primitive_data = self.get_primitive(seq_data, frame_idx, frame_idx + self.primitive_length)
                primitive_data_list.append(primitive_data)
            seq_list.append(primitive_data_list)

        # sort batch by gender
        batch = None
        for gender in ['female', 'male']:
            gender_idx = [idx for idx in range(len(seq_list)) if seq_list[idx][0]['primitive_dict']['gender'] == gender]
            if len(gender_idx) == 0:
                continue
            gender_seq_list = [seq_list[i] for i in gender_idx]
            gender_batch = []

            for primitive_idx in range(self.num_primitive):
                primitive_texts = [mp_seq[primitive_idx]['text'] for mp_seq in gender_seq_list]
                primitive_dict = {'gender': gender}
                for key in ['betas', 'transl', 'global_orient', 'body_pose', 'transf_rotmat', 'transf_transl']:
                    primitive_dict[key] = torch.cat([mp_seq[primitive_idx]['primitive_dict'][key] for mp_seq in gender_seq_list], dim=0)

                _, _, canonicalized_primitive_dict = self.primitive_utility.canonicalize(primitive_dict)
                transf_rotmat, transf_transl = canonicalized_primitive_dict['transf_rotmat'], canonicalized_primitive_dict[
                    'transf_transl']
                feature_dict = self.primitive_utility.calc_features(canonicalized_primitive_dict)
                feature_dict['transl'] = feature_dict['transl'][:, :-1, :]  # [B, T, 3]
                feature_dict['poses_6d'] = feature_dict['poses_6d'][:, :-1, :]  # [B, T, 66]
                feature_dict['joints'] = feature_dict['joints'][:, :-1, :]  # [B, T, 22 * 3]
                motion_tensor_normalized = self.normalize(self.dict_to_tensor(feature_dict))  # [B, T, D]
                motion_tensor_normalized = motion_tensor_normalized.permute(0, 2, 1).unsqueeze(2)  # [B, D, 1, T]
                history_mask = torch.zeros_like(motion_tensor_normalized, dtype=torch.bool, device=self.device)
                history_mask[..., :self.cfg.history_length] = True
                history_motion = torch.zeros_like(motion_tensor_normalized, dtype=torch.float32, device=self.device)
                history_motion[..., :self.cfg.history_length] = motion_tensor_normalized[..., :self.cfg.history_length]

                gender_batch.append(
                    {
                        'texts': primitive_texts,
                        'gender': [primitive_dict['gender']] * len(gender_seq_list),
                        'betas': primitive_dict['betas'][:, :-1, :10],
                        'motion_tensor_normalized': motion_tensor_normalized, # [B, D, 1, T]
                        'history_motion': history_motion,
                        'history_mask': history_mask,
                        'history_length': self.cfg.history_length,
                        'future_length': self.cfg.future_length,
                    }
                )

            if batch is None:
                batch = gender_batch
            else:  # concatenate different gender batch
                for primitive_idx in range(self.num_primitive):
                    for key in ['texts', 'gender']:
                        batch[primitive_idx][key] = batch[primitive_idx][key] + gender_batch[primitive_idx][key]
                    for key in ['betas', 'motion_tensor_normalized', 'history_motion', 'history_mask']:
                        batch[primitive_idx][key] = torch.cat([batch[primitive_idx][key], gender_batch[primitive_idx][key]], dim=0)

        return batch

    def get_batch_per_seq(self, batch_size=8):
        batch = []
        for _ in range(batch_size):
            if self.data_index >= len(self.dataset):
                self.reset()
            seq_data = self.dataset[self.data_index]
            self.data_index = self.data_index + 1

            num_frames = len(seq_data['motion']['trans'])
            start_frame = random.randint(0, num_frames - self.seq_length)  # [0, num_frames - seq_length], right end inclusive
            primitive_data_list = []
            for frame_idx in range(start_frame, start_frame + self.seq_length - self.future_length, self.future_length):
                primitive_data = self.get_primitive(seq_data, frame_idx, frame_idx + self.primitive_length)
                primitive_data_list.append(primitive_data)
            primitive_texts = [data['text'] for data in primitive_data_list]
            primitive_dict = {'gender': primitive_data_list[0]['primitive_dict']['gender']}
            for key in ['betas', 'transl', 'global_orient', 'body_pose', 'transf_rotmat', 'transf_transl']:
                primitive_dict[key] = torch.cat([data['primitive_dict'][key] for data in primitive_data_list], dim=0)

            _, _, canonicalized_primitive_dict = self.primitive_utility.canonicalize(primitive_dict)
            transf_rotmat, transf_transl = canonicalized_primitive_dict['transf_rotmat'], canonicalized_primitive_dict[
                'transf_transl']
            feature_dict = self.primitive_utility.calc_features(canonicalized_primitive_dict)
            feature_dict['transl'] = feature_dict['transl'][:, :-1, :]  # [num_primitive, T, 3]
            feature_dict['poses_6d'] = feature_dict['poses_6d'][:, :-1, :]  # [num_primitive, T, 66]
            feature_dict['joints'] = feature_dict['joints'][:, :-1, :]  # [num_primitive, T, 22 * 3]
            motion_tensor_normalized = self.normalize(self.dict_to_tensor(feature_dict))  # [num_primitive, T, D]
            motion_tensor_normalized = motion_tensor_normalized.permute(0, 2, 1).unsqueeze(2)  # [num_primitive, D, 1, T]
            history_mask = torch.zeros_like(motion_tensor_normalized, dtype=torch.bool, device=self.device)
            history_mask[..., :self.cfg.history_length] = True
            history_motion = torch.zeros_like(motion_tensor_normalized, dtype=torch.float32, device=self.device)
            history_motion[..., :self.cfg.history_length] = motion_tensor_normalized[..., :self.cfg.history_length]

            batch.append(
                {
                    'texts': primitive_texts,
                    'gender': primitive_dict['gender'],
                    'betas': primitive_dict['betas'][:, :-1, :10],
                    'motion_tensor_normalized': motion_tensor_normalized, # [num_primitive, D, 1, T]
                    'history_motion': history_motion,
                    'history_mask': history_mask,
                    'history_length': self.cfg.history_length,
                    'future_length': self.cfg.future_length,
                }
            )

        # sort batch by gender
        new_idx = []
        for gender in ['female', 'male']:
            new_idx = new_idx + [idx for idx in range(len(batch)) if batch[idx]['gender'] == gender]
        batch = [batch[i] for i in new_idx]

        return batch


    def __len__(self):
        return len(self.dataset)

    def reset(self):
        self.data_index = 0
        random.shuffle(self.dataset)


def get_subseq(seq_data,
               history_length=2, future_length=8,
               primitive_length=10, seq_length=11,
               target_fps=30, skip_text=False):
    num_frames = len(seq_data['motion']['trans'])
    start_frame = random.randint(0, num_frames - seq_length)  # [0, num_frames - seq_length], right end inclusive

    def get_primitive(seq_data, start_frame, end_frame):
        """end_frame included"""
        motion_data = seq_data['motion']
        primitive_dict = {
            'gender': motion_data['gender'],
            'betas': motion_data['betas'].expand(1, primitive_length + 1, 10),
            'transl': motion_data['trans'][start_frame:end_frame + 1].unsqueeze(0),
            # include one more frame for delta feature calculation
            'global_orient': transforms.axis_angle_to_matrix(
                motion_data['poses'][start_frame:end_frame + 1, :3].unsqueeze(0)),
            'body_pose': transforms.axis_angle_to_matrix(
                motion_data['poses'][start_frame:end_frame + 1, 3:66].unsqueeze(0).reshape(1,
                                                                                           end_frame - start_frame + 1,
                                                                                           21, 3)
            ),
            'transf_rotmat': torch.eye(3).unsqueeze(0),
            'transf_transl': torch.zeros(1, 1, 3),
        }
        # print('load data time: ', time.time() - self.time)
        # primitive_dict = tensor_dict_to_device(primitive_dict, self.device)
        # print('data to device time: ', time.time() - self.time)

        texts = []
        if not skip_text and 'frame_labels' in seq_data:
            future_start = (start_frame + history_length) / target_fps
            future_end = (start_frame + history_length + future_length - 1) / target_fps
            for seg in seq_data['frame_labels']:
                if have_overlap([seg['start_t'], seg['end_t']], [future_start, future_end]):
                    texts.append(seg['proc_label'])
        # print('text label time: ', time.time() - self.time)

        output = {
            'text': random.choice(texts) if len(texts) > 0 else '',
            'primitive_dict': primitive_dict,
        }
        return output

    primitive_data_list = []
    for frame_idx in range(start_frame, start_frame + seq_length - future_length, future_length):
        primitive_data = get_primitive(seq_data, frame_idx, frame_idx + primitive_length)
        primitive_data_list.append(primitive_data)
    return primitive_data_list

class WeightedPrimitiveSequenceDataset:
    def __init__(self, dataset_name='weighted_mp_seq',
                 dataset_path='./data/seq_data',
                 cfg_path='./config_files/config_hydra/motion_primitive/mp_2_8.yaml',
                 split="train",
                 device='cuda',
                 weight_scheme='uniform',
                 prob_static=0.0,
                 enforce_gender=None,
                 enforce_zero_beta=None,
                 load_data=True,
                 text_tolerance=0.0,
                 body_type='smplx',
                 **kwargs):
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.split = split
        self.device = device
        self.weight_scheme = weight_scheme
        self.prob_static = prob_static
        self.enforce_gender = enforce_gender
        self.enforce_zero_beta = enforce_zero_beta
        self.text_tolerance = text_tolerance
        print('enforce_gender: ', enforce_gender)
        print('enforce_zero_beta: ', enforce_zero_beta)

        self.primitive_utility = PrimitiveUtility(device=self.device, body_type=body_type)
        self.motion_repr = self.primitive_utility.motion_repr

        # cfg_path = Path(dataset_path, 'config.yaml')
        with open(cfg_path, 'r') as f:
            self.cfg = OmegaConf.load(f)
        self.target_fps = self.cfg.fps
        # self.downsample_rate = 120 // self.target_fps
        self.history_length = self.cfg.history_length
        self.future_length = self.cfg.future_length
        self.primitive_length = self.history_length + self.future_length
        self.num_primitive = self.cfg.num_primitive
        self.seq_length = self.history_length + self.future_length * self.num_primitive + 1

        if load_data:
            with open(pjoin(dataset_path, f'{split}.pkl'), 'rb') as f:
                dataset = pickle.load(f)
            dataset = [data for data in dataset if len(data['motion']['trans']) >= self.seq_length]
            for data in dataset:
                data['motion']['trans'] = torch.from_numpy(data['motion']['trans'].astype(np.float32))
                data['motion']['poses'] = torch.from_numpy(data['motion']['poses'].astype(np.float32))
                data['motion']['betas'] = torch.from_numpy(data['motion']['betas'].astype(np.float32))  # [10]
                if self.enforce_gender is not None:
                    data['motion']['gender'] = self.enforce_gender
                if self.enforce_zero_beta:
                    data['motion']['betas'] = torch.zeros_like(data['motion']['betas'])
                # if data['data_source'] == 'samp':
                #     data['motion']['gender'] = 'male'
            print('num of sequences: ', len(dataset))
            # assign sampling weights to each sequence

            with open('./data/action_statistics.json', 'r') as f:
                action_statistics = json.load(f)

            for data in dataset:
                if 'uniform' in weight_scheme:
                    data['weight'] = 1.0
                elif 'length' in weight_scheme:
                    data['weight'] = len(data['motion']['trans'])
                elif 'text' in weight_scheme:
                    if data['data_source'] == 'samp':  # ignore samp in text weight scheme
                        data['weight'] = 0
                        continue

                    seq_weight = 0
                    for seg in data['frame_labels']:
                        # print('act_cat:', seg['act_cat'])
                        act_weights = sum([action_statistics[act_cat]['weight'] for act_cat in seg['act_cat']])  # sum of unit weights of all action categories
                        seq_weight += (seg['end_t'] - seg['start_t']) * act_weights
                    data['weight'] = seq_weight
                    # print('calc frame segment weights:', data['seq_name'])
                    num_frames = len(data['motion']['trans'])
                    frame_weights = []  # [num_frames - self.seq_length + 1]
                    for frame_idx in range(0, num_frames - self.seq_length + 1):
                        start_t = frame_idx / self.target_fps
                        end_t = (frame_idx + self.seq_length - 1) / self.target_fps
                        frame_weight = 0  # at least weight one even if no text
                        for seg in data['frame_labels']:
                            overlap_len = get_overlap([seg['start_t'], seg['end_t']], [start_t, end_t])
                            if overlap_len > 0:
                                act_weights = sum([action_statistics[act_cat]['weight'] for act_cat in
                                                   seg['act_cat']])  # sum of unit weights of all action categories
                                frame_weight += overlap_len * act_weights
                        frame_weights.append(frame_weight)
                        # print(f'start frame{frame_idx} weight: {weight}')
                    data['frame_weights'] = frame_weights
            print('finish first assigning seq weights')

            # make the sum of weights of seqs from babel and samp to be 0.5 respectively
            if 'samp' in weight_scheme:
                babel_sum = sum([data['weight'] for data in dataset if data['data_source'] == 'babel'])
                print('babel sum: ', babel_sum)
                samp_sum = sum([data['weight'] for data in dataset if data['data_source'] == 'samp'])
                print('samp sum: ', samp_sum)
                samp_percent = float(weight_scheme.split('samp:')[-1].split('_')[0])
                print('samp percent: ', samp_percent)
                if babel_sum > 0 and samp_sum > 0:
                    for data in dataset:
                        if data['data_source'] == 'babel':
                            data['weight'] = data['weight'] / babel_sum * (1 - samp_percent)
                        elif data['data_source'] == 'samp':
                            data['weight'] = data['weight'] / samp_sum * samp_percent
                if 'lie' in weight_scheme and 'sit' in weight_scheme and 'loco' in weight_scheme:
                    lie_percent = float(weight_scheme.split('lie:')[-1].split('_')[0])
                    sit_percent = float(weight_scheme.split('sit:')[-1].split('_')[0])
                    loco_percent = float(weight_scheme.split('loco:')[-1].split('_')[0])
                    print('lie percent: ', lie_percent)
                    print('sit percent: ', sit_percent)
                    print('loco percent: ', loco_percent)
                    samp_data = [data for data in dataset if data['data_source'] == 'samp']
                    lie_data = []
                    sit_data = []
                    loco_data = []
                    for data in samp_data:
                        if 'lie' in data['seq_name']:
                            lie_data.append(data)
                        elif 'locomotion' in data['seq_name'] or 'run' in data['seq_name']:
                            loco_data.append(data)
                        else:
                            sit_data.append(data)
                    lie_sum = sum([data['weight'] for data in lie_data])
                    sit_sum = sum([data['weight'] for data in sit_data])
                    loco_sum = sum([data['weight'] for data in loco_data])
                    print('lie sum: ', lie_sum)
                    print('sit sum: ', sit_sum)
                    print('loco sum: ', loco_sum)
                    for data in lie_data:
                        data['weight'] = data['weight'] / lie_sum * lie_percent
                    for data in sit_data:
                        data['weight'] = data['weight'] / sit_sum * sit_percent
                    for data in loco_data:
                        data['weight'] = data['weight'] / loco_sum * loco_percent
                elif 'lie' in weight_scheme:
                    lie_percent = float(weight_scheme.split('lie:')[-1].split('_')[0])
                    print('lie percent: ', lie_percent)
                    lie_sum = 0
                    other_sum = 0
                    for data in dataset:
                        if data['data_source'] == 'samp' and 'lie' in data['seq_name']:
                            lie_sum += data['weight']
                        else:
                            other_sum += data['weight']
                    assert lie_sum > 0
                    assert other_sum > 0
                    for data in dataset:
                        if data['data_source'] == 'samp' and 'lie' in data['seq_name']:
                            data['weight'] = data['weight'] / lie_sum * lie_percent
                        else:
                            data['weight'] = data['weight'] / other_sum * (1 - lie_percent)


            if 'category' in weight_scheme:
                weight_categories = [
                    # 'walk',
                    # 'lie',
                    # 'sit',
                    'move up/down incline'
                ]
                exclude_categories = ['lie in prone position']
                percent = float(weight_scheme.split('category:')[-1].split('_')[0])
                print('categories: ', weight_categories)
                print('percent: ', percent)
                sum_incategory = 0
                sum_not_incategory = 0
                for data in dataset:
                    act_cat = []
                    if 'frame_labels' in data:
                        for seg in data['frame_labels']:
                            act_cat.extend(seg['act_cat'])
                    # if data['data_source'] == 'babel' and (set(act_cat) & {'lie'}):
                    #     data['category'] = 'exclude'
                    #     data['weight'] = 0.0
                    #     continue
                    if set(act_cat) & set(weight_categories):
                        data['category'] = 'weighted'
                        if data['weight'] == 0:  # only for samp:1_category:x
                            if 'uniform' in weight_scheme:
                                data['weight'] = 1.0
                            elif 'length' in weight_scheme:
                                data['weight'] = len(data['motion']['trans'])
                        sum_incategory += data['weight']
                        print('weighted: ', data['seq_name'])
                    elif set(act_cat) & set(exclude_categories):
                        data['category'] = 'exclude'
                        data['weight'] = 0.0
                        print('exclude: ', data['seq_name'])
                    else:
                        data['category'] = 'not_weighted'
                        sum_not_incategory += data['weight']
                assert sum_incategory > 0
                assert sum_not_incategory > 0
                for data in dataset:
                    if data['category'] == 'weighted':
                        data['weight'] = data['weight'] / sum_incategory * percent
                    elif data['category'] == 'not_weighted':
                        data['weight'] = data['weight'] / sum_not_incategory * (1 - percent)

            # overfit using one sequence
            if 'overfit' in weight_scheme:
                seq_id = int(weight_scheme.split('overfit:')[-1].split('_')[0])
                for idx, data in enumerate(dataset):
                    if idx == seq_id:
                        data['weight'] = 1.0
                    else:
                        data['weight'] = 0.0
            seq_weights = np.array([data['weight'] for data in dataset])
            seq_weights = seq_weights / seq_weights.sum()

            self.dataset = dataset
            self.seq_weights = seq_weights

        # load or calc mean and std
        self.tensor_mean_device_dict = {}
        file_name = f'mean_std_h{self.history_length}_f{self.future_length}'
        # TODO: use different mean and std when enforce gender and beta
        # if self.enforce_gender is not None:
        #     file_name = file_name + f'_{self.enforce_gender}'
        # if self.enforce_zero_beta:
        #     file_name = file_name + '_zero_beta'
        mean_std_path = Path(dataset_path, f'{file_name}.pkl')
        if mean_std_path.exists():
            print(f'loading mean and std from {mean_std_path}')
            with open(mean_std_path, 'rb') as f:
                self.tensor_mean, self.tensor_std = pickle.load(f)  # [1, 1, D]
        else:
            assert self.split == 'train'
            print('calculating mean and std using train split')
            self.tensor_mean, self.tensor_std = self.calc_mean_std()
            with open(mean_std_path, 'wb') as f:
                pickle.dump((self.tensor_mean.detach().cpu(), self.tensor_std.detach().cpu()), f)

        # load clip model, get train text embeddings
        self.clip_model = load_and_freeze_clip(clip_version='ViT-B/32', device=self.device)
        self.embedding_path = embedding_path = Path(dataset_path, f'{split}_text_embedding_dict.pkl')
        if embedding_path.exists():
            print(f'loading text embeddings from {embedding_path}')
            with open(embedding_path, 'rb') as f:
                self.text_embedding_dict = pickle.load(f)
        else:
            print('calculating text embeddings')
            raw_texts = []
            for data in self.dataset:
                if 'frame_labels' in data:
                    raw_texts.extend([seg['proc_label'] for seg in data['frame_labels']])
            raw_texts = list(set(raw_texts))
            num_texts = len(raw_texts)
            print('num of unique texts: ', len(raw_texts))
            # get text embeddings by batch
            text_embeddings = []
            batch_start_idx = 0
            while batch_start_idx < num_texts:
                batch_end_idx = min(batch_start_idx + 256, num_texts)
                text_embeddings.append(encode_text(self.clip_model, raw_texts[batch_start_idx:batch_end_idx]))
                batch_start_idx = batch_end_idx
            text_embeddings = torch.cat(text_embeddings, dim=0).detach().cpu().numpy()
            print(text_embeddings.shape)
            self.text_embedding_dict = {raw_texts[idx]: text_embeddings[idx] for idx in range(num_texts)}
            self.text_embedding_dict[''] = np.zeros(512).astype(np.float32)  # for empty text have zero embedding, compatible with mdm text masking
            with open(embedding_path, 'wb') as f:
                pickle.dump(self.text_embedding_dict, f)
        for key in self.text_embedding_dict:
            self.text_embedding_dict[key] = torch.from_numpy(self.text_embedding_dict[key]).to(dtype=torch.float32, device=self.device)

    def update_text_embedding_dict(self, new_texts):
        new_text_embeddings = encode_text(self.clip_model, new_texts)
        for idx, text in enumerate(new_texts):
            self.text_embedding_dict[text] = new_text_embeddings[idx]

    def export_text_embedding_dict(self):
        text_embedding_dict = {key: self.text_embedding_dict[key].detach().cpu().numpy() for key in self.text_embedding_dict}
        with open(self.embedding_path, 'wb') as f:
            pickle.dump(text_embedding_dict, f)

    def calc_mean_std(self, batch_size=512):
        all_mp_data = []
        for seq_data in self.dataset:
            motion_data = seq_data['motion']
            num_frames = motion_data['trans'].shape[0]
            primitive_data_list = []
            for start_frame in range(0, num_frames - self.primitive_length, self.future_length):
                end_frame = start_frame + self.primitive_length
                primitive_data_list.append(self.get_primitive(seq_data, start_frame, end_frame, skip_text=True))

            primitive_dict = {'gender': primitive_data_list[0]['primitive_dict']['gender']}
            for key in ['betas', 'transl', 'global_orient', 'body_pose', 'transf_rotmat', 'transf_transl']:
                primitive_dict[key] = torch.cat([data['primitive_dict'][key] for data in primitive_data_list], dim=0)
            primitive_dict = tensor_dict_to_device(primitive_dict, self.device)

            # split primitive_dict into batches
            batch_start_idx = 0
            while batch_start_idx < len(primitive_dict['transl']):
                batch_end_idx = min(batch_start_idx + batch_size, len(primitive_dict['transl']))
                batch_primitive_dict = {key: primitive_dict[key][batch_start_idx:batch_end_idx] for key in ['betas', 'transl', 'global_orient', 'body_pose', 'transf_rotmat', 'transf_transl']}
                batch_primitive_dict['gender'] = primitive_dict['gender']
                _, _, canonicalized_primitive_dict = self.primitive_utility.canonicalize(batch_primitive_dict)
                feature_dict = self.primitive_utility.calc_features(canonicalized_primitive_dict)
                feature_dict['transl'] = feature_dict['transl'][:, :-1, :]  # [num_primitive, T, 3]
                feature_dict['poses_6d'] = feature_dict['poses_6d'][:, :-1, :]  # [num_primitive, T, 66]
                feature_dict['joints'] = feature_dict['joints'][:, :-1, :]  # [num_primitive, T, 22 * 3]
                motion_tensor = self.dict_to_tensor(feature_dict)  # [num_primitive, T, D]
                all_mp_data.append(motion_tensor)

                batch_start_idx = batch_end_idx

        all_mp_data = torch.cat(all_mp_data, dim=0)  # [N, T, D]
        tensor_mean = all_mp_data.mean(dim=[0, 1], keepdim=True)  # [1, 1, D]
        tensor_std = all_mp_data.std(dim=[0, 1], keepdim=True)  # [1, 1, D]
        return tensor_mean, tensor_std

    def dict_to_tensor(self, data_dict):
        return self.primitive_utility.dict_to_tensor(data_dict)

    def tensor_to_dict(self, tensor):
        return self.primitive_utility.tensor_to_dict(tensor)

    def get_mean_std_by_device(self, device):
        if device not in self.tensor_mean_device_dict:
            self.tensor_mean_device_dict[device] = (self.tensor_mean.to(device=device), self.tensor_std.to(device=device))
        return self.tensor_mean_device_dict[device]

    def normalize(self, tensor):
        tensor_mean, tensor_std = self.get_mean_std_by_device(tensor.device)
        return (tensor - tensor_mean) / tensor_std  # [B, T, D]

    def denormalize(self, tensor):
        tensor_mean, tensor_std = self.get_mean_std_by_device(tensor.device)
        return tensor * tensor_std + tensor_mean  # [B, T, D]

    def get_primitive(self, seq_data, start_frame, end_frame, skip_text=False):
        """end_frame included"""
        motion_data = seq_data['motion']
        primitive_dict = {
            'gender': motion_data['gender'],
            'betas': motion_data['betas'].expand(1, self.primitive_length + 1, 10),
            'transl': motion_data['trans'][start_frame:end_frame + 1].unsqueeze(0),  # include one more frame for delta feature calculation
            'global_orient': transforms.axis_angle_to_matrix(motion_data['poses'][start_frame:end_frame + 1, :3].unsqueeze(0)),
            'body_pose': transforms.axis_angle_to_matrix(
                motion_data['poses'][start_frame:end_frame + 1, 3:66].unsqueeze(0).reshape(1, end_frame - start_frame + 1, 21, 3)
            ),
            'transf_rotmat': torch.eye(3).unsqueeze(0),
            'transf_transl': torch.zeros(1, 1, 3),
        }
        # print(primitive_dict['gender'], primitive_dict['betas'])
        # print('load data time: ', time.time() - self.time)
        # primitive_dict = tensor_dict_to_device(primitive_dict, self.device)
        # print('data to device time: ', time.time() - self.time)

        texts = []
        if not skip_text and 'frame_labels' in seq_data:
            future_start = (start_frame + self.history_length) / self.target_fps
            future_end = (start_frame + self.history_length + self.future_length - 1) / self.target_fps
            # print('text tolerance: ', self.text_tolerance)
            for seg in seq_data['frame_labels']:
                if have_overlap([seg['start_t'], seg['end_t']], [future_start - self.text_tolerance, future_end + self.text_tolerance]):
                    texts.append(seg['proc_label'])
        # print('text label time: ', time.time() - self.time)

        output = {
            'text': random.choice(texts) if len(texts) > 0 else '',
            # 'text': compose_texts_with_and(texts) if len(texts) > 0 else '',
            'primitive_dict': primitive_dict,
        }
        return output

    def get_batch_idx(self, batch_size=8):
        batch_idx = np.random.choice(len(self.dataset), size=batch_size, replace=True, p=self.seq_weights)
        return batch_idx

    def get_batch(self, batch_size=8):
        self.time = time.time()
        seq_list = []
        batch_idx = self.get_batch_idx(batch_size)
        # print('#batch_idx: ', len(batch_idx))

        # pool = mp.Pool(2)  # Create a process pool
        # seq_list = pool.starmap(get_subseq,
        #                         [(self.dataset[seq_idx], self.history_length, self.future_length, self.primitive_length, self.seq_length, self.target_fps, False) for seq_idx in batch_idx]
        #                         )  # Map the process_sequence function over batch_idx
        # pool.close()
        # pool.join()
        # print('num of sequences: ', len(seq_list))
        # print('num of mp:', len(seq_list[0]))

        for seq_idx in batch_idx:
            seq_data = self.dataset[seq_idx]
            num_frames = len(seq_data['motion']['trans'])
            if self.prob_static > 0 and random.random() < self.prob_static:
                static_frame = random.randint(0, num_frames - 1) # right end inclusive
                motion_data = seq_data['motion']
                primitive_length = self.primitive_length
                primitive_dict = {
                    'gender': motion_data['gender'],
                    'betas': motion_data['betas'].expand(1, primitive_length + 1, 10),
                    'transl': motion_data['trans'][[static_frame]].expand(primitive_length + 1, -1).unsqueeze(0),
                    # include one more frame for delta feature calculation
                    'global_orient': transforms.axis_angle_to_matrix(
                        motion_data['poses'][[static_frame], :3].expand(primitive_length + 1, -1).unsqueeze(0)),
                    'body_pose': transforms.axis_angle_to_matrix(
                        motion_data['poses'][[static_frame], 3:66].expand(primitive_length + 1, -1).unsqueeze(
                            0).reshape(1, primitive_length + 1, 21, 3)
                    ),
                    'transf_rotmat': torch.eye(3).unsqueeze(0),
                    'transf_transl': torch.zeros(1, 1, 3),
                }
                primitive_data = {
                    'text': '',
                    'primitive_dict': primitive_dict
                }
                primitive_data_list = [primitive_data] * self.num_primitive
                # print('get static sequenece')
            else:
                if 'text' in self.weight_scheme:
                    start_frame = random.choices(range(num_frames - self.seq_length + 1), weights=seq_data['frame_weights'], k=1)[0]
                else:
                    start_frame = random.randint(0, num_frames - self.seq_length)  # [0, num_frames - seq_length], right end inclusive
                primitive_data_list = []
                for frame_idx in range(start_frame, start_frame + self.seq_length - self.primitive_length, self.future_length):
                    # print('frame_idx: ', frame_idx, 'num_frames: ', num_frames, 'future_length: ', self.future_length)
                    primitive_data = self.get_primitive(seq_data, frame_idx, frame_idx + self.primitive_length)
                    primitive_data_list.append(primitive_data)
            seq_list.append(primitive_data_list)
        # print('get primitive time: ', time.time() - self.time)

        # sort batch by gender
        batch = None
        for gender in ['female', 'male']:
            gender_idx = [idx for idx in range(len(seq_list)) if seq_list[idx][0]['primitive_dict']['gender'] == gender]
            if len(gender_idx) == 0:
                continue
            gender_seq_list = [seq_list[i] for i in gender_idx]
            gender_batch_size = len(gender_idx)
            gender_batch = []

            gender_seq_texts = None
            gender_seq_dict = None
            for primitive_idx in range(self.num_primitive):
                primitive_texts = [mp_seq[primitive_idx]['text'] for mp_seq in gender_seq_list]
                primitive_dict = {'gender': gender}
                for key in ['betas', 'transl', 'global_orient', 'body_pose', 'transf_rotmat', 'transf_transl']:
                    primitive_dict[key] = torch.cat([mp_seq[primitive_idx]['primitive_dict'][key] for mp_seq in gender_seq_list], dim=0)
                gender_seq_texts = primitive_texts if gender_seq_texts is None else gender_seq_texts + primitive_texts
                if gender_seq_dict is None:
                    gender_seq_dict = primitive_dict
                else:
                    for key in ['betas', 'transl', 'global_orient', 'body_pose', 'transf_rotmat', 'transf_transl']:
                        gender_seq_dict[key] = torch.cat([gender_seq_dict[key], primitive_dict[key]], dim=0)

            gender_seq_dict = tensor_dict_to_device(gender_seq_dict, self.device)
            _, _, canonicalized_primitive_dict = self.primitive_utility.canonicalize(gender_seq_dict)
            transf_rotmat, transf_transl = canonicalized_primitive_dict['transf_rotmat'], canonicalized_primitive_dict['transf_transl']
            # print(f'{gender}:canonicalize time: ', time.time() - self.time)
            feature_dict = self.primitive_utility.calc_features(canonicalized_primitive_dict)
            # print(f'{gender}:calc feature time: ', time.time() - self.time)
            feature_dict['transl'] = feature_dict['transl'][:, :-1, :]  # [B*num_mp, T, 3]
            feature_dict['poses_6d'] = feature_dict['poses_6d'][:, :-1, :]  # [B*num_mp, T, 66]
            feature_dict['joints'] = feature_dict['joints'][:, :-1, :]  # [B*num_mp, T, 22 * 3]
            motion_tensor_normalized = self.normalize(self.dict_to_tensor(feature_dict))  # [B*num_mp, T, D]
            motion_tensor_normalized = motion_tensor_normalized.permute(0, 2, 1).unsqueeze(2)  # [B*num_mp, D, 1, T]
            history_mask = torch.zeros_like(motion_tensor_normalized, dtype=torch.bool, device=self.device)
            history_mask[..., :self.cfg.history_length] = True
            history_motion = torch.zeros_like(motion_tensor_normalized, dtype=torch.float32, device=self.device)
            history_motion[..., :self.cfg.history_length] = motion_tensor_normalized[..., :self.cfg.history_length]

            for primitive_idx in range(self.num_primitive):
                start_idx = primitive_idx * gender_batch_size
                end_idx = (primitive_idx + 1) * gender_batch_size
                primitive_texts = gender_seq_texts[start_idx:end_idx]
                unseen_texts = [text for text in primitive_texts if text not in self.text_embedding_dict]
                if len(unseen_texts) > 0:
                    self.update_text_embedding_dict(unseen_texts)
                text_embedding = torch.stack([self.text_embedding_dict[text] for text in primitive_texts], dim=0)  # [B, 512]
                gender_batch.append(
                    {
                        'texts': primitive_texts,
                        'text_embedding': text_embedding,
                        'gender': [gender_seq_dict['gender']] * gender_batch_size,
                        'betas': gender_seq_dict['betas'][start_idx:end_idx, :-1, :10],
                        'motion_tensor_normalized': motion_tensor_normalized[start_idx:end_idx, ...], # [B, D, 1, T]
                        'history_motion': history_motion[start_idx:end_idx, ...],
                        'history_mask': history_mask[start_idx:end_idx, ...],
                        'history_length': self.cfg.history_length,
                        'future_length': self.cfg.future_length,
                        'transf_rotmat': transf_rotmat,
                        'transf_transl': transf_transl,
                    }
                )

            # for primitive_idx in range(self.num_primitive):
            #     primitive_texts = [mp_seq[primitive_idx]['text'] for mp_seq in gender_seq_list]
            #     primitive_dict = {'gender': gender}
            #     for key in ['betas', 'transl', 'global_orient', 'body_pose', 'transf_rotmat', 'transf_transl']:
            #         primitive_dict[key] = torch.cat([mp_seq[primitive_idx]['primitive_dict'][key] for mp_seq in gender_seq_list], dim=0)
            #
            #     primitive_dict = tensor_dict_to_device(primitive_dict, self.device)
            #     _, _, canonicalized_primitive_dict = self.primitive_utility.canonicalize(primitive_dict)
            #     print(f'{gender}:canonicalize time: ', time.time() - self.time)
            #     # transf_rotmat, transf_transl = canonicalized_primitive_dict['transf_rotmat'], canonicalized_primitive_dict[
            #     #     'transf_transl']
            #     feature_dict = self.primitive_utility.calc_features(canonicalized_primitive_dict)
            #     print(f'{gender}:calc feature time: ', time.time() - self.time)
            #     feature_dict['transl'] = feature_dict['transl'][:, :-1, :]  # [B, T, 3]
            #     feature_dict['poses_6d'] = feature_dict['poses_6d'][:, :-1, :]  # [B, T, 66]
            #     feature_dict['joints'] = feature_dict['joints'][:, :-1, :]  # [B, T, 22 * 3]
            #     motion_tensor_normalized = self.normalize(self.dict_to_tensor(feature_dict))  # [B, T, D]
            #     motion_tensor_normalized = motion_tensor_normalized.permute(0, 2, 1).unsqueeze(2)  # [B, D, 1, T]
            #     history_mask = torch.zeros_like(motion_tensor_normalized, dtype=torch.bool, device=self.device)
            #     history_mask[..., :self.cfg.history_length] = True
            #     history_motion = torch.zeros_like(motion_tensor_normalized, dtype=torch.float32, device=self.device)
            #     history_motion[..., :self.cfg.history_length] = motion_tensor_normalized[..., :self.cfg.history_length]
            #     text_embedding = torch.stack([self.text_embedding_dict[text] for text in primitive_texts],
            #                                  dim=0)  # [B, 512]
            #
            #     gender_batch.append(
            #         {
            #             'texts': primitive_texts,
            #             'text_embedding': text_embedding,
            #             'gender': [primitive_dict['gender']] * len(gender_seq_list),
            #             'betas': primitive_dict['betas'][:, :-1, :10],
            #             'motion_tensor_normalized': motion_tensor_normalized, # [B, D, 1, T]
            #             'history_motion': history_motion,
            #             'history_mask': history_mask,
            #             'history_length': self.cfg.history_length,
            #             'future_length': self.cfg.future_length,
            #         }
            #     )
            #     print(f'{gender}:primitive {primitive_idx} batch time: ', time.time() - self.time)

            if batch is None:
                batch = gender_batch
            else:  # concatenate different gender batch
                for primitive_idx in range(self.num_primitive):
                    for key in ['texts', 'gender']:
                        batch[primitive_idx][key] = batch[primitive_idx][key] + gender_batch[primitive_idx][key]
                    for key in ['betas', 'motion_tensor_normalized', 'history_motion', 'history_mask', 'text_embedding', 'transf_rotmat', 'transf_transl']:
                        batch[primitive_idx][key] = torch.cat([batch[primitive_idx][key], gender_batch[primitive_idx][key]], dim=0)
            # print(f'{gender} batch time: ', time.time() - self.time)

        return batch

    def __len__(self):
        return len(self.dataset)

class WeightedPrimitiveSequenceDatasetV2(WeightedPrimitiveSequenceDataset):
    def __init__(self, dataset_name='weighted_mp_seq',
                 dataset_path='./data/seq_data',
                 cfg_path='./config_files/config_hydra/motion_primitive/mp_2_8.yaml',
                 split="train",
                 device='cuda',
                 weight_scheme='uniform',
                 prob_static=0.0,
                 enforce_gender=None,
                 enforce_zero_beta=None,
                 load_data=True,
                 text_tolerance=0.0,
                 use_frame_weights=True,
                 body_type='smplx',
                 **kwargs):
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.split = split
        self.device = device
        self.weight_scheme = weight_scheme
        self.prob_static = prob_static
        self.enforce_gender = enforce_gender
        self.enforce_zero_beta = enforce_zero_beta
        self.text_tolerance = text_tolerance
        print('enforce_gender: ', enforce_gender)
        print('enforce_zero_beta: ', enforce_zero_beta)

        self.primitive_utility = PrimitiveUtility(device=self.device, body_type=body_type)
        self.motion_repr = self.primitive_utility.motion_repr

        # cfg_path = Path(dataset_path, 'config.yaml')
        with open(cfg_path, 'r') as f:
            self.cfg = OmegaConf.load(f)
        self.target_fps = self.cfg.fps
        # self.downsample_rate = 120 // self.target_fps
        self.history_length = self.cfg.history_length
        self.future_length = self.cfg.future_length
        self.primitive_length = self.history_length + self.future_length
        self.num_primitive = self.cfg.num_primitive
        self.seq_length = self.history_length + self.future_length * self.num_primitive + 1

        if load_data:
            with open(pjoin(dataset_path, f'{split}.pkl'), 'rb') as f:
                dataset = pickle.load(f)
            dataset = [data for data in dataset if len(data['motion']['trans']) >= self.seq_length]
            for data in dataset:
                gender = self.enforce_gender if self.enforce_gender is not None else data['motion']['gender']
                betas =torch.from_numpy(data['motion']['betas'].astype(np.float32))
                if self.enforce_zero_beta:
                    betas = torch.zeros_like(betas)
                transl = torch.from_numpy(data['motion']['trans'].astype(np.float32))
                poses = torch.from_numpy(data['motion']['poses'].astype(np.float32))
                global_orient = transforms.axis_angle_to_matrix(poses[:, :3])  # [T, 3, 3]
                body_pose = transforms.axis_angle_to_matrix(poses[:, 3:66].reshape(-1, 21, 3))  # [T, 21, 3, 3]
                pelvis_delta = torch.from_numpy(data['motion']['pelvis_delta'].astype(np.float32))  # [3]
                joints = torch.from_numpy(data['motion']['joints'].astype(np.float32))  # [T, 22, 3]
                data['motion'] = {
                    'gender': gender,
                    'betas': betas,
                    'transl': transl,
                    'global_orient': global_orient,
                    'body_pose': body_pose,
                    'pelvis_delta': pelvis_delta,
                    'joints': joints,
                }
            print('num of sequences: ', len(dataset))
            # assign sampling weights to each sequence

            with open('./data/action_statistics.json', 'r') as f:
                action_statistics = json.load(f)

            for data in dataset:
                # if data['seq_name'].find('20160930_50032') >= 0 or data['seq_name'].find('20161014_50033') >= 0:
                #     data['weight'] = 0.0
                #     print('error seq:', data['seq_name'])  #  discard these sequences or scale the segment time labels?
                # elif
                if 'uniform' in weight_scheme:
                    data['weight'] = 1.0
                elif 'length' in weight_scheme:
                    data['weight'] = len(data['motion']['trans'])
                elif 'text' in weight_scheme:
                    if data['data_source'] == 'samp':  # ignore samp in text weight scheme
                        data['weight'] = 0
                        continue

                    seq_weight = 0
                    for seg in data['frame_labels']:
                        # print('act_cat:', seg['act_cat'])
                        # if int(seg['end_t'] * self.target_fps) > len(data['motion']['transl']) + 1:
                        #     print('error seq:', data['seq_name'], int(seg['end_t'] * self.target_fps), len(data['motion']['transl']))
                        #     error_seq = 1
                        #     break
                        act_weights = sum([action_statistics[act_cat]['weight'] for act_cat in seg['act_cat']])  # sum of unit weights of all action categories
                        seq_weight += (seg['end_t'] - seg['start_t']) * act_weights
                    data['weight'] = seq_weight
                    # print('calc frame segment weights:', data['seq_name'])
                    num_frames = len(data['motion']['transl'])
                    if use_frame_weights:
                        frame_weights = []  # [num_frames - self.seq_length + 1]
                        for frame_idx in range(0, num_frames - self.seq_length + 1):
                            start_t = frame_idx / self.target_fps
                            end_t = (frame_idx + self.seq_length - 1) / self.target_fps
                            frame_weight = 0  # at least weight one even if no text
                            for seg in data['frame_labels']:
                                overlap_len = get_overlap([seg['start_t'], seg['end_t']], [start_t, end_t])
                                if overlap_len > 0:
                                    act_weights = sum([action_statistics[act_cat]['weight'] for act_cat in
                                                       seg['act_cat']])  # sum of unit weights of all action categories
                                    frame_weight += overlap_len * act_weights
                            frame_weights.append(frame_weight)
                            # print(f'start frame{frame_idx} weight: {weight}')
                        data['frame_weights'] = frame_weights
            print('finish first assigning seq weights')

            # make the sum of weights of seqs from babel and samp to be 0.5 respectively
            if 'samp' in weight_scheme:
                babel_sum = sum([data['weight'] for data in dataset if data['data_source'] == 'babel'])
                print('babel sum: ', babel_sum)
                samp_sum = sum([data['weight'] for data in dataset if data['data_source'] == 'samp'])
                print('samp sum: ', samp_sum)
                samp_percent = float(weight_scheme.split('samp:')[-1].split('_')[0])
                print('samp percent: ', samp_percent)
                if babel_sum > 0 and samp_sum > 0:
                    for data in dataset:
                        if data['data_source'] == 'babel':
                            data['weight'] = data['weight'] / babel_sum * (1 - samp_percent)
                        elif data['data_source'] == 'samp':
                            data['weight'] = data['weight'] / samp_sum * samp_percent
                if 'lie' in weight_scheme and 'sit' in weight_scheme and 'loco' in weight_scheme:
                    lie_percent = float(weight_scheme.split('lie:')[-1].split('_')[0])
                    sit_percent = float(weight_scheme.split('sit:')[-1].split('_')[0])
                    loco_percent = float(weight_scheme.split('loco:')[-1].split('_')[0])
                    print('lie percent: ', lie_percent)
                    print('sit percent: ', sit_percent)
                    print('loco percent: ', loco_percent)
                    samp_data = [data for data in dataset if data['data_source'] == 'samp']
                    lie_data = []
                    sit_data = []
                    loco_data = []
                    for data in samp_data:
                        if 'lie' in data['seq_name']:
                            lie_data.append(data)
                        elif 'locomotion' in data['seq_name'] or 'run' in data['seq_name']:
                            loco_data.append(data)
                        else:
                            sit_data.append(data)
                    lie_sum = sum([data['weight'] for data in lie_data])
                    sit_sum = sum([data['weight'] for data in sit_data])
                    loco_sum = sum([data['weight'] for data in loco_data])
                    print('lie sum: ', lie_sum)
                    print('sit sum: ', sit_sum)
                    print('loco sum: ', loco_sum)
                    for data in lie_data:
                        data['weight'] = data['weight'] / lie_sum * lie_percent
                    for data in sit_data:
                        data['weight'] = data['weight'] / sit_sum * sit_percent
                    for data in loco_data:
                        data['weight'] = data['weight'] / loco_sum * loco_percent
                elif 'lie' in weight_scheme:
                    lie_percent = float(weight_scheme.split('lie:')[-1].split('_')[0])
                    print('lie percent: ', lie_percent)
                    lie_sum = 0
                    other_sum = 0
                    for data in dataset:
                        if data['data_source'] == 'samp' and 'lie' in data['seq_name']:
                            lie_sum += data['weight']
                        else:
                            other_sum += data['weight']
                    assert lie_sum > 0
                    assert other_sum > 0
                    for data in dataset:
                        if data['data_source'] == 'samp' and 'lie' in data['seq_name']:
                            data['weight'] = data['weight'] / lie_sum * lie_percent
                        else:
                            data['weight'] = data['weight'] / other_sum * (1 - lie_percent)


            if 'category' in weight_scheme:
                weight_categories = [
                    # 'walk',
                    # 'lie',
                    # 'sit',
                    'move up/down incline'
                ]
                exclude_categories = ['lie in prone position']
                percent = float(weight_scheme.split('category:')[-1].split('_')[0])
                print('categories: ', weight_categories)
                print('percent: ', percent)
                sum_incategory = 0
                sum_not_incategory = 0
                for data in dataset:
                    act_cat = []
                    if 'frame_labels' in data:
                        for seg in data['frame_labels']:
                            act_cat.extend(seg['act_cat'])
                    # if data['data_source'] == 'babel' and (set(act_cat) & {'lie'}):
                    #     data['category'] = 'exclude'
                    #     data['weight'] = 0.0
                    #     continue
                    if set(act_cat) & set(weight_categories):
                        data['category'] = 'weighted'
                        if data['weight'] == 0:  # only for samp:1_category:x
                            if 'uniform' in weight_scheme:
                                data['weight'] = 1.0
                            elif 'length' in weight_scheme:
                                data['weight'] = len(data['motion']['trans'])
                        sum_incategory += data['weight']
                        print('weighted: ', data['seq_name'])
                    elif set(act_cat) & set(exclude_categories):
                        data['category'] = 'exclude'
                        data['weight'] = 0.0
                        print('exclude: ', data['seq_name'])
                    else:
                        data['category'] = 'not_weighted'
                        sum_not_incategory += data['weight']
                assert sum_incategory > 0
                assert sum_not_incategory > 0
                for data in dataset:
                    if data['category'] == 'weighted':
                        data['weight'] = data['weight'] / sum_incategory * percent
                    elif data['category'] == 'not_weighted':
                        data['weight'] = data['weight'] / sum_not_incategory * (1 - percent)

            # overfit using one sequence
            if 'overfit' in weight_scheme:
                seq_id = int(weight_scheme.split('overfit:')[-1].split('_')[0])
                for idx, data in enumerate(dataset):
                    if idx == seq_id:
                        data['weight'] = 1.0
                    else:
                        data['weight'] = 0.0
            seq_weights = np.array([data['weight'] for data in dataset])
            seq_weights = seq_weights / seq_weights.sum()

            self.dataset = dataset
            self.seq_weights = seq_weights

        # load or calc mean and std
        self.tensor_mean_device_dict = {}
        file_name = f'mean_std_h{self.history_length}_f{self.future_length}'
        # TODO: use different mean and std when enforce gender and beta
        # if self.enforce_gender is not None:
        #     file_name = file_name + f'_{self.enforce_gender}'
        # if self.enforce_zero_beta:
        #     file_name = file_name + '_zero_beta'
        mean_std_path = Path(dataset_path, f'{file_name}.pkl')
        if mean_std_path.exists():
            print(f'loading mean and std from {mean_std_path}')
            with open(mean_std_path, 'rb') as f:
                self.tensor_mean, self.tensor_std = pickle.load(f)  # [1, 1, D]
        else:
            assert self.split == 'train'
            print('calculating mean and std using train split')
            self.tensor_mean, self.tensor_std = self.calc_mean_std()
            with open(mean_std_path, 'wb') as f:
                pickle.dump((self.tensor_mean.detach().cpu(), self.tensor_std.detach().cpu()), f)

        # load clip model, get train text embeddings
        self.clip_model = load_and_freeze_clip(clip_version='ViT-B/32', device=self.device)
        self.embedding_path = embedding_path = Path(dataset_path, f'{split}_text_embedding_dict.pkl')
        if embedding_path.exists():
            print(f'loading text embeddings from {embedding_path}')
            with open(embedding_path, 'rb') as f:
                self.text_embedding_dict = pickle.load(f)
        else:
            print('calculating text embeddings')
            raw_texts = []
            for data in self.dataset:
                if 'frame_labels' in data:
                    raw_texts.extend([seg['proc_label'] for seg in data['frame_labels']])
            raw_texts = list(set(raw_texts))
            num_texts = len(raw_texts)
            print('num of unique texts: ', len(raw_texts))
            # get text embeddings by batch
            text_embeddings = []
            batch_start_idx = 0
            while batch_start_idx < num_texts:
                batch_end_idx = min(batch_start_idx + 256, num_texts)
                text_embeddings.append(encode_text(self.clip_model, raw_texts[batch_start_idx:batch_end_idx]))
                batch_start_idx = batch_end_idx
            text_embeddings = torch.cat(text_embeddings, dim=0).detach().cpu().numpy()
            print(text_embeddings.shape)
            self.text_embedding_dict = {raw_texts[idx]: text_embeddings[idx] for idx in range(num_texts)}
            self.text_embedding_dict[''] = np.zeros(512).astype(np.float32)  # for empty text have zero embedding, compatible with mdm text masking
            with open(embedding_path, 'wb') as f:
                pickle.dump(self.text_embedding_dict, f)
        for key in self.text_embedding_dict:
            self.text_embedding_dict[key] = torch.from_numpy(self.text_embedding_dict[key]).to(dtype=torch.float32, device=self.device)

    def calc_mean_std(self, batch_size=512):
        all_mp_data = []
        for seq_data in self.dataset:
            motion_data = seq_data['motion']
            num_frames = motion_data['transl'].shape[0]
            primitive_data_list = []
            for start_frame in range(0, num_frames - self.primitive_length, self.future_length):
                end_frame = start_frame + self.primitive_length
                primitive_data_list.append(self.get_primitive(seq_data, start_frame, end_frame, skip_text=True))

            primitive_dict = {'gender': primitive_data_list[0]['primitive_dict']['gender']}
            for key in ['betas', 'transl', 'global_orient', 'body_pose', 'transf_rotmat', 'transf_transl', 'pelvis_delta', 'joints']:
                primitive_dict[key] = torch.cat([data['primitive_dict'][key] for data in primitive_data_list], dim=0)
            primitive_dict = tensor_dict_to_device(primitive_dict, self.device)

            # split primitive_dict into batches
            batch_start_idx = 0
            while batch_start_idx < len(primitive_dict['transl']):
                batch_end_idx = min(batch_start_idx + batch_size, len(primitive_dict['transl']))
                batch_primitive_dict = {key: primitive_dict[key][batch_start_idx:batch_end_idx] for key in ['betas', 'transl', 'global_orient', 'body_pose', 'transf_rotmat', 'transf_transl', 'pelvis_delta', 'joints']}
                batch_primitive_dict['gender'] = primitive_dict['gender']
                _, _, canonicalized_primitive_dict = self.primitive_utility.canonicalize(batch_primitive_dict, use_predicted_joints=True)
                feature_dict = self.primitive_utility.calc_features(canonicalized_primitive_dict, use_predicted_joints=True)
                feature_dict['transl'] = feature_dict['transl'][:, :-1, :]  # [num_primitive, T, 3]
                feature_dict['poses_6d'] = feature_dict['poses_6d'][:, :-1, :]  # [num_primitive, T, 66]
                feature_dict['joints'] = feature_dict['joints'][:, :-1, :]  # [num_primitive, T, 22 * 3]
                motion_tensor = self.dict_to_tensor(feature_dict)  # [num_primitive, T, D]
                all_mp_data.append(motion_tensor)

                batch_start_idx = batch_end_idx

        all_mp_data = torch.cat(all_mp_data, dim=0)  # [N, T, D]
        tensor_mean = all_mp_data.mean(dim=[0, 1], keepdim=True)  # [1, 1, D]
        tensor_std = all_mp_data.std(dim=[0, 1], keepdim=True)  # [1, 1, D]
        return tensor_mean, tensor_std

    def get_primitive(self, seq_data, start_frame, end_frame, skip_text=False):
        """end_frame included"""
        motion_data = seq_data['motion']
        primitive_dict = {
            'gender': motion_data['gender'],
            'betas': motion_data['betas'].expand(1, self.primitive_length + 1, 10),
            'transl': motion_data['transl'][start_frame:end_frame + 1].unsqueeze(0),  # include one more frame for delta feature calculation
            'global_orient': motion_data['global_orient'][start_frame:end_frame + 1].unsqueeze(0),
            'body_pose': motion_data['body_pose'][start_frame:end_frame + 1].unsqueeze(0),
            'pelvis_delta': motion_data['pelvis_delta'].unsqueeze(0),
            'joints': motion_data['joints'][start_frame:end_frame + 1].unsqueeze(0),
            'transf_rotmat': torch.eye(3).unsqueeze(0),
            'transf_transl': torch.zeros(1, 1, 3),
        }

        texts = []
        if not skip_text and 'frame_labels' in seq_data:
            future_start = (start_frame + self.history_length) / self.target_fps
            future_end = (start_frame + self.history_length + self.future_length - 1) / self.target_fps
            # print('text tolerance: ', self.text_tolerance)
            for seg in seq_data['frame_labels']:
                if have_overlap([seg['start_t'], seg['end_t']], [future_start - self.text_tolerance, future_end + self.text_tolerance]):
                    texts.append(seg['proc_label'])
        # print('text label time: ', time.time() - self.time)

        output = {
            'text': random.choice(texts) if len(texts) > 0 else '',
            # 'text': compose_texts_with_and(texts) if len(texts) > 0 else '',
            'primitive_dict': primitive_dict,
        }
        return output

    def get_batch(self, batch_size=8):
        self.time = time.time()
        seq_list = []
        batch_idx = self.get_batch_idx(batch_size)
        # print('#batch_idx: ', len(batch_idx))

        # pool = mp.Pool(2)  # Create a process pool
        # seq_list = pool.starmap(get_subseq,
        #                         [(self.dataset[seq_idx], self.history_length, self.future_length, self.primitive_length, self.seq_length, self.target_fps, False) for seq_idx in batch_idx]
        #                         )  # Map the process_sequence function over batch_idx
        # pool.close()
        # pool.join()
        # print('num of sequences: ', len(seq_list))
        # print('num of mp:', len(seq_list[0]))

        for seq_idx in batch_idx:
            seq_data = self.dataset[seq_idx]
            num_frames = len(seq_data['motion']['transl'])
            if self.prob_static > 0 and random.random() < self.prob_static:
                static_frame = random.randint(0, num_frames - 1) # right end inclusive
                motion_data = seq_data['motion']
                primitive_length = self.primitive_length
                primitive_dict = {
                    'gender': motion_data['gender'],
                    'betas': motion_data['betas'].expand(1, primitive_length + 1, 10),
                    'transl': motion_data['transl'][[static_frame]].expand(primitive_length + 1, -1).unsqueeze(0),
                    # include one more frame for delta feature calculation
                    'global_orient':
                        motion_data['global_orient'][[static_frame]].repeat(primitive_length + 1, 1, 1).unsqueeze(0),
                    'body_pose':
                        motion_data['body_pose'][[static_frame]].repeat(primitive_length + 1, 1, 1, 1).unsqueeze(0),
                    'pelvis_delta': motion_data['pelvis_delta'].unsqueeze(0),
                    'joints': motion_data['joints'][[static_frame]].repeat(primitive_length + 1, 1, 1).unsqueeze(0),
                    'transf_rotmat': torch.eye(3).unsqueeze(0),
                    'transf_transl': torch.zeros(1, 1, 3),
                }
                primitive_data = {
                    'text': '',
                    'primitive_dict': primitive_dict
                }
                primitive_data_list = [primitive_data] * self.num_primitive
                # print('get static sequenece')
            else:
                if 'text' in self.weight_scheme:
                    start_frame = random.choices(range(num_frames - self.seq_length + 1), weights=seq_data['frame_weights'], k=1)[0]
                else:
                    start_frame = random.randint(0, num_frames - self.seq_length)  # [0, num_frames - seq_length], right end inclusive
                primitive_data_list = []
                for frame_idx in range(start_frame, start_frame + self.seq_length - self.primitive_length, self.future_length):
                    primitive_data = self.get_primitive(seq_data, frame_idx, frame_idx + self.primitive_length)
                    primitive_data_list.append(primitive_data)
            seq_list.append(primitive_data_list)

        # sort batch by gender
        batch = None
        for gender in ['female', 'male']:
            gender_idx = [idx for idx in range(len(seq_list)) if seq_list[idx][0]['primitive_dict']['gender'] == gender]
            if len(gender_idx) == 0:
                continue
            gender_seq_list = [seq_list[i] for i in gender_idx]
            gender_batch_size = len(gender_idx)
            gender_batch = []

            gender_seq_texts = None
            gender_seq_dict = None
            for primitive_idx in range(self.num_primitive):
                primitive_texts = [mp_seq[primitive_idx]['text'] for mp_seq in gender_seq_list]
                primitive_dict = {'gender': gender}
                for key in ['betas', 'transl', 'global_orient', 'body_pose', 'transf_rotmat', 'transf_transl', 'pelvis_delta', 'joints']:
                    primitive_dict[key] = torch.cat([mp_seq[primitive_idx]['primitive_dict'][key] for mp_seq in gender_seq_list], dim=0)
                gender_seq_texts = primitive_texts if gender_seq_texts is None else gender_seq_texts + primitive_texts
                if gender_seq_dict is None:
                    gender_seq_dict = primitive_dict
                else:
                    for key in ['betas', 'transl', 'global_orient', 'body_pose', 'transf_rotmat', 'transf_transl', 'pelvis_delta', 'joints']:
                        gender_seq_dict[key] = torch.cat([gender_seq_dict[key], primitive_dict[key]], dim=0)

            gender_seq_dict = tensor_dict_to_device(gender_seq_dict, self.device)
            _, _, canonicalized_primitive_dict = self.primitive_utility.canonicalize(gender_seq_dict, use_predicted_joints=True)
            # print(f'{gender}:canonicalize time: ', time.time() - self.time)
            feature_dict = self.primitive_utility.calc_features(canonicalized_primitive_dict, use_predicted_joints=True)
            # print(f'{gender}:calc feature time: ', time.time() - self.time)
            feature_dict['transl'] = feature_dict['transl'][:, :-1, :]  # [B*num_mp, T, 3]
            feature_dict['poses_6d'] = feature_dict['poses_6d'][:, :-1, :]  # [B*num_mp, T, 66]
            feature_dict['joints'] = feature_dict['joints'][:, :-1, :]  # [B*num_mp, T, 22 * 3]
            motion_tensor_normalized = self.normalize(self.dict_to_tensor(feature_dict))  # [B*num_mp, T, D]
            motion_tensor_normalized = motion_tensor_normalized.permute(0, 2, 1).unsqueeze(2)  # [B*num_mp, D, 1, T]
            history_mask = torch.zeros_like(motion_tensor_normalized, dtype=torch.bool, device=self.device)
            history_mask[..., :self.cfg.history_length] = True
            history_motion = torch.zeros_like(motion_tensor_normalized, dtype=torch.float32, device=self.device)
            history_motion[..., :self.cfg.history_length] = motion_tensor_normalized[..., :self.cfg.history_length]

            for primitive_idx in range(self.num_primitive):
                start_idx = primitive_idx * gender_batch_size
                end_idx = (primitive_idx + 1) * gender_batch_size
                primitive_texts = gender_seq_texts[start_idx:end_idx]
                unseen_texts = [text for text in primitive_texts if text not in self.text_embedding_dict]
                if len(unseen_texts) > 0:
                    self.update_text_embedding_dict(unseen_texts)
                text_embedding = torch.stack([self.text_embedding_dict[text] for text in primitive_texts], dim=0)  # [B, 512]
                gender_batch.append(
                    {
                        'texts': primitive_texts,
                        'text_embedding': text_embedding,
                        'gender': [gender_seq_dict['gender']] * gender_batch_size,
                        'betas': gender_seq_dict['betas'][start_idx:end_idx, :-1, :10],
                        'motion_tensor_normalized': motion_tensor_normalized[start_idx:end_idx, ...], # [B, D, 1, T]
                        'history_motion': history_motion[start_idx:end_idx, ...],
                        'history_mask': history_mask[start_idx:end_idx, ...],
                        'history_length': self.cfg.history_length,
                        'future_length': self.cfg.future_length,
                    }
                )

            if batch is None:
                batch = gender_batch
            else:  # concatenate different gender batch
                for primitive_idx in range(self.num_primitive):
                    for key in ['texts', 'gender']:
                        batch[primitive_idx][key] = batch[primitive_idx][key] + gender_batch[primitive_idx][key]
                    for key in ['betas', 'motion_tensor_normalized', 'history_motion', 'history_mask', 'text_embedding']:
                        batch[primitive_idx][key] = torch.cat([batch[primitive_idx][key], gender_batch[primitive_idx][key]], dim=0)
            # print(f'{gender} batch time: ', time.time() - self.time)

        return batch

class SinglePrimitiveDataset(WeightedPrimitiveSequenceDataset):
    def __init__(self, cfg_path=None, sequence_path=None,
                 dataset_path=None,
                 device='cuda',
                 batch_size=16, texts=None,  #text: list of str lists
                 enforce_gender=None,
                 enforce_zero_beta=None,
                 clip_to_seq_length=True,
                 body_type='smplx',
                 **kwargs):
        self.batch_size = batch_size
        self.device = device
        self.prob_static = 0.0
        self.enforce_gender = enforce_gender
        self.enforce_zero_beta = enforce_zero_beta
        self.weight_scheme = 'uniform'
        self.enforce_gender = enforce_gender
        self.enforce_zero_beta = enforce_zero_beta
        self.clip_to_seq_length = clip_to_seq_length

        self.primitive_utility = PrimitiveUtility(device=self.device, body_type=body_type)
        self.motion_repr = self.primitive_utility.motion_repr

        # cfg_path = Path(dataset_path, 'config.yaml')
        with open(cfg_path, 'r') as f:
            self.cfg = OmegaConf.load(f)
        self.target_fps = self.cfg.fps
        self.history_length = self.cfg.history_length
        self.future_length = self.cfg.future_length
        self.primitive_length = self.history_length + self.future_length
        self.num_primitive = 1
        self.seq_length = self.history_length + self.future_length * self.num_primitive + 1

        self.tensor_mean_device_dict = {}
        if cfg_path == './config_files/config_hydra/motion_primitive/mp_2_8.yaml':  # backward compatibility
            mean_std_path = pjoin(dataset_path, 'mean_std.pkl')
            with open(mean_std_path, 'rb') as f:
                self.mean_std = pickle.load(f)
            mean_dict = {}
            std_dict = {}
            for key in self.mean_std:
                mean_dict[key] = self.mean_std[key]['mean']
                std_dict[key] = self.mean_std[key]['std']
            self.tensor_mean = self.dict_to_tensor(mean_dict).reshape(1, 1, -1)
            self.tensor_std = self.dict_to_tensor(std_dict).reshape(1, 1, -1)
        else:
            mean_std_path = Path(dataset_path, f'mean_std_h{self.history_length}_f{self.future_length}.pkl')
            if mean_std_path.exists():
                print(f'loading mean and std from {mean_std_path}')
                with open(mean_std_path, 'rb') as f:
                    self.tensor_mean, self.tensor_std = pickle.load(f)  # [1, 1, D]
            else:
                print('no mean std found, exit')
                exit()
        # load clip model
        self.clip_model = load_and_freeze_clip(clip_version='ViT-B/32', device=self.device)
        self.update_seq(sequence_path)



    def update_seq(self, sequence_path):
        with open(sequence_path, 'rb') as f:
            self.sequence = pickle.load(f)
        text_prompt = self.sequence['texts'][0] if 'texts' in self.sequence else ''
        clip_length = self.seq_length if self.clip_to_seq_length else len(self.sequence['transl'])
        body_pose = torch.tensor(self.sequence['body_pose'][:clip_length], dtype=torch.float32)
        if len(body_pose.shape) > 2 and body_pose.shape[-2:] == (3, 3):
            body_pose = transforms.matrix_to_axis_angle(body_pose).reshape(-1, 63)
        global_orient = torch.tensor(self.sequence['global_orient'][:clip_length], dtype=torch.float32)
        if len(global_orient.shape) > 2 and global_orient.shape[-2:] == (3, 3):
            global_orient = transforms.matrix_to_axis_angle(global_orient).reshape(-1, 3)
        poses = torch.cat([global_orient, body_pose], dim=1)
        transl = torch.tensor(self.sequence['transl'][:clip_length], dtype=torch.float32)
        if poses.shape[0] < self.seq_length:
            poses = torch.cat([poses, poses[-1].unsqueeze(0).expand(self.seq_length - poses.shape[0], -1)], dim=0)
            transl = torch.cat([transl, transl[-1].unsqueeze(0).expand(self.seq_length - transl.shape[0], -1)], dim=0)
        self.dataset = [{
            'motion':
                {
                    'gender': self.sequence['gender'] if self.enforce_gender is None else self.enforce_gender,
                    'betas': torch.tensor(self.sequence['betas'], dtype=torch.float32) if not self.enforce_zero_beta else torch.zeros(10, dtype=torch.float32),
                    'trans': transl,
                    'poses': poses,
                },
            'text': text_prompt,
        }]
        self.seq_weights = np.array([1.0])

        text_embedding = encode_text(self.clip_model, [text_prompt])
        self.text_embedding_dict = {text_prompt: text_embedding[0]}
