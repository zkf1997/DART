from data_loaders.humanml.data.dataset import *

class HML3dDataset(WeightedPrimitiveSequenceDataset):
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
                 seed_only=False,
                 use_frame_weights=True,
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
        self.seed_only = seed_only
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
        if seed_only:
            assert self.num_primitive == 1
        self.seq_length = self.history_length + self.future_length * self.num_primitive + 1

        if load_data:
            with open(pjoin(dataset_path, f'{split}.pkl'), 'rb') as f:
                dataset = pickle.load(f)
            dataset = [data for data in dataset if len(data['motion']['trans']) >= self.seq_length]

            for data in dataset:
                assert self.enforce_gender == data['motion']['gender']
                assert self.enforce_zero_beta
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


            for data in dataset:
                if 'uniform' in weight_scheme:
                    data['weight'] = 1.0
                elif 'length' in weight_scheme:
                    data['weight'] = len(data['motion']['trans'])
            print('finish first assigning seq weights')

            # overfit using one sequence
            # if 'overfit' in weight_scheme:
            #     seq_id = int(weight_scheme.split('overfit:')[-1].split('_')[0])
            #     for idx, data in enumerate(dataset):
            #         if idx == seq_id:
            #             data['weight'] = 1.0
            #         else:
            #             data['weight'] = 0.0

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
        if self.future_length == 1:
            batch_size = min(batch_size, 64)
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
                motion_tensor = self.dict_to_tensor(feature_dict).detach().cpu()  # [num_primitive, T, D]
                all_mp_data.append(motion_tensor)

                batch_start_idx = batch_end_idx

        all_mp_data = torch.cat(all_mp_data, dim=0)  # [N, T, D]
        tensor_mean = all_mp_data.mean(dim=[0, 1], keepdim=True)  # [1, 1, D]
        tensor_std = all_mp_data.std(dim=[0, 1], keepdim=True)  # [1, 1, D]
        return tensor_mean.to(self.device), tensor_std.to(self.device)

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
            elif self.seed_only:  # only take the first primitive for predicting initial seed
                # print('get seed')
                frame_labels = []
                for seg in seq_data['frame_labels']:
                    start_frame = int(seg['start_t'] * self.target_fps)
                    end_frame = start_frame + self.primitive_length
                    if end_frame < num_frames:
                        frame_labels.append((start_frame, end_frame, seg['proc_label']))
                start_frame, end_frame, text = random.choice(frame_labels)

                motion_data = seq_data['motion']
                primitive_dict = {
                    'gender': motion_data['gender'],
                    'betas': motion_data['betas'].expand(1, self.primitive_length + 1, 10),
                    'transl': motion_data['transl'][start_frame:end_frame + 1].unsqueeze(0),
                    # include one more frame for delta feature calculation
                    'global_orient': motion_data['global_orient'][start_frame:end_frame + 1].unsqueeze(0),
                    'body_pose': motion_data['body_pose'][start_frame:end_frame + 1].unsqueeze(0),
                    'pelvis_delta': motion_data['pelvis_delta'].unsqueeze(0),
                    'joints': motion_data['joints'][start_frame:end_frame + 1].unsqueeze(0),
                    'transf_rotmat': torch.eye(3).unsqueeze(0),
                    'transf_transl': torch.zeros(1, 1, 3),
                }

                primitive_data_list = [
                    {
                        'text': text,
                        'primitive_dict': primitive_dict,
                    }
                ]
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