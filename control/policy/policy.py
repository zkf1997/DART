import os, sys, glob
import time
from typing import NamedTuple
import random
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import loralib as lora

import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.distributions.normal import Normal

import pickle
import json
import pdb

class MLP(nn.Module):
    def __init__(self, in_dim,
                h_dims=[128,128], activation='tanh', use_lora=False, lora_rank=16):
        super().__init__()
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid
        elif activation == 'gelu':
            self.activation = torch.nn.GELU()
        elif activation == 'lrelu':
            self.activation = torch.nn.LeakyReLU()
        self.out_dim = h_dims[-1]
        self.layers = nn.ModuleList()
        in_dim_ = in_dim
        for h_dim in h_dims:
            layer = lora.Linear(in_dim_, h_dim, r=lora_rank) if use_lora else nn.Linear(in_dim_, h_dim)
            self.layers.append(layer)
            in_dim_ = h_dim

    def forward(self, x):
        for fc in self.layers:
            x = self.activation(fc(x))
        return x

class MLPBlock(nn.Module):
    def __init__(self, h_dim, out_dim, n_blocks, actfun='relu', residual=True, use_lora=False, lora_rank=16):
        super(MLPBlock, self).__init__()
        self.residual = residual
        self.layers = nn.ModuleList([MLP(h_dim, h_dims=(h_dim, h_dim),
                                        activation=actfun)
                                        for _ in range(n_blocks)]) # two fc layers in each MLP
        self.out_fc = lora.Linear(h_dim, out_dim, r=lora_rank) if use_lora else nn.Linear(h_dim, out_dim)

    def forward(self, x):
        h = x
        for layer in self.layers:
            r = h if self.residual else 0
            h = layer(h) + r
        y = self.out_fc(h)
        return y

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # require x to be of shape [seqlen, bs, d]
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)

class MotionEncoder(nn.Module):
    def __init__(self, latent_dim=512, num_heads=4, ff_size=512,
                 dropout=0.1, activation='gelu', num_layers=2,
                 seq_len=2, output_dim=256, input_dim=3
                 ):
        super(MotionEncoder, self).__init__()
        self.input_projection = nn.Linear(input_dim, latent_dim)
        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=latent_dim,
                                                          nhead=num_heads,
                                                          dim_feedforward=ff_size,
                                                          dropout=dropout,
                                                          activation=activation)

        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                     num_layers=num_layers)

        self.sequence_pos_encoder = PositionalEncoding(latent_dim, dropout)
        self.output_projection = nn.Linear(latent_dim, output_dim)

    def forward(self, x):
        x = self.input_projection(x)  # [seqlen, bs, d]
        x_pe = self.sequence_pos_encoder(x)  # [seqlen, bs, d]
        feature = self.seqTransEncoder(x_pe)  # [seqlen, bs, d]
        feature = feature.permute(1, 0, 2)  # [bs, seqlen, d]
        output = self.output_projection(feature).mean(dim=1)  # [bs, d_out]

        return output  # [bs, d_out]

class PolicyReachLocationMLP(nn.Module):
    def __init__(self, args):
        super(PolicyReachLocationMLP, self).__init__()
        self.args = args

        observation_structure = args.observation_structure
        motion_dim = observation_structure['motion']['numel']
        text_dim = observation_structure['goal_text_embedding']['numel']
        goal_dim = observation_structure['goal_dir']['numel'] + observation_structure['goal_dist']['numel']
        scene_dim = observation_structure['scene']['numel']
        latent_dim = args.latent_dim
        self.action_dim = action_dim = args.action_structure['numel']
        activation = args.activation

        self.motion_encoder = MLP(in_dim=motion_dim, h_dims=[latent_dim, ], activation=activation)
        self.text_encoder = MLP(in_dim=text_dim, h_dims=[latent_dim, ], activation=activation)
        self.goal_encoder = MLP(in_dim=goal_dim, h_dims=[latent_dim, ], activation=activation)
        self.scene_encoder = MLP(in_dim=scene_dim, h_dims=[latent_dim, ], activation=activation)
        self.embedding_encoder = MLP(in_dim=latent_dim * 4, h_dims=[latent_dim, ], activation=activation)
        # self.embedding_encoder = MLP(in_dim=latent_dim * 3, h_dims=[latent_dim, ], activation=activation)
        # self.embedding_encoder = MLP(in_dim=motion_dim + text_dim + goal_dim, h_dims=[latent_dim, ], activation=activation)

        self.actor = MLPBlock(latent_dim,
                              action_dim * 2 if self.args.pred_std else action_dim,
                              args.n_blocks,
                              use_lora=args.use_lora,
                              lora_rank=args.lora_rank,
                              actfun=activation)
        if not self.args.pred_std:
            self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
            # print('logstd param grad:', self.actor_logstd.requires_grad)

        # do last policy layer scaling, this will make initial actions have (close to)
        # 0 mean and std, and will help boost performances,
        # see https://arxiv.org/abs/2006.05990, Fig.24 for details
        if self.args.use_zero_init:
            for m in self.actor.modules():
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.zeros_(m.bias)
                    m.weight.data.copy_(0.01 * m.weight.data)

        self.critic = MLPBlock(latent_dim,
                             1,
                             args.n_blocks,
                             actfun=activation)

    def get_embedding(self, observation):
        batch_size = observation.shape[0]
        observation_structure = self.args.observation_structure
        start_idx, end_idx = observation_structure['motion']['start_idx'], observation_structure['motion']['end_idx']
        motion = observation[:, start_idx:end_idx]

        start_idx, end_idx = observation_structure['goal_text_embedding']['start_idx'], observation_structure['goal_text_embedding']['end_idx']
        text_embedding = observation[:, start_idx:end_idx]

        start_idx, end_idx = observation_structure['goal_dir']['start_idx'], observation_structure['goal_dir']['end_idx']
        goal_dir = observation[:, start_idx:end_idx]
        start_idx, end_idx = observation_structure['goal_dist']['start_idx'], observation_structure['goal_dist']['end_idx']
        goal_dist = observation[:, start_idx:end_idx]
        goal = torch.cat((goal_dir, goal_dist), dim=1)  # [bs, d]

        start_idx, end_idx = observation_structure['scene']['start_idx'], observation_structure['scene']['end_idx']
        scene = observation[:, start_idx:end_idx]

        motion_embedding = self.motion_encoder(motion)
        text_embedding = self.text_encoder(text_embedding)
        goal_embedding = self.goal_encoder(goal)
        scene_embedding = self.scene_encoder(scene)

        embedding = torch.cat((motion_embedding, text_embedding, goal_embedding, scene_embedding), dim=1)
        # embedding = torch.cat((motion_embedding, text_embedding, goal_embedding), dim=1)
        # embedding = torch.cat((motion, text_embedding, goal), dim=1)
        embedding = self.embedding_encoder(embedding)
        return embedding

    def get_value(self, x):
        embedding = self.get_embedding(x)
        return self.critic(embedding)

    def get_action_and_value(self, x, action=None):
        embedding = self.get_embedding(x)

        actor_output = self.actor(embedding)
        action_mean = actor_output[:, :self.action_dim]
        if self.args.use_tanh_scale:
            action_mean = torch.tanh(action_mean) * 4
        action_logstd = actor_output[:, self.action_dim:] if self.args.pred_std else self.actor_logstd.expand_as(action_mean)
        action_logstd = action_logstd.clamp(min=self.args.min_log_std, max=self.args.max_log_std)
        if torch.isnan(action_mean).any() or torch.isnan(action_logstd).any():
            print(action_mean, action_logstd)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)

        if action is None:
            action = probs.sample()
            # action = torch.zeros_like(action)
        log_prob = probs.log_prob(action).sum(1)
        entropy = probs.entropy().sum(1)
        value = self.critic(embedding)

        return action, log_prob, entropy, value
        # return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

class PolicyInteractionMLP(nn.Module):
    def __init__(self, args):
        super(PolicyInteractionMLP, self).__init__()
        self.args = args

        observation_structure = args.observation_structure
        motion_dim = observation_structure['motion']['numel']
        text_dim = observation_structure['goal_text_embedding']['numel']
        goal_dim = observation_structure['goal_joints']['numel']
        scene_dim = observation_structure['scene_bps']['numel']
        latent_dim = args.latent_dim
        self.action_dim = action_dim = args.action_structure['numel']
        activation = args.activation

        self.motion_encoder = MLP(in_dim=motion_dim, h_dims=[latent_dim, ], activation=activation)
        self.text_encoder = MLP(in_dim=text_dim, h_dims=[latent_dim, ], activation=activation)
        self.goal_encoder = MLP(in_dim=goal_dim, h_dims=[latent_dim, ], activation=activation)
        self.scene_encoder = MLP(in_dim=scene_dim, h_dims=[latent_dim, ], activation=activation)
        self.embedding_encoder = MLP(in_dim=latent_dim * 4, h_dims=[latent_dim, ], activation=activation)
        # self.embedding_encoder = MLP(in_dim=latent_dim * 3, h_dims=[latent_dim, ], activation=activation)
        # self.embedding_encoder = MLP(in_dim=motion_dim + text_dim + goal_dim, h_dims=[latent_dim, ], activation=activation)

        self.actor = MLPBlock(latent_dim,
                              action_dim * 2 if self.args.pred_std else action_dim,
                              args.n_blocks,
                              use_lora=args.use_lora,
                              lora_rank=args.lora_rank,
                              actfun=activation)
        if not self.args.pred_std:
            self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
            # print('logstd param grad:', self.actor_logstd.requires_grad)

        # do last policy layer scaling, this will make initial actions have (close to)
        # 0 mean and std, and will help boost performances,
        # see https://arxiv.org/abs/2006.05990, Fig.24 for details
        if self.args.use_zero_init:
            for m in self.actor.modules():
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.zeros_(m.bias)
                    m.weight.data.copy_(0.01 * m.weight.data)

        self.critic = MLPBlock(latent_dim,
                             1,
                             args.n_blocks,
                             actfun=activation)

    def get_embedding(self, observation):
        batch_size = observation.shape[0]
        observation_structure = self.args.observation_structure
        start_idx, end_idx = observation_structure['motion']['start_idx'], observation_structure['motion']['end_idx']
        motion = observation[:, start_idx:end_idx]

        start_idx, end_idx = observation_structure['goal_text_embedding']['start_idx'], observation_structure['goal_text_embedding']['end_idx']
        text_embedding = observation[:, start_idx:end_idx]

        start_idx, end_idx = observation_structure['goal_joints']['start_idx'], observation_structure['goal_joints']['end_idx']
        goal_joints = observation[:, start_idx:end_idx]
        goal = goal_joints

        start_idx, end_idx = observation_structure['scene_bps']['start_idx'], observation_structure['scene_bps']['end_idx']
        scene_bps = observation[:, start_idx:end_idx]
        scene = scene_bps

        motion_embedding = self.motion_encoder(motion)
        text_embedding = self.text_encoder(text_embedding)
        goal_embedding = self.goal_encoder(goal)
        scene_embedding = self.scene_encoder(scene)

        embedding = torch.cat((motion_embedding, text_embedding, goal_embedding, scene_embedding), dim=1)
        # embedding = torch.cat((motion_embedding, text_embedding, goal_embedding), dim=1)
        # embedding = torch.cat((motion, text_embedding, goal), dim=1)
        embedding = self.embedding_encoder(embedding)
        return embedding

    def get_value(self, x):
        embedding = self.get_embedding(x)
        return self.critic(embedding)

    def get_action_and_value(self, x, action=None):
        embedding = self.get_embedding(x)

        actor_output = self.actor(embedding)
        action_mean = actor_output[:, :self.action_dim]
        if self.args.use_tanh_scale:
            action_mean = torch.tanh(action_mean) * 4
        action_logstd = actor_output[:, self.action_dim:] if self.args.pred_std else self.actor_logstd.expand_as(action_mean)
        action_logstd = action_logstd.clamp(min=self.args.min_log_std, max=self.args.max_log_std)
        if torch.isnan(action_mean).any() or torch.isnan(action_logstd).any():
            print(action_mean, action_logstd)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)

        if action is None:
            action = probs.sample()
            # action = torch.zeros_like(action)
        log_prob = probs.log_prob(action).sum(1)
        entropy = probs.entropy().sum(1)
        value = self.critic(embedding)

        return action, log_prob, entropy, value


class PolicyReachLocationTransformer(nn.Module):
    def __init__(self, args):
        super(PolicyReachLocationTransformer, self).__init__()
        self.args = args

        observation_structure = args.observation_structure
        motion_shape = observation_structure['motion']['shape']
        text_dim = observation_structure['goal_text_embedding']['numel']
        goal_dim = observation_structure['goal_dir']['numel'] + observation_structure['goal_dist']['numel']
        scene_dim = observation_structure['scene']['numel']
        latent_dim = args.latent_dim
        self.action_dim = action_dim = args.action_structure['numel']
        self.action_shape = args.action_structure['shape']   # [future_length, D]
        activation = 'gelu'
        dropout = args.dropout

        self.motion_encoder = MotionEncoder(latent_dim=512, num_heads=4, ff_size=512,
                                            dropout=dropout, activation='gelu', num_layers=2,
                                            seq_len=motion_shape[0], input_dim=motion_shape[1],
                                            output_dim=latent_dim)
        self.text_encoder = MLP(in_dim=text_dim, h_dims=[latent_dim, ], activation=activation)
        self.goal_encoder = MLP(in_dim=goal_dim, h_dims=[latent_dim, ], activation=activation)
        self.scene_encoder = MLP(in_dim=scene_dim, h_dims=[latent_dim, ], activation=activation)

        if self.args.architecture == 'transformer_encoder':
            seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=latent_dim,
                                                              nhead=4,
                                                              dim_feedforward=512,
                                                              dropout=dropout,
                                                              activation=activation)
            self.actor = nn.TransformerEncoder(seqTransEncoderLayer, num_layers=2)
        elif self.args.architecture == 'transformer_decoder':
            seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=latent_dim,
                                                              nhead=4,
                                                              dim_feedforward=512,
                                                              dropout=dropout,
                                                              activation=activation)
            self.actor = nn.TransformerDecoder(seqTransDecoderLayer, num_layers=2)

        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=latent_dim,
                                                          nhead=4,
                                                          dim_feedforward=512,
                                                          dropout=dropout,
                                                          activation=activation)
        self.critic = nn.TransformerEncoder(seqTransEncoderLayer, num_layers=2)  # critic always uses encoder

        self.sequence_pos_encoder = PositionalEncoding(latent_dim, dropout)
        self.output_projection = nn.Linear(latent_dim, self.action_shape[1] * 2 if self.args.pred_std else self.action_shape[1])


        if not self.args.pred_std:
            self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
            # print('logstd param grad:', self.actor_logstd.requires_grad)

        # do last policy layer scaling, this will make initial actions have (close to)
        # 0 mean and std, and will help boost performances,
        # see https://arxiv.org/abs/2006.05990, Fig.24 for details
        m = self.output_projection
        torch.nn.init.zeros_(m.bias)
        m.weight.data.copy_(0.01 * m.weight.data)

    def get_embedding(self, observation):
        batch_size = observation.shape[0]
        observation_structure = self.args.observation_structure
        start_idx, end_idx = observation_structure['motion']['start_idx'], observation_structure['motion']['end_idx']
        motion = observation[:, start_idx:end_idx]

        start_idx, end_idx = observation_structure['goal_text_embedding']['start_idx'], observation_structure['goal_text_embedding']['end_idx']
        text_embedding = observation[:, start_idx:end_idx]

        start_idx, end_idx = observation_structure['goal_dir']['start_idx'], observation_structure['goal_dir']['end_idx']
        goal_dir = observation[:, start_idx:end_idx]
        start_idx, end_idx = observation_structure['goal_dist']['start_idx'], observation_structure['goal_dist']['end_idx']
        goal_dist = observation[:, start_idx:end_idx]
        goal = torch.cat((goal_dir, goal_dist), dim=1)  # [bs, d]

        start_idx, end_idx = observation_structure['scene']['start_idx'], observation_structure['scene']['end_idx']
        scene = observation[:, start_idx:end_idx]

        motion = motion.reshape((batch_size,) + observation_structure['motion']['shape']).permute(1, 0, 2)  # [seqlen, bs, d]
        motion_embedding = self.motion_encoder(motion)  # [bs, d]
        text_embedding = self.text_encoder(text_embedding)
        goal_embedding = self.goal_encoder(goal)
        scene_embedding = self.scene_encoder(scene)

        embedding = torch.stack((motion_embedding, text_embedding, goal_embedding, scene_embedding), dim=0)  # [4, bs, d]
        return embedding

    def get_value(self, x):
        embedding = self.get_embedding(x)
        embedding = self.sequence_pos_encoder(embedding)  # [4, bs, d]
        value = self.critic(embedding)  # [4, bs, d]
        value = value.permute(1, 0, 2).mean(dim=(1, 2))  # [bs,]
        return value

    def get_action_and_value(self, x, action=None):
        embedding = self.get_embedding(x)

        future_len = self.action_shape[0]
        batch_size = x.shape[0]
        zero_input = torch.zeros(future_len, batch_size, self.args.latent_dim).to(device=x.device)
        if self.args.architecture == 'transformer_encoder':
            seq_input = torch.cat((embedding, zero_input), dim=0)  # [T, bs, d]
            seq_input = self.sequence_pos_encoder(seq_input)  # [T, bs, d]
            actor_output = self.actor(seq_input)  # [T, bs, d]
            actor_output = actor_output[-future_len:, :, :]  # [future, bs, d]
            actor_output = self.output_projection(actor_output).permute(1, 0, 2)  # [bs, future, d_out]
        elif self.args.architecture == 'transformer_decoder':
            seq_input = zero_input
            seq_input = self.sequence_pos_encoder(seq_input)  # [T, bs, d]
            memory = self.sequence_pos_encoder(embedding)  # [4, bs, d]
            actor_output = self.actor(seq_input, memory)
            actor_output = self.output_projection(actor_output).permute(1, 0, 2)  # [bs, future, d_out]

        action_mean = actor_output[:, :, :self.action_shape[1]].reshape(batch_size, -1)
        action_logstd = actor_output[:, :, self.action_shape[1]:].reshape(batch_size, -1) if self.args.pred_std else self.actor_logstd.expand_as(action_mean)
        action_logstd = action_logstd.clamp(min=self.args.min_log_std, max=self.args.max_log_std)
        if torch.isnan(action_mean).any() or torch.isnan(action_logstd).any():
            print(action_mean, action_logstd)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)

        if action is None:
            action = probs.sample()
            # action = torch.zeros_like(action)
        log_prob = probs.log_prob(action).sum(1)
        entropy = probs.entropy().sum(1)

        value = self.critic(self.sequence_pos_encoder(embedding))  # [4, bs, d]
        value = value.permute(1, 0, 2).mean(dim=(1, 2))  # [bs,]

        return action, log_prob, entropy, value
        # return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


class PolicyBetweenMLP(nn.Module):
    def __init__(self, args):
        super(PolicyBetweenMLP, self).__init__()
        self.args = args

        observation_structure = args.observation_structure
        motion_dim = observation_structure['motion']['numel']
        text_dim = observation_structure['goal_text_embedding']['numel']
        goal_dim = observation_structure['goal_joints']['numel']
        scene_dim = observation_structure['scene']['numel']
        latent_dim = args.latent_dim
        self.action_dim = action_dim = args.action_structure['numel']
        activation = args.activation

        self.motion_encoder = MLP(in_dim=motion_dim, h_dims=[latent_dim, ], activation=activation)
        self.text_encoder = MLP(in_dim=text_dim, h_dims=[latent_dim, ], activation=activation)
        self.goal_encoder = MLP(in_dim=goal_dim, h_dims=[latent_dim, ], activation=activation)
        self.scene_encoder = MLP(in_dim=scene_dim, h_dims=[latent_dim, ], activation=activation)
        # self.embedding_encoder = MLP(in_dim=latent_dim * 4, h_dims=[latent_dim, ], activation=activation)
        self.embedding_encoder = MLP(in_dim=latent_dim * 3, h_dims=[latent_dim, ], activation=activation)
        # self.embedding_encoder = MLP(in_dim=motion_dim + text_dim + goal_dim, h_dims=[latent_dim, ], activation=activation)

        self.actor = MLPBlock(latent_dim,
                              action_dim * 2 if self.args.pred_std else action_dim,
                              args.n_blocks,
                              use_lora=args.use_lora,
                              lora_rank=args.lora_rank,
                              actfun=activation)
        if not self.args.pred_std:
            self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
            # print('logstd param grad:', self.actor_logstd.requires_grad)

        # do last policy layer scaling, this will make initial actions have (close to)
        # 0 mean and std, and will help boost performances,
        # see https://arxiv.org/abs/2006.05990, Fig.24 for details
        if self.args.use_zero_init:
            for m in self.actor.modules():
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.zeros_(m.bias)
                    m.weight.data.copy_(0.01 * m.weight.data)

        self.critic = MLPBlock(latent_dim,
                             1,
                             args.n_blocks,
                             actfun=activation)

    def get_embedding(self, observation):
        batch_size = observation.shape[0]
        observation_structure = self.args.observation_structure
        start_idx, end_idx = observation_structure['motion']['start_idx'], observation_structure['motion']['end_idx']
        motion = observation[:, start_idx:end_idx]

        start_idx, end_idx = observation_structure['goal_text_embedding']['start_idx'], observation_structure['goal_text_embedding']['end_idx']
        text_embedding = observation[:, start_idx:end_idx]

        start_idx, end_idx = observation_structure['goal_joints']['start_idx'], observation_structure['goal_joints']['end_idx']
        goal_joints = observation[:, start_idx:end_idx]
        goal = goal_joints  # [bs, d]

        start_idx, end_idx = observation_structure['scene']['start_idx'], observation_structure['scene']['end_idx']
        scene = observation[:, start_idx:end_idx]

        motion_embedding = self.motion_encoder(motion)
        text_embedding = self.text_encoder(text_embedding)
        goal_embedding = self.goal_encoder(goal)
        scene_embedding = self.scene_encoder(scene)

        # embedding = torch.cat((motion_embedding, text_embedding, goal_embedding, scene_embedding), dim=1)
        embedding = torch.cat((motion_embedding, text_embedding, goal_embedding), dim=1)
        # embedding = torch.cat((motion, text_embedding, goal), dim=1)
        embedding = self.embedding_encoder(embedding)
        return embedding

    def get_value(self, x):
        embedding = self.get_embedding(x)
        return self.critic(embedding)

    def get_action_and_value(self, x, action=None):
        embedding = self.get_embedding(x)
        value = self.critic(embedding)

        actor_output = self.actor(embedding)
        action_mean = actor_output[:, :self.action_dim]
        if self.args.use_tanh_scale:
            action_mean = torch.tanh(action_mean) * 4
        action_logstd = actor_output[:, self.action_dim:] if self.args.pred_std else self.actor_logstd.expand_as(action_mean)
        action_logstd = action_logstd.clamp(min=self.args.min_log_std, max=self.args.max_log_std)
        if torch.isnan(action_mean).any() or torch.isnan(action_logstd).any():
            print(action_mean, action_logstd)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)

        if action is None:
            action = probs.sample()
            if self.args.use_tanh_scale:
                action = action.clamp(min=-4.0, max=4.0)  # clamp to [-4, 4]
            # action = torch.zeros_like(action)
        log_prob = probs.log_prob(action).sum(1)
        entropy = probs.entropy().sum(1)

        return action, log_prob, entropy, value

class PolicyBetweenTransformer(nn.Module):
    def __init__(self, args):
        super(PolicyBetweenTransformer, self).__init__()
        self.args = args

        observation_structure = args.observation_structure
        motion_dim = observation_structure['motion']['numel']
        text_dim = observation_structure['goal_text_embedding']['numel']
        goal_dim = observation_structure['goal_joints']['numel']
        scene_dim = observation_structure['scene']['numel']
        latent_dim = args.latent_dim
        self.action_dim = action_dim = args.action_structure['numel']
        self.action_shape = args.action_structure['shape']  # [future_length, D]
        activation = 'gelu'
        dropout = args.dropout
        num_layers = args.num_layers

        self.motion_encoder = MLP(in_dim=motion_dim, h_dims=[latent_dim, ], activation=activation)
        self.text_encoder = MLP(in_dim=text_dim, h_dims=[latent_dim, ], activation=activation)
        self.goal_encoder = MLP(in_dim=goal_dim, h_dims=[latent_dim, ], activation=activation)
        # self.scene_encoder = MLP(in_dim=scene_dim, h_dims=[latent_dim, ], activation=activation)

        if self.args.architecture == 'transformer_encoder':
            seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=latent_dim,
                                                              nhead=4,
                                                              dim_feedforward=args.dim_feedforward,
                                                              dropout=dropout,
                                                              activation=activation)
            self.actor = nn.TransformerEncoder(seqTransEncoderLayer, num_layers=num_layers)
        elif self.args.architecture == 'transformer_decoder':
            seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=latent_dim,
                                                              nhead=4,
                                                              dim_feedforward=args.dim_feedforward,
                                                              dropout=dropout,
                                                              activation=activation)
            self.actor = nn.TransformerDecoder(seqTransDecoderLayer, num_layers=num_layers)

        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=latent_dim,
                                                          nhead=4,
                                                          dim_feedforward=args.dim_feedforward,
                                                          dropout=dropout,
                                                          activation=activation)
        self.critic = nn.TransformerEncoder(seqTransEncoderLayer, num_layers=num_layers)  # critic always uses encoder

        self.sequence_pos_encoder = PositionalEncoding(latent_dim, dropout)
        self.output_projection = nn.Linear(latent_dim, self.action_shape[1] * 2 if self.args.pred_std else self.action_shape[1])


        if not self.args.pred_std:
            self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
            # print('logstd param grad:', self.actor_logstd.requires_grad)

        # do last policy layer scaling, this will make initial actions have (close to)
        # 0 mean and std, and will help boost performances,
        # see https://arxiv.org/abs/2006.05990, Fig.24 for details
        if self.args.use_zero_init:
            m = self.output_projection
            torch.nn.init.zeros_(m.bias)
            m.weight.data.copy_(0.01 * m.weight.data)
            # for m in self.actor.modules():
            #     if isinstance(m, torch.nn.Linear):
            #         torch.nn.init.zeros_(m.bias)
            #         m.weight.data.copy_(0.01 * m.weight.data)

    def get_embedding(self, observation):
        batch_size = observation.shape[0]
        observation_structure = self.args.observation_structure
        start_idx, end_idx = observation_structure['motion']['start_idx'], observation_structure['motion']['end_idx']
        motion = observation[:, start_idx:end_idx]

        start_idx, end_idx = observation_structure['goal_text_embedding']['start_idx'], observation_structure['goal_text_embedding']['end_idx']
        text_embedding = observation[:, start_idx:end_idx]

        start_idx, end_idx = observation_structure['goal_joints']['start_idx'], observation_structure['goal_joints']['end_idx']
        goal_joints = observation[:, start_idx:end_idx]
        goal = goal_joints  # [bs, d]

        # start_idx, end_idx = observation_structure['scene']['start_idx'], observation_structure['scene']['end_idx']
        # scene = observation[:, start_idx:end_idx]

        motion_embedding = self.motion_encoder(motion)
        text_embedding = self.text_encoder(text_embedding)
        goal_embedding = self.goal_encoder(goal)
        # scene_embedding = self.scene_encoder(scene)

        # embedding = torch.stack((motion_embedding, text_embedding, goal_embedding, scene_embedding), dim=0)  # [4, bs, d]
        embedding = torch.stack((motion_embedding, text_embedding, goal_embedding), dim=0)  # [3, bs, d]
        return embedding

    def get_value(self, x):
        embedding = self.get_embedding(x)
        embedding = self.sequence_pos_encoder(embedding)  # [3, bs, d]
        value = self.critic(embedding)  # [3, bs, d]
        value = value.permute(1, 0, 2).mean(dim=(1, 2))  # [bs,]
        return value

    def get_action_and_value(self, x, action=None):
        embedding = self.get_embedding(x)

        value = self.critic(self.sequence_pos_encoder(embedding))  # [3, bs, d]
        value = value.permute(1, 0, 2).mean(dim=(1, 2))  # [bs,]

        future_len = self.action_shape[0]  # F=1 in mld
        batch_size = x.shape[0]
        zero_input = torch.zeros(future_len, batch_size, self.args.latent_dim).to(device=x.device)
        if self.args.architecture == 'transformer_encoder':
            seq_input = torch.cat((embedding, zero_input), dim=0)  # [T, bs, d]
            seq_input = self.sequence_pos_encoder(seq_input)  # [T, bs, d]
            actor_output = self.actor(seq_input)  # [T, bs, d]
            actor_output = actor_output[-future_len:, :, :]  # [future, bs, d]
            actor_output = self.output_projection(actor_output).permute(1, 0, 2)  # [bs, future, d_out]
        elif self.args.architecture == 'transformer_decoder':
            seq_input = zero_input
            seq_input = self.sequence_pos_encoder(seq_input)  # [T, bs, d]
            memory = self.sequence_pos_encoder(embedding)  # [3, bs, d]
            actor_output = self.actor(seq_input, memory)
            actor_output = self.output_projection(actor_output).permute(1, 0, 2)  # [bs, future, d_out]

        action_mean = actor_output[:, :, :self.action_shape[1]].reshape(batch_size, -1)  # [bs, future*D]
        if self.args.use_tanh_scale:
            action_mean = torch.tanh(action_mean) * 4
        action_logstd = actor_output[:, :, self.action_shape[1]:].reshape(batch_size, -1) if self.args.pred_std else self.actor_logstd.expand_as(action_mean)
        action_logstd = action_logstd.clamp(min=self.args.min_log_std, max=self.args.max_log_std)
        if torch.isnan(action_mean).any() or torch.isnan(action_logstd).any():
            print(action_mean, action_logstd)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)

        if action is None:
            action = probs.sample()
            if self.args.use_tanh_scale:
                action = action.clamp(min=-4.0, max=4.0)  # clamp to [-4, 4]
            # action = torch.zeros_like(action)
        log_prob = probs.log_prob(action).sum(1)
        entropy = probs.entropy().sum(1)

        return action, log_prob, entropy, value