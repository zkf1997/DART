# adapted from https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py
from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass, asdict, make_dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
import yaml
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

from utils.parser_util import load_args_from_model
from utils.model_util import create_model_and_diffusion, load_model_wo_clip, load_saved_model
from utils import dist_util
from torch.utils.data import DataLoader
from model.cfg_sampler import ClassifierFreeSampleModel
from data_loaders.humanml.data.dataset import SinglePrimitiveDataset
from utils.smpl_utils import *
from utils.misc_util import encode_text, compose_texts_with_and, dict_to_args
from pytorch3d import transforms
from tqdm import tqdm

from control.env.env_reach_location_mld import EnvReachLocationMLD
from control.policy.policy import PolicyReachLocationMLP, PolicyReachLocationTransformer

from mld.train_mvae import Args as MVAEArgs
from mld.train_mvae import DataArgs, TrainArgs
from mld.train_mld import DenoiserArgs, MLDArgs, create_gaussian_diffusion, DenoiserMLPArgs, DenoiserTransformerArgs
from mld.rollout_mld import load_mld, ClassifierFreeWrapper

debug = 0


@dataclass
class PolicyArgs:
    architecture: str = 'mlp'
    latent_dim: int = 512
    n_blocks: int = 2
    dropout: float = 0.1
    activation: str = 'lrelu'

    min_log_std: float = -1.0
    max_log_std: float = 1.0
    """clip range for log std"""

    pred_std: int = 0
    """whether to predict std or not"""

    use_tanh_scale: int = 0
    """whether to use tanh scale on the final layer output"""

    use_zero_init: int = 0
    """whether to initialize the network weights with zero bias and weights of small variance """

    use_lora: int = 1
    """obsolete"""
    lora_rank: int = 16
    """obsolete"""


@dataclass
class EnvArgs:
    env_id: str = "reach_location_mld"
    """the id of the environment"""
    num_envs: int = 128
    """the number of parallel game environments"""
    num_steps: int = 256
    """the number of steps to run in each environment per policy rollout"""
    texts: tuple[str, ...] = ('walk',)
    """the texts describing the locomotion skills"""

    use_predicted_joints: int = 1
    """if set to 1, use predicted joints to rollout, otherwise use the regressed joints from smplx body model"""

    """parameters for goal location curriculum"""
    goal_range: float = 1.0
    goal_angle_init: float = 0.0
    goal_angle_delta: float = 120.0
    goal_dist_min: float = 0.5
    goal_dist_max_init: float = 2.0
    goal_dist_max_delta: float = 1.0
    goal_dist_max_clamp: float = 5.0
    goal_schedule_interval: int = 10000

    """parameters for goal observation angle and distance clipping"""
    obs_goal_angle_clip: float = 180.0
    obs_goal_dist_clip: float = 5.0

    success_threshold: float = 0.3
    """distance threshold for success"""
    terminate_threshold: float = 100.0
    """distance threshold for termination"""

    weight_success: float = 10.0
    weight_dist: float = 1.0
    weight_foot_floor: float = 1.0
    weight_skate: float = 1.0
    weight_skate_delta: float = 0.0
    weight_skate_max: float = 1.0
    weight_skate_rigid: float = 0.0
    weight_orient: float = 1.0
    weight_rotation: float = 0.0
    weight_jerk: float = 0.0
    weight_delta: float = 0.0

    enable_export: int = 1
    export_interval: int = 100
    max_export: int = 16
    smooth_rollout: int = 0

@dataclass
class ReachLocationArgs:
    exp_name: str = 'test'
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    device: str = 'cuda'
    track: int = 1
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "reach_location"
    """the wandb's project name"""
    wandb_entity: str = "interaction"
    """the entity (team) of wandb's project"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""

    # Algorithm specific arguments
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    minibatch_size: int = 1024
    """the mini-batch size"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size for rl, the batch size for motion model is num_envs (computed in runtime)"""
    num_minibatches: int = 0
    """the number of mini-batches"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

    env_args: EnvArgs = EnvArgs()
    policy_args: PolicyArgs = PolicyArgs()
    denoiser_checkpoint: str = ''
    respacing: str = 'ddim10'
    guidance_param: float = 5.0
    init_data_path: str = './data/rl_seed/walk.pkl'
    motion_model_args: dict = None
    resume_checkpoint: str = None
    test_goal_path: str = None


def load_motion_model(input_args):
    init_data_path, device, batch_size = input_args.init_data_path, input_args.device, input_args.num_envs

    denoiser_args, denoiser_model, vae_args, vae_model = load_mld(input_args.denoiser_checkpoint, device)

    diffusion_args = denoiser_args.diffusion_args
    diffusion_args.respacing = input_args.respacing
    print('diffusion_args:', asdict(diffusion_args))
    diffusion = create_gaussian_diffusion(diffusion_args)

    # load initial seed dataset
    dataset = SinglePrimitiveDataset(cfg_path=vae_args.data_args.cfg_path,  # cfg path from model checkpoint
                                     dataset_path=vae_args.data_args.data_dir,  # dataset path from model checkpoint
                                     sequence_path=init_data_path,
                                     batch_size=input_args.batch_size,
                                     device=device,
                                     enforce_gender='male',
                                     enforce_zero_beta=1,
                                     clip_to_seq_length=False,
                                     )

    return denoiser_args, denoiser_model, vae_args, vae_model, diffusion, dataset

def set_up(arg_path=None):
    if arg_path is None:
        args = tyro.cli(ReachLocationArgs)
    else:
        with open(arg_path, "r") as f:
            args = tyro.extras.from_yaml(ReachLocationArgs, yaml.safe_load(f))
    args.env_id = args.env_args.env_id
    args.num_envs = args.env_args.num_envs
    args.num_steps = args.env_args.num_steps
    args.batch_size = args.num_envs * args.num_steps
    args.num_minibatches = args.batch_size // args.minibatch_size  # not used
    args.num_iterations = args.total_timesteps // args.batch_size

    args.save_dir = Path(f"./policy_train/{args.env_id}/{args.exp_name}")
    args.save_dir.mkdir(parents=True, exist_ok=True)

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_default_dtype(torch.float32)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.device = device

    # load motion model
    denoiser_args, denoiser_model, vae_args, vae_model, diffusion, dataset = load_motion_model(args)

    # create env
    # args.env_args.texts = ['walk']
    args.env_args.save_dir = args.save_dir / 'env'
    args.env_args.save_dir.mkdir(parents=True, exist_ok=True)
    env = EnvReachLocationMLD(args, denoiser_args, denoiser_model, vae_args, vae_model, diffusion, dataset)

    # create agent
    policy_args = args.policy_args
    policy_args.observation_structure = env.observation_structure
    policy_args.action_structure = env.action_structure
    if policy_args.architecture == 'mlp':
        agent = PolicyReachLocationMLP(policy_args).to(device)
    else:
        agent = PolicyReachLocationTransformer(policy_args).to(device)
    if args.resume_checkpoint is not None:
        print(f"Loading checkpoint from {args.resume_checkpoint}")
        agent.load_state_dict(torch.load(args.resume_checkpoint))
        agent = agent.to(device)

    if arg_path is None:
        with open(args.save_dir / "args.yaml", "w") as f:
            yaml.dump(tyro.extras.to_yaml(args), f)
        with open(args.save_dir / "args_read.yaml", "w") as f:
            yaml.dump(asdict(args), f)

    return args, env, agent

if __name__ == "__main__":
    args, env, agent = set_up()
    run_name = f"{args.exp_name}__seed{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            save_code=True,
            settings=wandb.Settings(code_dir="./control"),
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # exit()

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + env.observation_shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + env.action_shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = env.reset()
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in tqdm(range(1, args.num_iterations + 1)):
        # t1 = time.time()
        env.global_iteration = iteration - 1
        if iteration % args.env_args.goal_schedule_interval == 0:
            env.curriculum_step()
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        num_success = 0
        num_terminated = 0
        num_truncated = 0
        reward_dict = None
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, success, terminations, truncations, infos = env.step(action)
            next_done = terminations | truncations
            rewards[step] = reward.view(-1)
            # next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            num_success += infos['num_success']
            num_terminated += infos['num_terminated']
            num_truncated += infos['num_truncated']
            if reward_dict is None:
                reward_dict = {}
                for key in infos['reward_dict']:
                    reward_dict[key] = [infos['reward_dict'][key]]
            else:
                for key in infos['reward_dict']:
                    reward_dict[key] += [infos['reward_dict'][key]]

        writer.add_scalar("done/num_success", num_success, global_step)
        writer.add_scalar("done/num_terminated", num_terminated, global_step)
        writer.add_scalar("done/num_truncated", num_truncated, global_step)
        writer.add_scalar("reward/reward", rewards.mean().item(), global_step)
        for key in reward_dict:
            writer.add_scalar(f"reward/{key}", torch.stack(reward_dict[key]).mean().item(), global_step)

        # rollout_time = time.time()

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done.float()
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1].float()
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + env.observation_shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + env.action_shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        # b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            # np.random.shuffle(b_inds)
            b_inds = torch.randperm(args.batch_size)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        if args.save_model and iteration % args.env_args.export_interval == 0:
            model_path = args.save_dir / f"iter_{iteration}.pth"
            torch.save(agent.state_dict(), model_path)
            print(f"model saved to {model_path}")

    env.close()
    writer.close()

