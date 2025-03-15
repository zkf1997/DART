from model.mdm import MDM
from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps
from utils.parser_util import get_cond_mode
import torch


def load_model_wo_clip(model, state_dict):
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print('unexpected_keys:', unexpected_keys)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])


def create_model_and_diffusion(args, data):
    model = MDM(**get_model_args(args, data))
    diffusion = create_gaussian_diffusion(args)
    return model, diffusion

def create_model_and_diffusion_ddim(args, data):
    model = MDM(**get_model_args(args, data))
    diffusion = create_gaussian_diffusion(args, enable_ddim=False)
    diffusion_ddim = create_gaussian_diffusion(args, enable_ddim=True)
    return model, diffusion, diffusion_ddim

def get_model_args(args, data=None):

    # default args
    clip_version = 'ViT-B/32'
    action_emb = 'tensor'
    # cond_mode = get_cond_mode(args)
    cond_mode = 'text'
    # if hasattr(data.dataset, 'num_actions'):
    #     num_actions = data.dataset.num_actions
    # else:
    #     num_actions = 1
    num_actions = 1

    # SMPL defaults
    data_rep = 'loc_rot_delta'
    njoints = args.feature_dim  # actually the dimension of the feature for one frame containing all joints
    nfeats = 1  # dummy dimension
    ff_size = getattr(args, 'ff_size', 1024)  # backward compatibility

    return {'modeltype': '', 'njoints': njoints, 'nfeats': nfeats, 'num_actions': num_actions,
            'translation': True, 'pose_rep': 'rot6d', 'glob': True, 'glob_rot': True,
            'latent_dim': args.latent_dim, 'ff_size': ff_size, 'num_layers': args.layers, 'num_heads': 4,
            'dropout': 0.1, 'activation': "gelu", 'data_rep': data_rep, 'cond_mode': cond_mode,
            'cond_mask_prob': args.cond_mask_prob, 'action_emb': action_emb, 'arch': args.arch,
            'emb_trans_dec': args.emb_trans_dec, 'clip_version': clip_version, 'dataset': args.dataset,
            'output_cumsum': args.output_cumsum,
            }


def create_gaussian_diffusion(args, enable_ddim=True):
    # default params
    predict_xstart = True  # we always predict x_start (a.k.a. x0), that's our deal!
    steps = args.diffusion_steps
    scale_beta = 1.  # no scaling
    timestep_respacing = args.respacing if enable_ddim else ''  # can be used for ddim sampling, we don't use it.
    learn_sigma = False
    rescale_timesteps = False

    betas = gd.get_named_beta_schedule(args.noise_schedule, steps, scale_beta)
    loss_type = gd.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not args.sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        lambda_vel=args.lambda_vel,
        lambda_rcxyz=args.lambda_rcxyz,
        lambda_fc=args.lambda_fc,
        lambda_smpl_joints=args.lambda_smpl_joints,
        lambda_smpl_vertices=args.lambda_smpl_vertices,
        lambda_joints_consistency=args.lambda_joints_consistency,
        lambda_joints_delta=args.lambda_joints_delta,
        lambda_transl_delta=args.lambda_transl_delta,
        lambda_orient_delta=args.lambda_orient_delta,
        lambda_first_joints_delta=args.lambda_first_joints_delta,
        lambda_first_transl_delta=args.lambda_first_transl_delta,
        lambda_first_orient_delta=args.lambda_first_orient_delta,
        lambda_rel_joints_delta=args.lambda_rel_joints_delta,
        lambda_rel_transl_delta=args.lambda_rel_transl_delta,
        lambda_rel_orient_delta=args.lambda_rel_orient_delta,
        lambda_delta_norm=args.lambda_delta_norm,
        lambda_smooth=args.lambda_smooth,
        lambda_dct=args.lambda_dct,
        lambda_jerk=args.lambda_jerk,
        ignore_history=args.ignore_history,
        mask_thresh_step=args.mask_thresh_step,
    )

def load_saved_model(model, model_path, use_avg: bool=True):  # use_avg_model
    state_dict = torch.load(model_path, map_location='cpu')
    # Use average model when possible
    if use_avg and 'model_avg' in state_dict.keys():
    # if use_avg_model:
        print('loading avg model')
        state_dict = state_dict['model_avg']
    else:
        if 'model' in state_dict:
            print('loading model without avg')
            state_dict = state_dict['model']
        else:
            print('checkpoint has no avg model')
    load_model_wo_clip(model, state_dict)
    return model