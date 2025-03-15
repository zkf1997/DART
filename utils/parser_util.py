from argparse import ArgumentParser
import argparse
import os
import json


def parse_and_load_from_model(parser):
    # args according to the loaded model
    # do not try to specify them from cmd line since they will be overwritten
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    args = parser.parse_args()
    args_to_overwrite = []
    for group_name in ['dataset', 'model', 'diffusion']:
        args_to_overwrite += get_args_per_group_name(parser, args, group_name)
    args_to_overwrite = args_to_overwrite + ['history_length', 'future_length', 'feature_dim']

    # load args from model
    model_path = get_model_path_from_args()
    args_path = os.path.join(os.path.dirname(model_path), 'args.json')
    assert os.path.exists(args_path), 'Arguments json file was not found!'
    with open(args_path, 'r') as fr:
        model_args = json.load(fr)

    for a in args_to_overwrite:
        if a in ['respacing']:
            print(f'skip loading arg:{a} from model')
            continue
        if a in model_args.keys():
            setattr(args, a, model_args[a])

        elif 'cond_mode' in model_args: # backward compitability
            unconstrained = (model_args['cond_mode'] == 'no_cond')
            setattr(args, 'unconstrained', unconstrained)

        else:
            print('Warning: was not able to load [{}], using default value [{}] instead.'.format(a, args.__dict__[a]))

    if args.cond_mask_prob == 0:
        args.guidance_param = 1
    return args

def load_args_from_model(model_path):
    args_path = os.path.join(os.path.dirname(model_path), 'args.json')
    assert os.path.exists(args_path), 'Arguments json file was not found!'
    with open(args_path, 'r') as fr:
        model_args = json.load(fr)
    if 'output_cumsum' not in model_args:
        model_args['output_cumsum'] = 0
    if 'lambda_dct' not in model_args:
        model_args['lambda_dct'] = 0.0
    if 'lambda_jerk' not in model_args:
        model_args['lambda_jerk'] = 0.0
    if 'lambda_first_joints_delta' not in model_args:
        model_args['lambda_first_joints_delta'] = 0.0
    if 'lambda_first_transl_delta' not in model_args:
        model_args['lambda_first_transl_delta'] = 0.0
    if 'lambda_first_orient_delta' not in model_args:
        model_args['lambda_first_orient_delta'] = 0.0
    if 'lambda_rel_joints_delta' not in model_args:
        model_args['lambda_rel_joints_delta'] = 0.0
    if 'lambda_rel_transl_delta' not in model_args:
        model_args['lambda_rel_transl_delta'] = 0.0
    if 'lambda_rel_orient_delta' not in model_args:
        model_args['lambda_rel_orient_delta'] = 0.0
    if 'mask_thresh_step' not in model_args:
        model_args['mask_thresh_step'] = 1.0
    model_args['model_path'] = model_path
    return model_args

def get_args_per_group_name(parser, args, group_name):
    for group in parser._action_groups:
        if group.title == group_name:
            group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
            return list(argparse.Namespace(**group_dict).__dict__.keys())
    return ValueError('group_name was not found.')

def get_model_path_from_args():
    try:
        dummy_parser = ArgumentParser()
        dummy_parser.add_argument('model_path')
        dummy_args, _ = dummy_parser.parse_known_args()
        return dummy_args.model_path
    except:
        raise ValueError('model_path argument must be specified.')


def add_base_options(parser):
    group = parser.add_argument_group('base')
    group.add_argument("--cuda", default=True, type=bool, help="Use cuda device, otherwise use CPU.")
    group.add_argument("--device", default=0, type=int, help="Device id to use.")
    group.add_argument("--seed", default=10, type=int, help="For fixing random seed.")
    group.add_argument("--batch_size", default=64, type=int, help="Batch size during training.")


def add_diffusion_options(parser):
    group = parser.add_argument_group('diffusion')
    group.add_argument("--noise_schedule", default='cosine', choices=['linear', 'cosine'], type=str,
                       help="Noise schedule type")
    group.add_argument("--diffusion_steps", default=1000, type=int,
                       help="Number of diffusion steps (denoted T in the paper)")
    group.add_argument("--respacing", default='', type=str,
                       help="empty for no respacing, or ddimN for N steps DDIM solver. ")
    group.add_argument("--sigma_small", default=True, type=bool, help="Use smaller sigma values.")
    group.add_argument("--ignore_history", default=0, type=int)


def add_model_options(parser):
    group = parser.add_argument_group('model')
    group.add_argument("--arch", default='trans_enc',
                       choices=['trans_enc', 'trans_dec', 'gru'], type=str,
                       help="Architecture types as reported in the paper.")
    group.add_argument("--emb_trans_dec", default=False, type=bool,
                       help="For trans_dec architecture only, if true, will inject condition as a class token"
                            " (in addition to cross-attention).")
    group.add_argument("--layers", default=8, type=int,
                       help="Number of layers.")
    group.add_argument("--latent_dim", default=512, type=int,
                       help="Transformer/GRU width.")
    group.add_argument("--ff_size", default=1024, type=int)
    group.add_argument("--cond_mask_prob", default=.1, type=float,
                       help="The probability of masking the condition during training."
                            " For classifier-free guidance learning.")
    group.add_argument("--lambda_rcxyz", default=0.0, type=float, help="Joint positions loss.")
    group.add_argument("--lambda_vel", default=0.0, type=float, help="Joint velocity loss.")
    group.add_argument("--lambda_fc", default=0.0, type=float, help="Foot contact loss.")
    group.add_argument("--lambda_smpl_joints", default=0.0, type=float)
    group.add_argument("--lambda_smpl_vertices", default=0.0, type=float)
    group.add_argument("--lambda_joints_consistency", default=0.0, type=float)
    group.add_argument("--lambda_joints_delta", default=0.0, type=float)
    group.add_argument("--lambda_transl_delta", default=0.0, type=float)
    group.add_argument("--lambda_orient_delta", default=0.0, type=float)
    group.add_argument("--lambda_first_joints_delta", default=0.0, type=float)
    group.add_argument("--lambda_first_transl_delta", default=0.0, type=float)
    group.add_argument("--lambda_first_orient_delta", default=0.0, type=float)
    group.add_argument("--lambda_rel_joints_delta", default=0.0, type=float)
    group.add_argument("--lambda_rel_transl_delta", default=0.0, type=float)
    group.add_argument("--lambda_rel_orient_delta", default=0.0, type=float)
    group.add_argument("--lambda_delta_norm", default=0.0, type=float)
    group.add_argument("--lambda_smooth", default=0.0, type=float)
    group.add_argument("--lambda_dct", default=0.0, type=float)
    group.add_argument("--lambda_jerk", default=0.0, type=float)
    group.add_argument("--mask_thresh_step", default=1.0, type=float)
    group.add_argument("--unconstrained", action='store_true',
                       help="Model is trained unconditionally. That is, it is constrained by neither text nor action. "
                            "Currently tested on HumanAct12 only.")
    group.add_argument("--output_cumsum", default=0, type=int)



def add_data_options(parser):
    group = parser.add_argument_group('dataset')
    group.add_argument("--dataset", default='humanml', type=str,
                       help="Dataset name (choose from list).")
    group.add_argument("--data_dir", default="", type=str,
                       help="If empty, will use defaults according to the specified dataset.")
    group.add_argument("--cfg_path", type=str, default='./config_files/config_hydra/motion_primitive/mp_2_8.yaml')
    group.add_argument("--enforce_gender", type=str, default=None)
    group.add_argument("--enforce_zero_beta", type=int, default=0)
    group.add_argument("--body_type", type=str, default='smplx')
    group.add_argument("--seed_only", type=int, default=0)


def add_training_options(parser):
    group = parser.add_argument_group('training')
    group.add_argument("--save_dir", required=True, type=str,
                       help="Path to save checkpoints and results.")
    group.add_argument("--overwrite", action='store_true',
                       help="If True, will enable to use an already existing save_dir.")
    group.add_argument("--train_platform_type", default='TensorboardPlatform', choices=['NoPlatform', 'ClearmlPlatform', 'TensorboardPlatform'], type=str,
                       help="Choose platform to log results. NoPlatform means no logging.")
    group.add_argument("--lr", default=1e-4, type=float, help="Learning rate.")
    group.add_argument("--weight_decay", default=0.0, type=float, help="Optimizer weight decay.")
    group.add_argument("--lr_anneal_steps", default=0, type=int, help="Number of learning rate anneal steps.")
    group.add_argument("--eval_batch_size", default=32, type=int,
                       help="Batch size during evaluation loop. Do not change this unless you know what you are doing. "
                            "T2m precision calculation is based on fixed batch size 32.")
    group.add_argument("--eval_split", default='test', choices=['val', 'test'], type=str,
                       help="Which split to evaluate on during training.")
    group.add_argument("--eval_during_training", action='store_true',
                       help="If True, will run evaluation during training.")
    group.add_argument("--eval_rep_times", default=3, type=int,
                       help="Number of repetitions for evaluation loop during training.")
    group.add_argument("--eval_num_samples", default=1_000, type=int,
                       help="If -1, will use all samples in the specified split.")
    group.add_argument("--log_interval", default=1_000, type=int,
                       help="Log losses each N steps")
    group.add_argument("--save_interval", default=100_000, type=int,
                       help="Save checkpoints and run evaluation each N steps")
    group.add_argument("--num_steps", default=600_000, type=int,
                       help="Training will stop after the specified number of steps.")
    group.add_argument("--stage1_steps", default=100_000, type=int,
                       help="train with GT data")
    group.add_argument("--stage2_steps", default=100_000, type=int,
                       help="schedule train with random mixture of GT data and generated data rollout")
    group.add_argument("--stage3_steps", default=300_000, type=int,
                       help="train with generated data rollout")
    group.add_argument("--num_frames", default=60, type=int,
                       help="Limit for the maximal number of frames. In HumanML3D and KIT this field is ignored.")
    group.add_argument("--resume_checkpoint", default="", type=str,
                       help="If not empty, will start from the specified checkpoint (path to model###.pt file).")
    group.add_argument("--load_opt", default=1, type=int)
    group.add_argument("--use_fp16", action='store_true', default=False)
    group.add_argument("--avg_model_beta", default=0, type=float,
                       help="If > 0, will use the moving average model with the specified beta value.")
    group.add_argument("--grad_clip", default=0, type=float,
                       help="If > 0, clip grad norm.")
    group.add_argument("--ddim_rec", default=0, type=int)
    group.add_argument("--rollout_type", default='ddpm_rec', type=str)
    group.add_argument("--weight_scheme", default='uniform', type=str)
    group.add_argument("--text_tolerance", default=0.0, type=float)
    group.add_argument("--prob_static", default=0, type=float,
                       help="If > 0, randomly train on static sequence.")
    group.add_argument("--use_predicted_joints", default=0, type=int)


def add_sampling_options(parser):
    group = parser.add_argument_group('sampling')
    group.add_argument("--model_path", required=True, type=str,
                       help="Path to model####.pt file to be sampled.")
    group.add_argument("--output_dir", default='', type=str,
                       help="Path to results dir (auto created by the script). "
                            "If empty, will create dir in parallel to checkpoint.")
    group.add_argument("--num_samples", default=1, type=int,
                       help="Maximal number of prompts to sample, "
                            "if loading dataset from file, this field will be ignored.")
    group.add_argument("--num_repetitions", default=3, type=int,
                       help="Number of repetitions, per sample (text prompt/action)")
    group.add_argument("--guidance_param", default=2.5, type=float,
                       help="For classifier-free sampling - specifies the s parameter, as defined in the paper.")
    group.add_argument("--export_smpl", default=0, type=int)


def add_generate_options(parser):
    group = parser.add_argument_group('generate')
    group.add_argument("--motion_length", default=6.0, type=float,
                       help="The length of the sampled motion [in seconds]. "
                            "Maximum is 9.8 for HumanML3D (text-to-motion), and 2.0 for HumanAct12 (action-to-motion)")
    group.add_argument("--input_text", default='', type=str,
                       help="Path to a text file lists text prompts to be synthesized. If empty, will take text prompts from dataset.")
    group.add_argument("--action_file", default='', type=str,
                       help="Path to a text file that lists names of actions to be synthesized. Names must be a subset of dataset/uestc/info/action_classes.txt if sampling from uestc, "
                            "or a subset of [warm_up,walk,run,jump,drink,lift_dumbbell,sit,eat,turn steering wheel,phone,boxing,throw] if sampling from humanact12. "
                            "If no file is specified, will take action names from dataset.")
    group.add_argument("--text_prompt", default='', type=str,
                       help="A text prompt to be generated. If empty, will take text prompts from dataset.")
    group.add_argument("--action_name", default='', type=str,
                       help="An action name to be generated. If empty, will take text prompts from dataset.")
    group.add_argument("--num_rollout", default=1, type=int,
                       help="Number of motion segments. ")
    group.add_argument("--smooth", default=0, type=int,
                       help="whether smooth generated motion by blending")
    group.add_argument("--zero_noise", default=0, type=int)
    group.add_argument("--use_predicted_joints", default=0, type=int)

def add_optim_options(parser):
    group = parser.add_argument_group('optim')
    group.add_argument('--optim_lr', type=float, default=0.01)
    group.add_argument('--optim_steps', type=int, default=300)
    group.add_argument('--optim_unit_grad', type=int, default=0)
    group.add_argument('--optim_anneal_lr', type=int, default=0)

    group.add_argument('--optim_input', type=str)

    group.add_argument('--optim_goal', type=str)
    group.add_argument('--optim_mode', type=str)


    parser.add_argument('--interaction_cfg', type=str, default=None)
    parser.add_argument('--weight_collision', type=float, default=0)
    parser.add_argument('--contact_thresh', type=float, default=0.03)
    parser.add_argument('--weight_contact', type=float, default=0)
    parser.add_argument('--weight_skate', type=float, default=0)
    parser.add_argument('--weight_jerk', type=float, default=0)
    parser.add_argument('--load_cache', type=int, default=0)

def add_edit_options(parser):
    group = parser.add_argument_group('edit')
    group.add_argument("--edit_mode", default='in_between', choices=['in_between', 'upper_body'], type=str,
                       help="Defines which parts of the input motion will be edited.\n"
                            "(1) in_between - suffix and prefix motion taken from input motion, "
                            "middle motion is generated.\n"
                            "(2) upper_body - lower body joints taken from input motion, "
                            "upper body is generated.")
    group.add_argument("--text_condition", default='', type=str,
                       help="Editing will be conditioned on this text prompt. "
                            "If empty, will perform unconditioned editing.")
    group.add_argument("--num_rollout", default=1, type=int,
                       help="Number of motion segments. ")
    group.add_argument("--prefix_end", default=0.25, type=float,
                       help="For in_between editing - Defines the end of input prefix (ratio from all frames).")
    group.add_argument("--suffix_start", default=0.75, type=float,
                       help="For in_between editing - Defines the start of input suffix (ratio from all frames).")


def add_evaluation_options(parser):
    group = parser.add_argument_group('eval')
    group.add_argument("--model_path", required=True, type=str,
                       help="Path to model####.pt file to be sampled.")
    group.add_argument("--eval_mode", default='wo_mm', choices=['wo_mm', 'mm_short', 'debug', 'full'], type=str,
                       help="wo_mm (t2m only) - 20 repetitions without multi-modality metric; "
                            "mm_short (t2m only) - 5 repetitions with multi-modality metric; "
                            "debug - short run, less accurate results."
                            "full (a2m only) - 20 repetitions.")
    group.add_argument("--guidance_param", default=2.5, type=float,
                       help="For classifier-free sampling - specifies the s parameter, as defined in the paper.")


def get_cond_mode(args):
    if args.unconstrained:
        cond_mode = 'no_cond'
    elif args.dataset in ['kit', 'humanml']:
        cond_mode = 'text'
    else:
        cond_mode = 'action'
    return cond_mode


def train_args():
    parser = ArgumentParser()
    add_base_options(parser)
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    add_training_options(parser)
    return parser.parse_args()


def generate_args():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_sampling_options(parser)
    add_generate_options(parser)
    args = parse_and_load_from_model(parser)
    # cond_mode = get_cond_mode(args)
    #
    # if (args.input_text or args.text_prompt) and cond_mode != 'text':
    #     raise Exception('Arguments input_text and text_prompt should not be used for an action condition. Please use action_file or action_name.')
    # elif (args.action_file or args.action_name) and cond_mode != 'action':
    #     raise Exception('Arguments action_file and action_name should not be used for a text condition. Please use input_text or text_prompt.')

    return args

def optim_args():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_sampling_options(parser)
    add_generate_options(parser)
    add_optim_options(parser)
    args = parse_and_load_from_model(parser)

    return args

def flowmdm_args():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_sampling_options(parser)
    add_generate_options(parser)
    parser.add_argument('--flowmdm_dir', type=str, help='Path to the flowmdm results directory.')
    # parser.add_argument('ours_dir', type=str, help='')
    args = parse_and_load_from_model(parser)
    # cond_mode = get_cond_mode(args)
    #
    # if (args.input_text or args.text_prompt) and cond_mode != 'text':
    #     raise Exception('Arguments input_text and text_prompt should not be used for an action condition. Please use action_file or action_name.')
    # elif (args.action_file or args.action_name) and cond_mode != 'action':
    #     raise Exception('Arguments action_file and action_name should not be used for a text condition. Please use input_text or text_prompt.')

    return args

def edit_args():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_sampling_options(parser)
    add_edit_options(parser)
    return parse_and_load_from_model(parser)


def evaluation_parser():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_evaluation_options(parser)
    return parse_and_load_from_model(parser)