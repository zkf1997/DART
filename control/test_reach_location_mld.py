import pickle
import json
from control.train_reach_location_mld import *

def load_checkpoint(test_args):
    checkpoint_dir = Path(test_args.resume_checkpoint).parent
    arg_path = checkpoint_dir / "args.yaml"
    with open(arg_path, "r") as f:
        args = tyro.extras.from_yaml(ReachLocationArgs, yaml.safe_load(f))

    args.env_id = args.env_args.env_id
    args.num_envs = args.env_args.num_envs = test_args.env_args.num_envs
    args.num_steps = args.env_args.num_steps = test_args.env_args.num_steps
    args.env_args.export_interval = 1
    args.env_args.enable_export = test_args.env_args.enable_export
    # enforce obs goal clipping
    args.env_args.obs_goal_angle_clip = test_args.env_args.obs_goal_angle_clip
    args.env_args.obs_goal_dist_clip = test_args.env_args.obs_goal_dist_clip
    args.init_data_path = test_args.init_data_path

    # args.save_dir = Path(f"./policy_train/{args.env_id}/{args.exp_name}")
    args.save_dir = checkpoint_dir
    args.save_dir.mkdir(parents=True, exist_ok=True)

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_default_dtype(torch.float32)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # load motion model
    denoiser_args, denoiser_model, vae_args, vae_model, diffusion, dataset = load_motion_model(args)

    # create env
    args.env_args.export_interval = 1
    args.env_args.save_dir = args.save_dir / 'env_test'
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

    agent.load_state_dict(torch.load(test_args.resume_checkpoint))

    return args, env, agent

if __name__ == "__main__":
    test_args = tyro.cli(ReachLocationArgs)
    args, env, agent = load_checkpoint(test_args)
    # print('batch size:', env.batch_size)
    with open(test_args.test_goal_path, 'r') as f:
        test_goal_data = json.load(f)
    for iter in range(len(test_goal_data)):
        print(f"Testing goal sequence {iter}")
        env.global_iteration = iter
        goal_location_list = test_goal_data[iter]['goal_location']
        goal_location_list.append([0, 0, 0])
        goal_text_list = test_goal_data[iter]['goal_text']
        goal_text_list.append('walk')  # add a dummy goal to determine the end of the sequence
        goal_location_list = torch.tensor(goal_location_list, dtype=torch.float32, device=env.device)
        goal_text_list = np.array(goal_text_list)  # convert list of string to array of string is ok, but if try to assign new value to elements, it cannot contain space symbol
        # print('goal_text_list', goal_text_list)
        next_goal_idx = np.zeros(args.num_envs, dtype=int)
        assert len(goal_location_list) == len(goal_text_list)
        num_goals = len(goal_location_list)

        next_obs, _ = env.reset(goal_location=goal_location_list[next_goal_idx], goal_texts=goal_text_list[next_goal_idx])
        next_goal_idx = (next_goal_idx + 1).clip(max=num_goals-1)
        finish_list = []
        for step in tqdm(range(0, args.num_steps)):
            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
            next_obs, reward, success, terminations, truncations, infos = env.step(action,
                                                                                   next_goal_location=goal_location_list[next_goal_idx],
                                                                                   next_goal_texts=goal_text_list[next_goal_idx],
                                                                                   reset_text=True)
            if success.any():
                succ_idx = torch.nonzero(success, as_tuple=True)[0].cpu().numpy()
                for idx in succ_idx:
                    if idx not in finish_list and next_goal_idx[idx] == num_goals - 1:
                        # print(f"Env {idx} reached the last goal")
                        finish_list.append(idx)
                        sequence = deepcopy(env.sequences[idx])
                        save_path = env.args.save_dir / f'{Path(test_args.test_goal_path).stem}_path{iter}' / f'{idx}.pkl'
                        save_path.parent.mkdir(parents=True, exist_ok=True)
                        sequence['texts'] = sequence['goal_texts_list']
                        sequence['text_idx'] = sequence['goal_texts_idx']  # for backward compatibility
                        sequence['action'] = torch.stack(sequence['action'], dim=0)  # [num_rollout, future_len, D]
                        sequence['obs'] = torch.stack(sequence['obs'], dim=0)
                        sequence = tensor_dict_to_device(sequence, 'cpu')
                        with open(save_path, 'wb') as f:
                            pickle.dump(sequence, f)
                next_goal_idx[succ_idx] = (next_goal_idx[succ_idx] + 1).clip(max=num_goals-1)
            if len(finish_list) == args.num_envs:
                print(f"Finished all environments at step {step}")
                break

    print(f"Save dir: {args.env_args.save_dir.resolve()}")


