python -m control.test_reach_location_mld --env_args.obs_goal_angle_clip 60.0 --env_args.obs_goal_dist_clip 5.0 --env_args.num_envs 4 --env_args.num_steps 256  --test_goal_path './data/test_locomotion/demo_walk.json' --init_data_path './data/stand.pkl' --resume_checkpoint './policy_train/reach_location_mld/fixtext_repeat_floor100_hop10_skate100/iter_2000.pth'
