# DartControl
## A Diffusion-Based Autoregressive Motion Model for Real-Time Text-Driven Motion Control (ICLR 2025, Spotlight)

### [[website](https://zkf1997.github.io/DART/)] | [[paper](https://arxiv.org/abs/2410.05260)] 


https://github.com/user-attachments/assets/b26e95e7-4af0-4548-bdca-8f361594951c



# Updates
This repository is under construction and the documentations for the following for will be updated. If you encounter any problems, please do not hesitate to contact us.

- [x] Setup, generation demos, and visualization
- [x] Data preparation and training
- [x] Evaluation

# Getting Started

## Environment Setup
Setup conda env:
```
conda env create -f environment.yml
conda activate DART
```
Tested system:

Our experiments and performance profiling are conducted on a workstation with single RTX 4090
GPU, intel i7-13700K CPU, 64GiB memory. The workstation runs with Ubuntu 22.04.4 LTS system.

## Data and Model Checkpoints
* Please download this [google drive link](https://drive.google.com/drive/folders/1vJg3GFVPT6kr6cA0HrQGmiAEBE2dkaps?usp=drive_link) containing model checkpoints and necessary data, extract and merge it to the project folder.

* Please download the following data from the respective websites and organize as shown below:
  * [SMPL-X body model](https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=smplx_lockedhead_20230207.zip)
  * [SMPL-H body model](https://download.is.tue.mpg.de/download.php?domain=mano&resume=1&sfile=smplh.tar.xz)
    * [How to convert to PKL](https://github.com/vchoutas/smplx/blob/main/tools/README.md)
  * [AMASS](https://amass.is.tue.mpg.de/) (Only required for training, please down the gender-specific data for SMPL-H and SMPL-X)
  * [BABEL](https://download.is.tue.mpg.de/download.php?domain=teach&resume=1&sfile=babel-data/babel-teach.zip) (Only required for training)
  * [HumanML3D](https://github.com/EricGuo5513/HumanML3D)(Only required for training)

  
  * <details>

    <summary><b>Project folder structure of separately downloaded data:</b></summary>

    ```
      ./
      ├── data
      │   ├── smplx_lockedhead_20230207
      │   │   └── models_lockedhead
      │   │       ├── smplh
      │   │       │   ├── SMPLH_FEMALE.pkl
      │   │       │   └── SMPLH_MALE.pkl
      │   │       └── smplx
      │   │           ├── SMPLX_FEMALE.npz
      │   │           ├── SMPLX_MALE.npz
      │   │           └── SMPLX_NEUTRAL.npz
      │   ├── amass
      │   │   ├──  babel-teach
      │   │   │        ├── train.json
      │   │   │        └── val.json
      │   │   ├──  smplh_g
      │   │   │        ├── ACCAD
      │   │   │        ├── BioMotionLab_NTroje
      │   │   │        ├── BMLhandball
      │   │   │        ├── BMLmovi
      │   │   │        ├── CMU
      │   │   │        ├── CNRS
      │   │   │        ├── DanceDB
      │   │   │        ├── DFaust_67
      │   │   │        ├── EKUT
      │   │   │        ├── Eyes_Japan_Dataset
      │   │   │        ├── GRAB
      │   │   │        ├── HUMAN4D
      │   │   │        ├── HumanEva
      │   │   │        ├── KIT
      │   │   │        ├── MPI_HDM05
      │   │   │        ├── MPI_Limits
      │   │   │        ├── MPI_mosh
      │   │   │        ├── SFU
      │   │   │        ├── SOMA
      │   │   │        ├── SSM_synced
      │   │   │        ├── TCD_handMocap
      │   │   │        ├── TotalCapture
      │   │   │        ├── Transitions_mocap
      │   │   │        └── WEIZMANN
      │   │   └──  smplx_g
      │   │   │        ├── ACCAD
      │   │   │        ├── BMLmovi
      │   │   │        ├── BMLrub
      │   │   │        ├── CMU
      │   │   │        ├── CNRS
      │   │   │        ├── DanceDB
      │   │   │        ├── DFaust
      │   │   │        ├── EKUT
      │   │   │        ├── EyesJapanDataset
      │   │   │        ├── GRAB
      │   │   │        ├── HDM05
      │   │   │        ├── HUMAN4D
      │   │   │        ├── HumanEva
      │   │   │        ├── KIT
      │   │   │        ├── MoSh
      │   │   │        ├── PosePrior
      │   │   │        ├── SFU
      │   │   │        ├── SOMA
      │   │   │        ├── SSM
      │   │   │        ├── TCDHands
      │   │   │        ├── TotalCapture
      │   │   │        ├── Transitions
      │   │   │        └── WEIZMANN
      │   ├── HumanML3D
      │   │   ├── HumanML3D
      │   │   │   ├──...
      │   │   └── index.csv
    ```
    </details>

## Visualization 

### Pyrender Viewer
* We use `pyrender` for interactive visualization of generated motions by default. Please refer to [pyrender viewer](https://pyrender.readthedocs.io/en/latest/generated/pyrender.viewer.Viewer.html) for the usage of the interactive viewer, such as rotating, panning, and zooming.
* The [visualization script](./visualize/vis_seq.py) can render a generated sequence by specifying the `seq_path` argument. It also supports several optional functions, such as multi-sequence visualization, interactive play with frame forward/backward control using keyboards, and automatic body-following camera. More details of the configurable arguments can be found in the [vis script](https://github.com/zkf1997/DART/blob/7c1c922ae08f98b507eb7bdcc2e8029ed82e3b64/visualize/vis_seq.py#L375).
* The script can be slow when visualizing multiple humans together. You can choose to visualize only one human at a time by setting `--max_seq 1` in the command line, or use the blender visualization described below which is several times more efficient.

### Blender Visualization
* We also support exporting the generated motions as `npz` files and visualize in [Blender](https://www.blender.org/) for advanced rendering. To import one motion sequence into blender, please first install the [SMPL-X Blender Add-on](https://gitlab.tuebingen.mpg.de/jtesch/smplx_blender_addon#installation), and use the "add animation" feature as shown in this video. You can use the space key to start/stop playing animation in Blender.
 
  
  <details>

   <summary><b>Demonstration of importing motion into Blender:</b></summary>

    https://github.com/user-attachments/assets/a15fc9d6-507e-4521-aa3f-64b2db8c0252

  </details>


# Motion Generation Demos
We offer a range of motion generation demos, including online text-conditioned motion generation and applications with spatial constraints and goals. 
These applications include motion in-betweening, waypoint goal reaching, and human-scene interaction generation.

## Interactive Online Text-Conditioned Motion Generation
```
source ./demos/run_demo.sh
```
This will open an interactive viewer and a command-line interface for text input. You can input text prompts and the model will generate the corresponding motion sequence on the fly.
The model is trained on the BABEL dataset, which describes motions using verbs or phrases. The action coverage in the dataset can be found [here](https://babel.is.tue.mpg.de/explore.html). 
A demonstration video is shown below:

https://github.com/user-attachments/assets/ce84ab14-4b3e-42bd-8a8b-db721ee108e3



## Headless Text-Conditioned Motion Composition 
We offer a headless script for text-conditioned motion composition, enabling users to generate motions from a timeline of actions defined via text prompts.
The text prompt follows the format:  
**`action_1*num_1,action_2*num_2,...,action_n*num_n`**  
where:  
- **`action_x`**: A text description of the action (e.g., "walk forward," "turn left").  
- **`num_x`**: The duration of the action, measured in **motion primitives** (each primitive corresponds to 8 frames).  

You can run the following command to generate example motions of walking in circles:
```
source ./demos/rollout.sh
```
We also provide some additional **example text prompts** which are commented out in this [file](./demos/rollout.sh).The output directory of generated motions will be displayed in the command line. The generated motions can be visualized using the [pyrender viewer](#pyrender-viewer) as follows:
```
python -m visualize.vis_seq --add_floor 1 --translate_body 1 --seq_path './mld_denoiser/mld_fps_clip_repeat_euler/checkpoint_300000/rollout/walk_in_circles*20_guidance5.0_seed0/*.pkl' 
```
We refer to the [vis script](https://github.com/zkf1997/DART/blob/7c1c922ae08f98b507eb7bdcc2e8029ed82e3b64/visualize/vis_seq.py#L375) for detailed visualization configuration. The output directory also contains the exported motion sequences as `npz` files for [Blender visualization](#blender-visualization).
 
## Text-Conditioned Motion in-betweening
We provide a script to generate motions between two keyframes conditioned on text prompts.
The keyframes and the duration of inbetweening is specified using a SMPL parameter sequence via `--optim_input` while the text prompt is specified using `--text_prompt`.
The script offers two modes, selectable via the `--seed_type` argument: `repeat` and `history`. These modes are designed to handle scenarios where either a single start keyframe or multiple start keyframes are provided. When multiple start keyframes are available, we aim to ensure velocity consistency in addition to maintaining initial location consistency.
* Repeat mode: The first frame of the input sequence is the start keyframe and the last frame is the goal keyframe, the rest frames are the repeat padding of the first frame. The output sequence length equals to the input sequence length.
* History mode: The first three frames of the input sequence serve as start keyframes to provide velocity context, and the last frame is the goal keyframe. The remaining frames can be filled using zero-padding or repeat-padding.

We show an example of in-betweening "pace in circles" between two keyframes:
```
source ./demos/inbetween_babel.sh
```
The generated sequences can be visualized using the commands below.
The white bodies represent the keyframes for reference, while the colored bodies depict the generated results. 
To better assess goal keyframe reaching accuracy, you can enable **interactive play mode** by adding `--interactive 1` and pressing `a` to display only the last frame.
* Repeat mode:
  ```
  python -m visualize.vis_seq --add_floor 1 --body_type smplx --seq_path './mld_denoiser/mld_fps_clip_repeat_euler/checkpoint_300000/optim/inbetween/repeatseed/scale0.1_floor0.0_jerk0.0_use_pred_joints_ddim10_pace_in_circles*15_guidance5.0_seed0/*.pkl'
  ```
  
* History mode:
  ```
  python -m visualize.vis_seq --add_floor 1 --body_type smplx --seq_path './mld_denoiser/mld_fps_clip_repeat_euler/checkpoint_300000/optim/inbetween/historyseed/scale0.1_floor0.0_jerk0.0_use_pred_joints_ddim10_pace_in_circles*15_guidance5.0_seed0/*.pkl'
  ``` 

You can easily test custom in-betweening by customizing `--optim_input` and `--text_prompt`. The input SMPL sequence should include the attributes `gender, betas, transl, global_orient, body_pose`. Example sequences can be found [here](./data/inbetween/pace_in_circles).

 
<details>  
<summary><b>Using model trained on the HML3D dataset:</b></summary>
In addition to inbetweening with the model trained on the BABEL dataset (as demonstrated above), we also provide a script for inbetweening using a model trained on the HML3D dataset [here](./demos/inbetween_hml.sh). While you can generally use the HML3D-trained model for **all optimization-based demos** below, please note the following:

- The text prompt style in HML3D differs from BABEL.
- HML3D assumes **20 fps** motions, whereas BABEL uses **30 fps**.
- When visualizing HML3D results with the visualization script, please add `--body_type smplh` to specify the body type, as HML3D utilizes **SMPL-H** bodies.
</details>











## Human-Scene Interaction Synthesis
We provide a script to generate human-scene interaction motions.
Given an input 3D scene and the text prompts specifying the actions and durations, we control the human to reach the goal joint location starting from an initial pose while adhering to the scene contact and collision constraints.
We show two examples of climbing downstairs and sitting to a chair in the demo below:
```
source ./demos/scene.sh
```
The generated sequences can be visualized using:
```
python -m visualize.vis_seq --add_floor 0 --seq_path './mld_denoiser/mld_fps_clip_repeat_euler/checkpoint_300000/optim/sit_use_pred_joints_ddim10_guidance5.0_seed0_contact0.1_thresh0.0_collision0.1_jerk0.1/sample_*.pkl'
```
```
python -m visualize.vis_seq --add_floor 0 --seq_path './mld_denoiser/mld_fps_clip_repeat_euler/checkpoint_300000/optim/climb_down_use_pred_joints_ddim10_guidance5.0_seed0_contact0.1_thresh0.0_collision0.1_jerk0.1/sample_*.pkl'
```

To use a custom 3D scene, you need to first calculate the scene SDF for evaluating human-scene collision and contact constraints.
Please ensure the 3D scene is z-up and the floor plane has zero height.
We use [mesh2sdf](https://github.com/wang-ps/mesh2sdf) for SDF calculation, as shown in [this script](./scenes/test_sdf.py).
Example configuration files for an interaction sequence can be found [here](./data/optim_interaction). We currently initialize the human using a standing pose, with its location and orientation determined by the pelvis, left hip and right hip location specified using `init_joints`.
The goal joint locations are specified using `goal_joints`. The current [script](./mld/optim_scene_mld.py) only use pelvis as the goal joint, you can modify the goal joints to be another joint or multiple joints.
You may also tune the optimization parameters to modulate the generation, such as increasing the learning rate to obtain more diverse results, adjusting number of optimization steps to balance quality and speed, and adjusting the loss weights. 


[//]: # (## Sparse and Dense Joint locations Control)

## Text-Conditioned Goal Reaching using Motion Control Policy
We train a motion control policy capable of reaching dynamic goal locations by leveraging locomotion skills specified through text prompts. The motion control policy is trained for three kinds of locomotion: walking, running, and hopping on the left leg. The control policy can generate >300 frames per second.
we demonstrate how to define a sequence of waypoints to be reached in the [cfg files](./data/test_locomotion).
You can run the following command to generate example motions of walking to a sequence of goals:
```
source ./demos/goal_reach.sh
```
The results can be visualized as follows:
```
python -m visualize.vis_seq --add_floor 1 --seq_path './policy_train/reach_location_mld/fixtext_repeat_floor100_hop10_skate100/env_test/demo_walk_path0/0.pkl' 
```

## Sparse and Dense Joint Trajectory Control
We provide a script to generate motions with **sparse/dense joint trajectory control**.
Below we demonstrate some examples of controlling hand wrists and 2D pelvis trajectories.
This script assumes starting from a **standing pose** and the specified joint trajectory needs to be feasible with the starting pose.
To accommodate this, we set a tolerance period (1.5 seconds in the script) at the start of the sequence. During this period, no trajectory constraints are enforced, allowing sufficient time for the human to transition smoothly and feasibly to the controlled trajectory from the standing pose.
You can run the following command to generate example motions:
```
source ./demos/traj.sh
```
The generated sequences can be visualized using the four commands below. 
The trajectories are visualized as a sequence of spheres, with colors transitioning from dark to red to represent relative time.
    
- In the punch example, there is a single trajectory point at 1.5 seconds.
    
- In the other three examples, trajectory points are distributed across a range from 1.5 to 6 seconds.

You can find the utility script for creating the example control trajectories [here](./data_scripts/export_traj.py).This script includes definitions for: frame index and location for each control trajectory point, and index of the joint to be controlled.
```
python -m visualize.vis_seq --add_floor 1 --translate_body 1 --vis_joint 1 --seq_path './data/traj_test/dense_frame180_walk_circle/mld_optim_global/floor1.0_skate1.0_jerk0.0_use_pred_joints_init1.0_ddim10_guidance5.0_seed0_lr0.05_steps100/*.pkl'
```
```
python -m visualize.vis_seq --add_floor 1 --translate_body 1 --vis_joint 1 --seq_path './data/traj_test/sparse_frame180_walk_square/mld_optim_global/floor1.0_skate1.0_jerk0.0_use_pred_joints_init1.0_ddim10_guidance5.0_seed0_lr0.05_steps100/*.pkl'
```
```
python -m visualize.vis_seq --add_floor 1 --translate_body 1 --vis_joint 1 --seq_path './data/traj_test/dense_frame180_wave_right_hand_circle/mld_optim_global/floor1.0_skate1.0_jerk0.0_use_pred_joints_init1.0_ddim10_guidance5.0_seed0_lr0.05_steps100/*.pkl'
```
```
python -m visualize.vis_seq --add_floor 1 --translate_body 1 --vis_joint 1 --seq_path './data/traj_test/sparse_punch/mld_optim_global/floor1.0_skate1.0_jerk0.0_use_pred_joints_init1.0_ddim10_guidance5.0_seed0_lr0.05_steps100/*.pkl'
```
You can test with custom trajectories by setting `--input_path` to your custom control trajectories.
If you have ground truth initial bodies and joint trajectories from dataset, you can modify the script to use initial bodies from dataset instead of the rest standing pose similar to the [inbetweening script](./mld/optim_mld.py).

# Training


- Below we provide the documentation of data processing and model training using different data sources.
By default, we provide commands for training on the BABEL dataset. Instructions for training on the HML3D dataset are available in the collapsible section. Additionally, guidance is provided for training on a custom motion dataset with text annotations.

- We use [wandb](https://wandb.ai/site/) for training logging. You may need to set up your own wandb account and log in before running the training scripts.

- Our training process includes stochastic factors such as random data sampling, scheduled training, and reinforcement learning-based policy training. As a result, we observed that different behaviors may occur when training on different environments.

- You can test the trained models by changing the model path in the demo scripts.

[//]: # (## Data Preparation)

[//]: # (```)

[//]: # (python -m data_scripts.extract_dataset)

[//]: # (```)

[//]: # ()
[//]: # (## Motion Primitive VAE)

[//]: # (``` )

[//]: # (python -m mld.train_mvae --track 1 --exp_name 'mvae_fps_clip' --data_args.dataset 'mp_seq_v2' --data_args.data_dir './data/seq_data_zero_male' --data_args.cfg_path './config_files/config_hydra/motion_primitive/mp_h2_f8_r8.yaml' --data_args.weight_scheme 'text_samp:0.' --train_args.batch_size 128  --train_args.weight_kl 1e-6  --train_args.stage1_steps 100000 --train_args.stage2_steps 50000 --train_args.stage3_steps 50000 --train_args.save_interval 50000  --train_args.weight_smpl_joints_rec 10.0 --train_args.weight_joints_consistency 10.0 --train_args.weight_transl_delta 100 --train_args.weight_joints_delta 100 --train_args.weight_orient_delta 100  --model_args.arch 'all_encoder' --train_args.ema_decay 0.999 --model_args.num_layers 7 --model_args.latent_dim 1 256)

[//]: # (```)

[//]: # ()
[//]: # (## Latent Motion Primitive Diffusion Model)

[//]: # (``` )

[//]: # (python -m mld.train_mld --track 1 --exp_name 'mld_fps_clip_repeat_euler' --train_args.batch_size 1024 --train_args.use_amp 1 --data_args.dataset 'mp_seq_v2' --data_args.data_dir './data/seq_data_zero_male' --data_args.cfg_path './config_files/config_hydra/motion_primitive/mp_h2_f8_r4.yaml' --denoiser_args.mvae_path './mvae/mvae_fps_clip/checkpoint_200000.pt' --denoiser_args.train_rollout_type 'full' --denoiser_args.train_rollout_history 'rollout' --train_args.stage1_steps 100000 --train_args.stage2_steps 100000 --train_args.stage3_steps 100000 --train_args.save_interval 100000 --train_args.weight_latent_rec 1.0 --train_args.weight_feature_rec 1.0 --train_args.weight_smpl_joints_rec 0 --train_args.weight_joints_consistency 0 --train_args.weight_transl_delta 1e4 --train_args.weight_joints_delta 1e4 --train_args.weight_orient_delta 1e4 --data_args.weight_scheme 'text_samp:0.' denoiser-args.model-args:denoiser-transformer-args)

[//]: # (```)

[//]: # ()
[//]: # (## Motion Control Policy)

[//]: # (```)

[//]: # (python -m control.train_reach_location_mld --track 1 --exp_name 'fixtext_repeat_floor100_hop10_skate100' --denoiser_checkpoint './mld_denoiser/mld_fps_clip_euler/checkpoint_300000.pt' --total_timesteps 200000000 --env_args.export_interval 1000 --env_args.num_envs 256 --env_args.num_steps 32 --minibatch_size 1024 --update_epochs 10 --learning_rate 3e-4 --max_grad_norm 0.1 --env_args.texts 'walk' 'run' 'hop on left leg' --env_args.success_threshold 0.3 --env_args.weight_success 1.0 --env_args.weight_dist 1.0 --env_args.weight_foot_floor 100.0 --env_args.weight_skate 100.0 --env_args.weight_orient 0.1 --policy_args.min_log_std -1.0 --policy_args.max_log_std 1.0 --policy_args.latent_dim 512 --env_args.goal_dist_max_init 5.0 --env_args.goal_schedule_interval 50000  --policy_args.use_lora 0 --policy_args.lora_rank 16 --policy_args.n_blocks 2 --policy_args.use_tanh_scale 1 --policy_args.use_zero_init 1 --init_data_path './data/stand.pkl' --env_args.weight_rotation 10.0 --env_args.weight_delta 0.0  --env_args.obs_goal_angle_clip 60.0 --env_args.obs_goal_dist_clip 5.0  --env_args.use_predicted_joints 1 --env_args.goal_angle_init 120.0 --env_args.goal_angle_delta 0.0)

[//]: # (```)

[//]: # ()
[//]: # (<details>)

[//]: # (<summary> Train on HML3D dataset:</summary>)

[//]: # ()
[//]: # (## Data Preparation - HML3D)

[//]: # (```)

[//]: # (python -m data_scripts.extract_dataset_hml3d_smplh)

[//]: # (```)

[//]: # ()
[//]: # (## Motion Primitive VAE - HML3D)

[//]: # (```)

[//]: # (python -m mld.train_mvae --track 1 --exp_name 'mvae_smplh_hml3d_2_8_4' --data_args.body_type 'smplh' --data_args.dataset 'hml3d' --data_args.data_dir './data/hml3d_smplh/seq_data_zero_male/' --data_args.cfg_path './config_files/config_hydra/motion_primitive/hml_mp_h2_f8_r4.yaml' --data_args.weight_scheme 'uniform' --train_args.batch_size 128  --train_args.weight_kl 1e-6  --train_args.stage1_steps 100000 --train_args.stage2_steps 50000 --train_args.stage3_steps 50000 --train_args.save_interval 50000  --train_args.weight_smpl_joints_rec 10.0 --train_args.weight_joints_consistency 10.0 --train_args.weight_transl_delta 100 --train_args.weight_joints_delta 100 --train_args.weight_orient_delta 100  --model_args.arch 'all_encoder' --train_args.ema_decay 0.999 --model_args.num_layers 7 --model_args.latent_dim 1 256)

[//]: # (```)

[//]: # ()
[//]: # (## Latent Motion Primitive Diffusion Model - HML3D)

[//]: # (```)

[//]: # (python -m mld.train_mld --track 1 --exp_name 'smplh_hml3d_2_8_4' --train_args.batch_size 1024 --train_args.use_amp 1 --data_args.body_type 'smplh' --data_args.dataset 'hml3d' --data_args.data_dir './data/hml3d_smplh/seq_data_zero_male/' --data_args.cfg_path './config_files/config_hydra/motion_primitive/hml_mp_h2_f8_r4.yaml' --denoiser_args.mvae_path './mvae/mvae_smplh_hml3d_2_8_4/checkpoint_200000.pt' --denoiser_args.train_rollout_type 'full' --denoiser_args.train_rollout_history 'rollout' --train_args.stage1_steps 100000 --train_args.stage2_steps 100000 --train_args.stage3_steps 100000 --train_args.save_interval 100000 --train_args.weight_latent_rec 1.0 --train_args.weight_feature_rec 1.0 --train_args.weight_smpl_joints_rec 0 --train_args.weight_joints_consistency 0 --train_args.weight_transl_delta 1e4 --train_args.weight_joints_delta 1e4 --train_args.weight_orient_delta 1e4 --data_args.weight_scheme 'uniform' denoiser-args.model-args:denoiser-transformer-args)

[//]: # (```)

[//]: # (</details>)

## Data Preparation
- Please first download the BABEL and AMASS SMPL-X gendered dataset and structure the folder as in [data setup section](#data-and-model-checkpoints).
- Please execute the following command to preprocess the BABEL dataset and extract the motion-text data. 
- For details of data preprocessing, you can check the collapsed section of training using custom dataset below.
```
python -m data_scripts.extract_dataset
```

## Train Motion Primitive VAE
``` 
python -m mld.train_mvae --track 1 --exp_name 'mvae_babel_smplx' --data_args.dataset 'mp_seq_v2' --data_args.data_dir './data/seq_data_zero_male' --data_args.cfg_path './config_files/config_hydra/motion_primitive/mp_h2_f8_r8.yaml' --data_args.weight_scheme 'text_samp:0.' --train_args.batch_size 128  --train_args.weight_kl 1e-6  --train_args.stage1_steps 100000 --train_args.stage2_steps 50000 --train_args.stage3_steps 50000 --train_args.save_interval 50000  --train_args.weight_smpl_joints_rec 10.0 --train_args.weight_joints_consistency 10.0 --train_args.weight_transl_delta 100 --train_args.weight_joints_delta 100 --train_args.weight_orient_delta 100  --model_args.arch 'all_encoder' --train_args.ema_decay 0.999 --model_args.num_layers 7 --model_args.latent_dim 1 256
```

## Train Latent Motion Primitive Diffusion Model
``` 
python -m mld.train_mld --track 1 --exp_name 'mld_babel_smplx' --train_args.batch_size 1024 --train_args.use_amp 1 --data_args.dataset 'mp_seq_v2' --data_args.data_dir './data/seq_data_zero_male' --data_args.cfg_path './config_files/config_hydra/motion_primitive/mp_h2_f8_r4.yaml' --denoiser_args.mvae_path './mvae/mvae_babel_smplx/checkpoint_200000.pt' --denoiser_args.train_rollout_type 'full' --denoiser_args.train_rollout_history 'rollout' --train_args.stage1_steps 100000 --train_args.stage2_steps 100000 --train_args.stage3_steps 100000 --train_args.save_interval 100000 --train_args.weight_latent_rec 1.0 --train_args.weight_feature_rec 1.0 --train_args.weight_smpl_joints_rec 0 --train_args.weight_joints_consistency 0 --train_args.weight_transl_delta 1e4 --train_args.weight_joints_delta 1e4 --train_args.weight_orient_delta 1e4 --data_args.weight_scheme 'text_samp:0.' denoiser-args.model-args:denoiser-transformer-args
```

## Train Motion Control Policy
```
python -m control.train_reach_location_mld --track 1 --exp_name 'control_policy' --denoiser_checkpoint './mld_denoiser/mld_fps_clip_euler/checkpoint_300000.pt' --total_timesteps 200000000 --env_args.export_interval 1000 --env_args.num_envs 256 --env_args.num_steps 32 --minibatch_size 1024 --update_epochs 10 --learning_rate 3e-4 --max_grad_norm 0.1 --env_args.texts 'walk' 'run' 'hop on left leg' --env_args.success_threshold 0.3 --env_args.weight_success 1.0 --env_args.weight_dist 1.0 --env_args.weight_foot_floor 100.0 --env_args.weight_skate 100.0 --env_args.weight_orient 0.1 --policy_args.min_log_std -1.0 --policy_args.max_log_std 1.0 --policy_args.latent_dim 512 --env_args.goal_dist_max_init 5.0 --env_args.goal_schedule_interval 50000  --policy_args.use_lora 0 --policy_args.lora_rank 16 --policy_args.n_blocks 2 --policy_args.use_tanh_scale 1 --policy_args.use_zero_init 1 --init_data_path './data/stand.pkl' --env_args.weight_rotation 10.0 --env_args.weight_delta 0.0  --env_args.obs_goal_angle_clip 60.0 --env_args.obs_goal_dist_clip 5.0  --env_args.use_predicted_joints 1 --env_args.goal_angle_init 120.0 --env_args.goal_angle_delta 0.0
```

<details>
<summary><b>Train with HML3D dataset:</b></summary>

## Data Preparation - HML3D

Please first download the HML3D and AMASS SMPL-H gendered dataset and structure the folder as in [data setup section](#data-and-model-checkpoints).
```
python -m data_scripts.extract_dataset_hml3d_smplh
```

## Train Motion Primitive VAE - HML3D
```
python -m mld.train_mvae --track 1 --exp_name 'mvae_hml3d_smplh' --data_args.body_type 'smplh' --data_args.dataset 'hml3d' --data_args.data_dir './data/hml3d_smplh/seq_data_zero_male/' --data_args.cfg_path './config_files/config_hydra/motion_primitive/hml_mp_h2_f8_r4.yaml' --data_args.weight_scheme 'uniform' --train_args.batch_size 128  --train_args.weight_kl 1e-6  --train_args.stage1_steps 100000 --train_args.stage2_steps 50000 --train_args.stage3_steps 50000 --train_args.save_interval 50000  --train_args.weight_smpl_joints_rec 10.0 --train_args.weight_joints_consistency 10.0 --train_args.weight_transl_delta 100 --train_args.weight_joints_delta 100 --train_args.weight_orient_delta 100  --model_args.arch 'all_encoder' --train_args.ema_decay 0.999 --model_args.num_layers 7 --model_args.latent_dim 1 256
```

## Train Latent Motion Primitive Diffusion Model - HML3D
```
python -m mld.train_mld --track 1 --exp_name 'mld_hml3d_smplh' --train_args.batch_size 1024 --train_args.use_amp 1 --data_args.body_type 'smplh' --data_args.dataset 'hml3d' --data_args.data_dir './data/hml3d_smplh/seq_data_zero_male/' --data_args.cfg_path './config_files/config_hydra/motion_primitive/hml_mp_h2_f8_r4.yaml' --denoiser_args.mvae_path './mvae/mvae_hml3d_smplh/checkpoint_200000.pt' --denoiser_args.train_rollout_type 'full' --denoiser_args.train_rollout_history 'rollout' --train_args.stage1_steps 100000 --train_args.stage2_steps 100000 --train_args.stage3_steps 100000 --train_args.save_interval 100000 --train_args.weight_latent_rec 1.0 --train_args.weight_feature_rec 1.0 --train_args.weight_smpl_joints_rec 0 --train_args.weight_joints_consistency 0 --train_args.weight_transl_delta 1e4 --train_args.weight_joints_delta 1e4 --train_args.weight_orient_delta 1e4 --data_args.weight_scheme 'uniform' denoiser-args.model-args:denoiser-transformer-args
```
</details>

<details>
<summary><b>Train using custom dataset:</b></summary>

- Our model can train on custom motion dataset with text annotations.
We expect the motion data to be sequences of SMPL-X/H parameters.
We structure the text annotations for each sequence according to the BABEL annotation format. In this format, a sequence can have an arbitrary number of segment text labels. Each segment is defined by a start time (`start_t`) and an end time (`end_t`), both measured in seconds. The text annotation for each segment is stored under the key `proc_label`.
The segments can overlap, and the segments can also range the whole sequence as in the HML3D dataset.
Please check the data preprocessing script for [BABEL](./data_scripts/extract_dataset.py) and [HML3D](./data_scripts/extract_dataset_hml3d_smplh.py) for details. 
- Please export the dataset to a separate folder and recalcualte the mean and std statistics for motion features using the custom dataset.
- You can specify the data source when training the motion primitive VAE or latent diffusion model using `--data_args.data_dir`, and the body type using `--data_args.body_type`.
The configurations of motion primitive length, max rollout number in scheduled training, FPS of motion data are set in the [cfg files](). 
</details>

# Evaluation

## Text-Conditioned Temporal Motion Composition
The evaluation for text-conditioned temporal motion composition is based on the [FlowMDM](https://github.com/BarqueroGerman/FlowMDM) code. 
Please first set up the FlowMDM dependencies as follows:
- Set up the required dependencies: `source ./FlowMDM/setup.sh`
- Download the processed BABEL dataset for evaluation:
    - Download the processed version [here](https://drive.google.com/file/d/18a4eRh8mbIFb55FMHlnmI8B8tSTkbp4t/view?usp=share_link), and place it at `./FlowMDM/dataset/babel`.
    - Download the following [here](https://drive.google.com/file/d/1PBlbxawaeFTxtKkKDsoJwQGuDTdp52DD/view?usp=sharing), and place it at `./FlowMDM/dataset/babel`.

[//]: # (    - Reference from FlowMDM: https://github.com/BarqueroGerman/FlowMDM/tree/main/runners)

After setting up the dependencies, you can run the evaluation using the following command. The FlowMDM generation part may take around 1 day.
```
source ./evaluation/eval_gen_composition.sh
```
- The evaluation results of FlowMDM will be saved at `./FlowMDM/results/babel/FlowMDM/evaluations_summary/001300000_fast_10_transLen30babel_random_seed0.json`.
- The evaluation results of DART will be saved at `./FlowMDM/results/babel/Motion_FlowMDM_001300000_gscale1.5_fastbabel_random_seed0_s10/mld_fps_clip_repeat_euler_checkpoint_300000_guidance5.0_seed0/evaluations_summary/fast_10_transLen30babel_random_seed0.json`.

## Text-Conditioned Motion In-betweening
The generation and evaluation can be executed with the command below. The results will be displayed in the command line, and the file save path will also be indicated there.
```
source ./evaluation/eval_gen_inbetween.sh
```


## Text-Conditioned Goal Reaching
The generation and evaluation can be executed with the command below. The results will be displayed in the command line, and the file save path will also be indicated there.
```
source ./evaluation/eval_gen_goal_reach.sh
```

# Acknowledgements
Our code is built upon many prior projects, including but not limited to:

[DNO](https://github.com/korrawe/Diffusion-Noise-Optimization), [MDM](https://github.com/GuyTevet/motion-diffusion-model), [MLD](https://github.com/ChenFengYe/motion-latent-diffusion), [FlowMDM](https://github.com/BarqueroGerman/FlowMDM), [text-to-motion](https://github.com/EricGuo5513/text-to-motion), [guided-diffusion](https://github.com/openai/guided-diffusion), [ACTOR](https://github.com/Mathux/ACTOR), [DIMOS](https://github.com/zkf1997/DIMOS)

[//]: # (# License)

[//]: # (* Our code and model checkpoints employ the MIT License.)

[//]: # (* Note that our code depends on third-party software and datasets that employ their respective licenses. Here are some examples:)

[//]: # (    * Code/model/data relevant to the SMPL-X body model follows its own license.)

[//]: # (    * Code/model/data relevant to the AMASS dataset follows its own license.)

[//]: # (    * Blender and its SMPL-X add-on employ their respective license.)

  
# Citation
```
@inproceedings{Zhao:DartControl:2025,
   title = {{DartControl}: A Diffusion-Based Autoregressive Motion Model for Real-Time Text-Driven Motion Control},
   author = {Zhao, Kaifeng and Li, Gen and Tang, Siyu},
   booktitle = {The Thirteenth International Conference on Learning Representations (ICLR)},
   year = {2025}
}
```

# Contact

If you run into any problems or have any questions, feel free to contact [Kaifeng Zhao](mailto:kaifeng.zhao@inf.ethz.ch) or create an issue.
