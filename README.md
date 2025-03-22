# DartControl
## A Diffusion-Based Autoregressive Motion Model for Real-Time Text-Driven Motion Control (ICLR 2025, Spotlight)

### [[website](https://zkf1997.github.io/DART/)] | [[paper](https://arxiv.org/abs/2410.05260)] 


https://github.com/user-attachments/assets/b26e95e7-4af0-4548-bdca-8f361594951c



# Updates
This repository is under construction and the documentations for the following for will be updated.  

- [ ] Setup, generation demos, and visualization
- [ ] Data preparation and training
- [ ] Evaluation

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
* Please download this [link](https://drive.google.com/drive/folders/1vJg3GFVPT6kr6cA0HrQGmiAEBE2dkaps?usp=drive_link) containing model checkpoints and necessary data, extract and merge it to the project folder.

* Please download the following data from the respective websites and organize as shown below:
  * [SMPL-X body model](https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=smplx_lockedhead_20230207.zip)
  * [SMPL-H body model](https://download.is.tue.mpg.de/download.php?domain=mano&resume=1&sfile=smplh.tar.xz)
  * [AMASS](https://amass.is.tue.mpg.de/) (Only required for training, please down the gender-specific data for SMPL-H and SMPL-X)
  * [BABEL](https://download.is.tue.mpg.de/download.php?domain=teach&resume=1&sfile=babel-data/babel-teach.zip) (Only required for training)
  * [HumanML3D](https://github.com/EricGuo5513/HumanML3D)(Only required for training)

  <summary> 
  Project folder structure of separately downloaded data:
  </summary>
  <details>
  
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
  <summary>Demonstration of importing motion into Blender:
  </summary>
  
  <details>

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

<summary>Inbetweening using model trained on the HML3D dataset:</summary> 
<details>  

In addition to inbetweening with model trained on the BABEL dataset as shown above, we provide a script for inbetweening using a model trained on the HML3D dataset [here](./demos/inbetween_hml.sh). Please note: 

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
# Training

[//]: # (## Data Preparation)

[//]: # ()
[//]: # (## Motion Primitive VAE)

[//]: # ()
[//]: # (## Latent Motion Primitive Diffusion Model)

[//]: # ()
[//]: # (## Motion Control Policy)


# Evaluation

[//]: # (## Text-Conditioned Temporal Motion Composition)

[//]: # ()
[//]: # (## Text-Conditioned Motion In-betweening)

[//]: # ()
[//]: # (## Text-Conditioned Goal Reaching)

# Acknowledgements
Our code is built upon many prior projects, including but not limited to:

[DNO](https://github.com/korrawe/Diffusion-Noise-Optimization), [MDM](https://github.com/GuyTevet/motion-diffusion-model), [MLD](https://github.com/ChenFengYe/motion-latent-diffusion), [text-to-motion](https://github.com/EricGuo5513/text-to-motion), [guided-diffusion](https://github.com/openai/guided-diffusion), [ACTOR](https://github.com/Mathux/ACTOR), [DIMOS](https://github.com/zkf1997/DIMOS)

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
