# VolumetricSMPL Integration for interaction Modelling
## Overview
We present the integration of the [VolumetricSMPL](https://github.com/markomih/VolumetricSMPL) model for interaction modeling into DART's optimization-based scene interaction motion synthesis framework.
VolumetricSMPL is an extension of the SMPL body model that incorporates a volumetric (signed distance field, SDF) representation.
Using VolumetricSMPL, we can compute the signed distance from scene points to the human body surfaces, enabling a human-centric evaluation of human-scene interactions. This approach eliminates the need for a precomputed scene SDF grid, which is often challenging to generate from real-world scene scans.

We provide a simple demo of generating **navigation sequences in indoor scenes** from EgoBody dataset using the VolumetricSMPL model. 
The demo showcases how to use the VolumetricSMPL model to compute the signed distance from scene points to the body surface and how to use this information in the DART optimization framework for interaction modeling.

## Setup
Please first follow the setup of [DartControl](https://github.com/zkf1997/DART?tab=readme-ov-file#getting-started). After setting up the environment and downloading the necessary data files of DartControl, please install the following packages:
```
pip install VolumetricSMPL
pip install git+https://github.com/markomih/COAP.git
```

To run the navigation demo in EgoBody scenes, please download the scene assets from [EgoBody dataset](https://egobody.ethz.ch/data/dataset/scene_mesh_4render_dart.zip) and place the scene assets in `./data/scene_mesh_4render`. The dataset structure should look like this:
    
```
  ./
  ├── data
  │   ├── scene_mesh_4render
  │   │   ├── SCENE_NAME
  │   │   │   ├── mesh_floor_zup.ply  # scene mesh
  │   │   │   ├── points_16384.ply  # scene point cloud
```

## Demo

To run the demo, execute the following command:
```
source ./VolSMPL/test_optim_egobody.sh
```
Please note that this demo may require over 20 GiB of CUDA memory, depending on the length of the generated sequence and the size of the scene.

To visualize the generated sequence, you can use the provided script below. The goal location is visualized as a red sphere. 
```
python -m visualize.vis_seq --translate_body 1 --seq_path './mld_denoiser/mld_fps_clip_repeat_euler/checkpoint_300000/optim_volsmpl/cnb_dlab_0215_walk_use_pred_joints_ddim10_guidance5.0_seed0_lr0.05_coll1.0_contact0.0_jerk0.1/sample*.pkl'
```
You can specify the sequence index in `--seq_path` to investigate a specific sequence. The `--translate_body` flag is used to translate multiple sequences vertically to avoid overlapping.

For advanced visualization using Blender, please refer to the provided documentation [here](https://github.com/zkf1997/DART?tab=readme-ov-file#blender-visualization).

The implementation of VolumetricSMPL-based interaction modelling can be found at `calc_coll_cont_volsmpl` of [optim_egobody.py](./optim_egobody.py).