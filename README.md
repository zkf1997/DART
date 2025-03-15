# DartControl: A Diffusion-Based Autoregressive Motion Model for Real-Time Text-Driven Motion Control (ICLR 2025, Spotlight)

[[website](https://zkf1997.github.io/DART/)] [[paper](https://arxiv.org/abs/2410.05260)] 

[teaser](https://zkf1997.github.io/DART/videos/teaser.mp4)

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
<summary>Tested system:</summary>
<details>
Our experiments and performance profiling are conducted on a workstation with single RTX 4090
GPU, intel i7-13700K CPU, 64GiB memory. The workstation runs with Ubuntu 22.04.4 LTS system.
</details>

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
* We use `pyrender` for interactive visualization of generated motions by default. Please refer to [pyrender viewer](https://pyrender.readthedocs.io/en/latest/generated/pyrender.viewer.Viewer.html) for the usage of the interactive viewer, such as rotating, panning, and zooming.
* We also support exporting the generated motions and visualize in [Blender](https://www.blender.org/) for advanced rendering. To import one motion sequence into blender, please first install the [SMPL-X Blender Add-on](https://gitlab.tuebingen.mpg.de/jtesch/smplx_blender_addon#installation), and use the "add animation" feature as shown in this video. You can use the space key to start/stop playing animation in Blender.
  <summary>Demonstration of importing motion into Blender:
  </summary>
  
  <details>
  
  Import demonstration video:

  ./assets/import.mp4

  </details>


# Motion Generation Demos
We offer a range of motion generation demos, including online text-conditioned motion generation and applications with spatial constraints and goals. 
These applications include motion in-betweening, waypoint goal reaching, and human-scene interaction generation.

## Interactive Online Text-Conditioned Motion Generation
```
source ./demos/run_demo.sh
```
This will open an interactive viewer and a command-line interface for text input. You can input text prompts and the model will generate the corresponding motion sequence on the fly.
A demonstration video is shown below:

https://zkf1997.github.io/DART/videos/0927.mp4


## Headless Text-Conditioned Motion Composition 
 
## Text-Conditioned Motion in-betweening

## Human-Scene Interaction Synthesis

[//]: # (## Sparse and Dense Joint locations Control)

## Text-Conditioned Goal Reaching using Motion Control Policy

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