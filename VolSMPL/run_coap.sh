#!/bin/bash
respacing='ddim10'
guidance=5
export_smpl=1
use_predicted_joints=1
batch_size=1
num_seq=8
optim_lr=0.05
optim_steps=100
optim_unit_grad=1
optim_anneal_lr=1

load_cache=0
weight_jerk=0.1
weight_contact=0
weight_skate=0.0
contact_thresh=0.00
init_noise_scale=0.1
weight_collision=0.1

interaction_cfg='./VolSMPL/cfg/egobody.json'

coap_max_frames=16
loss_type='coap'
volsmpl_gender='neutral'
num_points=16384

model_list=(
'./mld_denoiser/mld_fps_clip_repeat_euler/checkpoint_300000.pt'
)

for model in "${model_list[@]}"; do
  python -m VolSMPL.optim_egobody --denoiser_checkpoint "$model" --interaction_cfg "$interaction_cfg" --optim_lr $optim_lr --optim_steps $optim_steps --batch_size $batch_size --guidance_param $guidance --respacing "$respacing" --export_smpl $export_smpl  --use_predicted_joints $use_predicted_joints  --optim_unit_grad $optim_unit_grad  --optim_anneal_lr $optim_anneal_lr  --weight_jerk $weight_jerk --weight_collision $weight_collision  --weight_contact $weight_contact  --weight_skate $weight_skate  --contact_thresh $contact_thresh  --load_cache $load_cache  --init_noise_scale $init_noise_scale  --loss_type $loss_type  --num_points $num_points  --coap_max_frames $coap_max_frames  --num_seq $num_seq
done