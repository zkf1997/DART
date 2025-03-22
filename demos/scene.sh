respacing='ddim10'
guidance=5
export_smpl=1
use_predicted_joints=1
batch_size=8
optim_lr=0.01
#optim_lr=0.1
optim_steps=100
optim_unit_grad=1
optim_anneal_lr=1

weight_jerk=0.1
weight_collision=0.1
weight_contact=0.1
weight_skate=0.0
contact_thresh=0.00
init_noise_scale=0.1

load_cache=0
interaction_cfg_list=(
'./data/optim_interaction/climb_down.json'
'./data/optim_interaction/sit.json'
)

model_list=(
'./mld_denoiser/mld_fps_clip_repeat_euler/checkpoint_300000.pt'
)

for interaction_cfg in "${interaction_cfg_list[@]}"; do
  for model in "${model_list[@]}"; do
    python -m mld.optim_scene_mld --denoiser_checkpoint "$model" --interaction_cfg "$interaction_cfg" --optim_lr $optim_lr --optim_steps $optim_steps --batch_size $batch_size --guidance_param $guidance --respacing "$respacing" --export_smpl $export_smpl  --use_predicted_joints $use_predicted_joints  --optim_unit_grad $optim_unit_grad  --optim_anneal_lr $optim_anneal_lr  --weight_jerk $weight_jerk --weight_collision $weight_collision  --weight_contact $weight_contact  --weight_skate $weight_skate  --contact_thresh $contact_thresh  --load_cache $load_cache  --init_noise_scale $init_noise_scale
  done
done