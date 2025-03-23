respacing='ddim10'
guidance=5
export_smpl=1
use_predicted_joints=1
batch_size=4
optim_lr=0.05
optim_steps=100
#optim_steps=300
optim_unit_grad=1
optim_anneal_lr=1
weight_floor=1.0
weight_skate=1.0
weight_jerk=0.0
init_scale=1.0
fps=30
mode='global'

model_list=(
'./mld_denoiser/mld_fps_clip_repeat_euler/checkpoint_300000.pt'
)

use_2d_dist=0
input_path_list=(
'./data/traj_test/sparse_punch/traj_text.pkl'
'./data/traj_test/dense_frame180_wave_right_hand_circle/traj_text.pkl'
)

for model in "${model_list[@]}"; do
  for input_path in "${input_path_list[@]}"; do
    python -m mld.optim_pelvis_global_mld --mode "$mode" --denoiser_checkpoint "$model" --input_path "$input_path" --fps $fps --optim_lr $optim_lr --optim_steps $optim_steps --batch_size $batch_size --guidance_param $guidance --respacing "$respacing" --export_smpl $export_smpl  --use_predicted_joints $use_predicted_joints  --optim_unit_grad $optim_unit_grad  --optim_anneal_lr $optim_anneal_lr  --weight_skate $weight_skate  --weight_floor $weight_floor --weight_jerk $weight_jerk --init_scale $init_scale  --use_2d_dist $use_2d_dist
  done
done

use_2d_dist=1
input_path_list=(
'./data/traj_test/dense_frame180_walk_circle/traj_text.pkl'
'./data/traj_test/sparse_frame180_walk_square/traj_text.pkl'
)
for model in "${model_list[@]}"; do
  for input_path in "${input_path_list[@]}"; do
    python -m mld.optim_pelvis_global_mld --mode "$mode" --denoiser_checkpoint "$model" --input_path "$input_path" --fps $fps --optim_lr $optim_lr --optim_steps $optim_steps --batch_size $batch_size --guidance_param $guidance --respacing "$respacing" --export_smpl $export_smpl  --use_predicted_joints $use_predicted_joints  --optim_unit_grad $optim_unit_grad  --optim_anneal_lr $optim_anneal_lr  --weight_skate $weight_skate  --weight_floor $weight_floor --weight_jerk $weight_jerk --init_scale $init_scale  --use_2d_dist $use_2d_dist
  done
done
