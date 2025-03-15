respacing=''
guidance=5
export_smpl=0
zero_noise=0
batch_size=1
use_predicted_joints=1

model_list=(
'./mld_denoiser/mld_fps_clip_repeat_euler/checkpoint_300000.pt'
)

for model in "${model_list[@]}"; do
  python -m mld.rollout_demo --denoiser_checkpoint "$model" --batch_size $batch_size --guidance_param $guidance --respacing "$respacing" --use_predicted_joints $use_predicted_joints
done
