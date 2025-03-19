respacing=''
guidance=5.0
export_smpl=1
zero_noise=0
batch_size=4
use_predicted_joints=1
dataset='babel'
fix_floor=0

text_prompt='walk in circles*20'
#text_prompt='sidestep left*20'
#text_prompt='walk backwards*20'
#text_prompt='dance*20'
#text_prompt='crawl*20'
#text_prompt='climb down stairs*20'
#text_prompt='wave hands*10,walk forward*5,cartwheel*8,walk forward*5,turn left*3,sit down*6,stand up*4,hop on left leg*10,pace in circle*10,dance*12,stand*5,wave right hand*10'


model_list=(
'./mld_denoiser/mld_fps_clip_repeat_euler/checkpoint_300000.pt'
)

for model in "${model_list[@]}"; do
  python -m mld.rollout_mld --denoiser_checkpoint "$model" --batch_size $batch_size  --text_prompt "$text_prompt" --guidance_param $guidance --respacing "$respacing" --export_smpl $export_smpl --zero_noise $zero_noise --use_predicted_joints $use_predicted_joints  --dataset "$dataset"  --fix_floor $fix_floor
done
