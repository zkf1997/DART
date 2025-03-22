respacing='ddim10'
guidance=5
export_smpl=0
use_predicted_joints=1
batch_size=4
optim_lr=0.05
optim_steps=100
# larger optim steps such as 300 can further reduce the goal joints location error
optim_unit_grad=1
optim_anneal_lr=1
weight_jerk=0.0
weight_floor=0.0
init_noise_scale=0.1

model='./mld_denoiser/mld_fps_clip_repeat_euler/checkpoint_300000.pt'
text_prompt='pace in circles'

# the input seq only has one start frame and one goal frame, the between frames are the repeating of the first frame
optim_input='./data/inbetween/pace_in_circles/babel_1f.pkl'
seed_type='repeat'
python -m mld.optim_mld --denoiser_checkpoint "$model" --optim_input "$optim_input" --text_prompt "$text_prompt" --optim_lr $optim_lr --optim_steps $optim_steps --batch_size $batch_size --guidance_param $guidance --respacing "$respacing" --export_smpl $export_smpl  --use_predicted_joints $use_predicted_joints  --optim_unit_grad $optim_unit_grad  --optim_anneal_lr $optim_anneal_lr  --weight_jerk $weight_jerk  --weight_floor $weight_floor --seed_type $seed_type  --init_noise_scale $init_noise_scale

# the input seq has several history frames and one goal frame, the first H+1 frames will be used to extract the initial history seed
optim_input='./data/inbetween/pace_in_circles/babel_2f.pkl'
seed_type='history'
python -m mld.optim_mld --denoiser_checkpoint "$model" --optim_input "$optim_input" --text_prompt "$text_prompt" --optim_lr $optim_lr --optim_steps $optim_steps --batch_size $batch_size --guidance_param $guidance --respacing "$respacing" --export_smpl $export_smpl  --use_predicted_joints $use_predicted_joints  --optim_unit_grad $optim_unit_grad  --optim_anneal_lr $optim_anneal_lr  --weight_jerk $weight_jerk  --weight_floor $weight_floor --seed_type $seed_type  --init_noise_scale $init_noise_scale