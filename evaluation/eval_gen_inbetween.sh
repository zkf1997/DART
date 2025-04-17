python -m data_scripts.export_optim_subseq

respacing='ddim10'
guidance=5
export_smpl=0
use_predicted_joints=1
batch_size=8
optim_lr=0.05
optim_steps=300
optim_unit_grad=1
optim_anneal_lr=1
weight_jerk=0.0
weight_floor=0.0
#init_noise_scale=1.0
init_noise_scale=0.1

seed_type='repeat'
optim_input_list=(
'./data/inbetween/opt_eval_20fps_1f/0_walk.pkl'
'./data/inbetween/opt_eval_20fps_1f/1_run forward.pkl'
'./data/inbetween/opt_eval_20fps_1f/2_jump forward.pkl'
'./data/inbetween/opt_eval_20fps_1f/3_pace in circles.pkl'
'./data/inbetween/opt_eval_20fps_1f/4_crawl.pkl'
'./data/inbetween/opt_eval_20fps_1f/5_dance.pkl'
'./data/inbetween/opt_eval_20fps_1f/6_walk backwards.pkl'
'./data/inbetween/opt_eval_20fps_1f/7_climb down stairs.pkl'
'./data/inbetween/opt_eval_20fps_1f/8_sit down.pkl'
)

text_prompt_list=(
'a person walks'
'a person runs forward'
'a person jumps forward'
'a person paces in circles'
'a person crawls'
'a person dances'
'a person walks backwards'
'a person climbs down stairs'
'a person sits down'
)


model_list=(
'./mld_denoiser/smplh_hml3d_2_8_4/checkpoint_300000.pt'
)

for model in "${model_list[@]}"; do
  for idx in "${!optim_input_list[@]}"; do
    python -m mld.optim_mld --denoiser_checkpoint "$model" --optim_input "${optim_input_list[idx]}" --text_prompt "${text_prompt_list[idx]}" --optim_lr $optim_lr --optim_steps $optim_steps --batch_size $batch_size --guidance_param $guidance --respacing "$respacing" --export_smpl $export_smpl  --use_predicted_joints $use_predicted_joints  --optim_unit_grad $optim_unit_grad  --optim_anneal_lr $optim_anneal_lr  --weight_jerk $weight_jerk  --weight_floor $weight_floor --seed_type $seed_type  --init_noise_scale $init_noise_scale
  done
done

python -m evaluation.inbetween