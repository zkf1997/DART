# generate evaluation timeline config.
# The timeline configurations are extracted from the BABEL validation set, involving some random text selection and shuffling. Consequently, the exported timeline might vary slightly, leading to minor differences in the evaluation metrics.
python -m data_scripts.get_val_cfg

# flowmdm generation and evaluation
cd FlowMDM
python -m runners.eval --model_path ./results/babel/FlowMDM/model001300000.pt --dataset babel --eval_mode fast --bpe_denoising_step 125 --guidance_param 1.5 --transition_length 30
cd ..

# dart generation
respacing=''
guidance=5.0
export_smpl=0
zero_noise=0
use_predicted_joints=0
fix_floor=0
flowmdm_dir='./FlowMDM/results/babel/FlowMDM/evaluation_precomputed/Motion_FlowMDM_001300000_gscale1.5_fastbabel_random_seed0_s10'
model_list=(
'./mld_denoiser/mld_fps_clip_repeat_euler/checkpoint_300000.pt'
)
for model in "${model_list[@]}"; do
  python -m mld.sample_flowmdm_mld --denoiser_checkpoint "$model" --guidance_param $guidance --respacing "$respacing" --export_smpl $export_smpl --flowmdm_dir "$flowmdm_dir" --use_predicted_joints $use_predicted_joints --zero_noise $zero_noise  --fix_floor $fix_floor
done

#dart evaluation
cd FlowMDM
dir_list=(
'./results/babel/Motion_FlowMDM_001300000_gscale1.5_fastbabel_random_seed0_s10/mld_fps_clip_repeat_euler_checkpoint_300000_guidance5.0_seed0'
)
for dir in "${dir_list[@]}"; do
  python -m runners.eval_load --model_path './results/babel/FlowMDM/model001300000.pt' --dataset babel --eval_mode fast --transition_length 30  --load_dir "$dir"
done
cd ..