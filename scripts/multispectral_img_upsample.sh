################# 32 x 32 -> 64 x 64 ###########################################################
MODEL_FLAGS="--cross_attention_resolutions 16,8 --attention_resolutions 16,8
--dropout 0.1 --learn_sigma False --num_channels 128
--num_head_channels 64 --num_res_blocks 3 --resblock_updown True --use_fp16 True 
--use_scale_shift_norm True --ema_rate 0.9999"


DIFFUSION_FLAGS="--noise_schedule linear --diffusion_steps 1000" 

SAMPLING_FLAGS="--batch size 256 --sample_fn ddpm"

# Modify the following pathes to your own paths
DATA_DIR="/home/iman/CodeBase/python/AI/DeepLearning/cv/master_program/multispectralDiffusionCodeGithub/data"
OUTPUT_DIR="/home/iman/CodeBase/python/AI/DeepLearning/cv/master_program/multispectralDiffusionCodeGithub/upsamples"
MODEL_PATH="/home/iman/CodeBase/python/AI/DeepLearning/cv/master_program/multispectralDiffusionCodeGithub/outputs/model000008.pt"

python3 -m MultiSpectral-Diffusion.image_upsample --data_dir ${DATA_DIR} --output_dir ${OUTPUT_DIR} --model_path ${MODEL_PATH} $MODEL_FLAGS $DIFFUSION_FLAGS $SAMPLING_FLAGS

