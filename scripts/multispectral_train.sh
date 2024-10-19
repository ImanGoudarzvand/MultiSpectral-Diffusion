################# 64 x 64 ###########################################################
MODEL_FLAGS="--cross_attention_resolutions 16,8 --attention_resolutions 16,8
--dropout 0.1 --learn_sigma False --num_channels 128
--num_head_channels 64 --num_res_blocks 3 --resblock_updown True --use_fp16 True 
--use_scale_shift_norm True --multitask_loss_weights 0.7,0.15,0.15 --ema_rate 0.9999"


TRAIN_FLAGS="--lr 0.0001 --batch_size 64 
# --log_interval 100 --save_interval 15000"  
DIFFUSION_FLAGS="--noise_schedule linear --diffusion_steps 1000" 

# Modify the following pathes to your own paths

DATA_DIR="/home/iman/CodeBase/python/AI/DeepLearning/cv/master_program/multispectralDiffusionCodeGithub/data"
OUTPUT_DIR="/home/iman/CodeBase/python/AI/DeepLearning/cv/master_program/multispectralDiffusionCodeGithub/outputs"
NUM_GPUS=2

mpiexec -n $NUM_GPUS python3 MultiSpectral-Diffusion/multispectral_train.py --data_dir ${DATA_DIR} --output_dir ${OUTPUT_DIR} $MODEL_FLAGS $TRAIN_FLAGS $DIFFUSION_FLAGS 
# python3 -m MultiSpectral-Diffusion.multispectral_train --data_dir ${DATA_DIR} --output_dir ${OUTPUT_DIR} $MODEL_FLAGS $TRAIN_FLAGS $DIFFUSION_FLAGS 

