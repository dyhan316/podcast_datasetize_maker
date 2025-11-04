#!/bin/bash
#SBATCH --job-name=small_model 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=3G
#SBATCH --gpus=1
#SBATCH --nodelist=node3
#SBATCH --qos interactive
#SBATCH --time 1:00:00
#SBATCH -o ./logs/test_%j.o
#SBATCH -e ./logs/test_%j.e

#!====DANNY!!===========================

#* uri path thing : /scratch/connectome/ahhyun724/DIVER/DATA/Uri_podcast/ => not used, moved to somewhere else
conda init bash
source activate /storage/connectome/DIVER/DIVER_py311_cu124

cd /scratch/connectome/dyhan316/DIVER_CLIP/podcast-ecog-paper/code

python encoding.py --layer 24 


#* gpt2-xl (python encoding.py --layer 24 --modelname=gpt2-xl)
# (Pdb) X.shape
# (5130, 1600)
# (Pdb) Y.shape #* this should be changed haha  => only two things (1) resampling to 500Hz (2) no flattening across channels
# (5130, 12672)

#* syntactic (python encoding.py  --modelname=syntactic)
# (Pdb) X.shape
# (5130, 96)
# (Pdb) Y.shape
# (5130, 12672)

#* other : phonetic, spectral, whisper-medium, en_core_web_lg, gpt2-xl, syntactic

python encoding.py  --modelname=syntactic --band none/highgamma

#*.... However, these are just aggregates or sth across channels and stuff... so need to look into more.

#!====DANNY!!===========================



conda init bash
source activate /storage/connectome/DIVER/DIVER_py311_cu124

# conda activate /storage/connectome/DIVER/DIVER_py311_cu124



cd /scratch/connectome/ahhyun724/DIVER/DIVER/CBraMod/

# run merge
#python merge_fp32.py --part-dir /pscratch/sd/t/tylee/Dataset/generalized_test/0709_modified/new_cliped/1.0_TUEG_fp32 \
#--dst /pscratch/sd/t/tylee/Dataset/generalized_test/0709_modified/new_cliped/1.0_TUEG_fp32/all_resample-500_highpass-0.3_lowpass-None.lmdb  \
#--src /pscratch/sd/t/tylee/Dataset/generalized_test/0709_modified/new_cliped/1.0_TUEG/all_resample-500_highpass-0.3_lowpass-None.lmdb

# DIVER_iEEG_FINAL_model
# LaBraM_2025SEP
# CBraMod_2025SEP

# run float64 to 32
#python float64_to_float32.py /pscratch/sd/t/tylee/Dataset/generalized_test/0709_modified/new_cliped/1.0_TUEG/all_resample-500_highpass-0.3_lowpass-None.lmdb /pscratch/sd/t/tylee/Dataset/generalized_test/0709_modified/new_cliped/1.0_TUEG_fp32/all_resample-500_highpass-0.3_lowpass-None.lmdb

# /global/cfs/cdirs/m4727/DIVER/DIVER_PRETRAINING/Aug_polaris_scale_temp/EEG_all_50M_notoptimal/trial_1_lr2.86e-03_wd6.93e-07/epoch01_step32506/mp_rank_00_model_states.pt
# run float64 to 32 for polaris (resume check)
# original lr is 6.73e-4
# optuna check lr is 1.80e-4, wd is 7.64e-5

# --foundation_dir "/global/cfs/cdirs/m4727/DIVER/DIVER_PRETRAINING/ABLATION/modality/iEEG_and_EEG/epoch8/iEEG_and_EEG/layers12/DIVER_iEEG_FINAL_model-dmodel512_layers12-N4_B96/lr2.61e-03_wd1.36e-03_usemup1/epoch07_step18025/mp_rank_00_model_states.pt" \
# multi_head_take_org_x

# 5M foundation_dir
# /global/cfs/cdirs/m4727/DIVER/DIVER_PRETRAINING/ABLATION/modality/iEEG_and_EEG/epoch32/iEEG_and_EEG/DIVER_iEEG_FINAL_model-dmodel256_layers12-N4_B960/lr2.61e-03_wd1.36e-03_usemup1/epoch31_step15040/mp_rank_00_model_states.pt

# 50M foundation_dir
# /global/cfs/cdirs/m4727/DIVER/DIVER_PRETRAINING/ABLATION/modality/iEEG_and_EEG/epoch8/iEEG_and_EEG/layers12/DIVER_iEEG_FINAL_model-dmodel512_layers12-N4_B96/lr2.61e-03_wd1.36e-03_usemup1/epoch07_step18025/mp_rank_00_model_states.pt


# /global/cfs/cdirs/m4727/DIVER/DIVER_PRETRAINING/ABLATION/modality/iEEG_V2/epoch32/iEEG_V2/DIVER_iEEG_FINAL_model-dmodel256_layers12-N4_B24/lr2.30e-03_wd2.17e-07_usemup1/epoch31_step28384/mp_rank_00_model_states.pt
#   50M lin
#    --lr 1.37e-04 \
#    --weight_decay 2.64e-03

#   50M full
#    --lr 7.59e-04 \
#    --weight_decay 1.87e-04

#   5M lin
#    --lr 1.03e-04 \
#    --weight_decay 1.15e-04

# Physio
#   50M lin
#    --lr 3.25e-03 \
#    --weight_decay 8.06e-05

#   50M full
#    --lr 6.03e-04 \
#    --weight_decay 1.87e-03

#   5M lin
#    --lr 6.44e-04 \
#    --weight_decay 6.25e-05

#   5M full
#    --lr 5.08e-04 \
#    --weight_decay 3.91e-03

# faced 50M 16 epoch full 45 run
# faced 2048 1epoch 41 43 run 42 re 44 ready
# 41 42 43 run 44 ready

# /pscratch/sd/t/tylee/Dataset/generalized_test/0709_modified/clipping_check/FACED_lowpass100_test/1.0_FACED/merged_resample-500_highpass-0.3_lowpass-100.0.lmdb 

# /global/cfs/cdirs/m4727/DIVER/DIVER_PRETRAINING/ABLATION/modality/epoch16/TUEG_and_iEEG/DIVER_iEEG_FINAL_model-dmodel512_layers12-N4_B96/lr2.61e-03_wd1.36e-03_usemup1/epoch15_step16240/bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt


#! SELECT MODEL TYPE (only thing to do)
MODEL_TYPE=patch50_512_ep64 #patch50_512_ep64
if [ $MODEL_TYPE == "patch50_256_ep64" ]
then
    #FOUNDATION_DIR="/data/ahhyun724/DIVER/model_weights/DIVER-1/patch50_256_ep64/epoch63_step106048/mp_rank_00_model_states.pt"
    FOUNDATION_DIR="/scratch/connectome/ahhyun724/DIVER/DIVER/CBraMod/tmp_model_weight/patch50_256_ep64/epoch63_step106048/mp_rank_00_model_states.pt"
elif [ $MODEL_TYPE == "patch50_512_ep64" ]
then
    #FOUNDATION_DIR="/data/ahhyun724/DIVER/model_weights/DIVER-1/patch50_512_ep64/epoch63_step106432/mp_rank_00_model_states.pt"
    FOUNDATION_DIR="/scratch/connectome/ahhyun724/DIVER/DIVER/CBraMod/tmp_model_weight/patch50_512_ep64/epoch63_step106048/mp_rank_00_model_states.pt"
fi

#/data/ahhyun724/DIVER/data/FACED
#/scratch/connectome/ahhyun724/DIVER/DIVER/CBraMod/tmp_data/FACED

python finetune_main.py \
    --seed 44 \
    --downstream_dataset FACED \
    --datasets_dir /scratch/connectome/ahhyun724/DIVER/DIVER/CBraMod/tmp_data/FACED \
    --model_dir ../../output_sample/$MODEL_TYPE/faced/seed44 \
    --foundation_dir $FOUNDATION_DIR \
    --backbone_config DIVER_iEEG_FINAL_model_patch50 \
    --cuda 0 \
    --feature_extraction_type multi_head_take_org_x \
    --use_optuna False \
    --ft_config flatten_linear \
    --width 256 \
    --depth 12 \
    --mup_weights True \
    --use_amp True \
    --deepspeed_pth_format True \
    --early_stop_criteria val_f1 \
    --early_stop_patience 50 \
    --frozen False \
    --precompute_features False \
    --lr 8.01e-04 \
    --weight_decay 9.03e-04 \
    --batch_size 1 \
    --num_workers 0
