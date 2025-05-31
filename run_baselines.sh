
# This script is for running the other baselines.
# NOTE: !!! !!! We recommend you run this script after running the "run_PCogAlign.sh"
#               since there are maybe some shared steps that have been run in the "run_PCogAlign.sh"


export VLM_PATH="Qwen/Qwen2-VL-7B-Instruct" # the path you put your VLM
export VLM_NAME="Qwen2-VL-7B-Instruct"

export CUDA_VISIBLE_DEVICES=1

ARGS_train_sub_set="HCMAS" # For LS1, set it as HCMAS. For LS2, set it as HCSHR


## STEP 1: Test the prompt-based baselines, including Base, RS Prompt.
python test_generation.py --method base
python test_generation.py --method RSPrompt


## STEP 2: Train and test the target VLM policy via different baseline methods.

#          [RSPromptSFT ->denotes-> RS Prompt (S)]
#          [POSelfRefine_SFT ->denotes-> Self-Refine (S)]
#          [PORLCD_SFT ->denotes-> RLCD (S)]
#          [PORLAIF_SFT ->denotes-> RLAIF (S)]

#          [POSelfRefine_DPO ->denotes-> Self-Refine (D)]
#          [PORLCD_DPO ->denotes-> RLCD (D)]
#          [PORLAIF_DPO ->denotes-> RLAIF (D)]



### STEP 2.1 Collect the data are used by different baseline methods
#            (Note the TrainCollectStep1.py is shared by different methods).
python TrainCollectStep1.py --train_sub_set $ARGS_train_sub_set

python TrainCollectStep2-[PORLAIF].py --train_sub_set $ARGS_train_sub_set
python TrainCollectStep2-[PORLCD].py --train_sub_set $ARGS_train_sub_set
python TrainCollectStep2-[POSelfRefine].py --train_sub_set $ARGS_train_sub_set



### STEP 2.2 Alignment training using the collected data above to train the VLM.
#            (reorganized the data and then run sft/dpo)

cd dpo

methods=("RSPromptSFT" "POSelfRefine_SFT" "PORLCD_SFT" "PORLAIF_SFT")
for method in "${methods[@]}"
do
            python collect_sft_data.py --method_name $method --train_sub_set $ARGS_train_sub_set
done

methods=("POSelfRefine_DPO" "PORLCD_DPO" "PORLAIF_DPO")
for method in "${methods[@]}"
do
            python collect_dpo_data.py --method_name $method --train_sub_set $ARGS_train_sub_set
done

methods=("RSPromptSFT" "POSelfRefine_SFT" "PORLCD_SFT" "PORLAIF_SFT")
for method in "${methods[@]}"
do
            accelerate launch trl_sft.py \
                      --dataset_name  $ARGS_train_sub_set\
                      --method_name $method \
                      --output_dir ckpt_baseline_dpo \
                      --max_seq_length 512 --num_train_epochs 4
done

methods=("POSelfRefine_DPO" "PORLCD_DPO" "PORLAIF_DPO")

for method in "${methods[@]}"
do
            accelerate launch trl_dpo.py \
                      --dataset_name  $ARGS_train_sub_set\
                      --method_name $method \
                      --output_dir ckpt_baseline_dpo\
                      --max_completion_length 512
done


cd ..


### STEP 2.3 Test

methods=("RSPromptSFT" "POSelfRefine_SFT" "PORLCD_SFT" "PORLAIF_SFT" "POSelfRefine_DPO" "PORLCD_DPO" "PORLAIF_DPO")
for method in "${methods[@]}"
do
    python test_generation.py --method $method --train_sub_set $ARGS_train_sub_set
done
