# This script is for running our proposed PCogAlign for a quick check.
# Methods: PCogAlign(P), PCogAlign(D), PCogAlign(S), PCogAlign

# (We recommend you run the 4 variants togather since they share some similar data collection steps)

export VLM_PATH="Qwen/Qwen2-VL-7B-Instruct" # the path you put your VLM
export VLM_NAME="Qwen2-VL-7B-Instruct"

export CUDA_VISIBLE_DEVICES=1

ARGS_train_sub_set="HCMAS" # For LS1, set it as HCMAS. For LS2, set it as HCSHR


## STEP 1: Train the reward model (we call it "PCogAlign-FdSFT" in all the codes)
### STEP 1.1 Collect the data used for reward model training
python TrainCollectStep1.py --train_sub_set $ARGS_train_sub_set
python TrainCollectStep2-[PCogAlign-FgdSFT].py --train_sub_set $ARGS_train_sub_set


### STEP 1.2 Reward model training
cd dpo
methods=("PCogAlign-FdSFT")
for method in "${methods[@]}"
do
      python collect_sft_data.py --method_name $method --train_sub_set $ARGS_train_sub_set
done

methods=("PCogAlign-FdSFT")
for method in "${methods[@]}"
do
           accelerate launch trl_sft.py \
                     --dataset_name  $ARGS_train_sub_set\
                     --method_name $method \
                     --output_dir ckpt_baseline_dpo \
                     --max_seq_length 2048 \
                     --num_train_epochs 6
done

cd ..


## STEP 2: The iterative process of collecting the "Best-of-N" samplings (using the trained reward model as a judge)
#          that are used for training the target policy
iters=(0 1 2 3 4)
for iter in "${iters[@]}"
do
   python TrainCollectStep3-[PCogAlign]-[FeedbackAndResample].py --online_iter $iter --train_sub_set $ARGS_train_sub_set
   python TrainCollectStep3-[PCogAlign]-[Reward].py --online_iter $iter --train_sub_set $ARGS_train_sub_set
done

python TrainCollectStep3-[PCogAlign]-[MultiCompare].py --train_sub_set $ARGS_train_sub_set




## STEP 3: Training the target VLM policy via different approaches
#          [PCogAlign_DPO ->denotes-> PCogAlign(D)]
#          [PCogAlign_SFT ->denotes-> PCogAlign(S)]
#          [BestOfNSFT ->denotes-> PCogAlign]

### STEP 3.1 Re-organize the data used for different alignment training approaches
cd dpo

methods=("BestOfNSFT" "PCogAlign_SFT")
for method in "${methods[@]}"
do
            python collect_sft_data.py --method_name $method --train_sub_set $ARGS_train_sub_set
done


methods=("PCogAlign_DPO")
for method in "${methods[@]}"
do
            python collect_dpo_data.py --method_name $method --train_sub_set $ARGS_train_sub_set
done


### STEP 3.2 Training and test the target policy using the reorganized training  data.
#### STEP 3.2.1 Training and test using "BestOfNSFT" [BestOfNSFT ->denotes-> PCogAlign]
methods=("BestOfNSFT")
for method in "${methods[@]}"
do
            accelerate launch trl_sft.py \
                      --dataset_name  $ARGS_train_sub_set\
                      --method_name $method \
                      --output_dir ckpt_baseline_dpo \
                      --max_seq_length 512 --num_train_epochs 4
done

cd ..

methods=("BestOfNSFT")
for method in "${methods[@]}"
do
    python test_generation.py --method $method --train_sub_set $ARGS_train_sub_set
done



#### STEP 3.2.2 Training and test using "PCogAlign_SFT" [PCogAlign_SFT ->denotes-> PCogAlign(S)]
cd dpo
methods=("PCogAlign_SFT")
for method in "${methods[@]}"
do
            accelerate launch trl_sft.py \
                      --dataset_name  $ARGS_train_sub_set\
                      --method_name $method \
                      --output_dir ckpt_baseline_dpo \
                      --max_seq_length 512 --num_train_epochs 4
done
cd ..

methods=("PCogAlign_SFT")
for method in "${methods[@]}"
do
    python test_generation.py --method $method --train_sub_set $ARGS_train_sub_set
done


#### STEP 3.2.3 Training and test using "PCogAlign_DPO" [PCogAlign_DPO ->denotes-> PCogAlign(D)]
cd dpo
methods=("PCogAlign_DPO")
for method in "${methods[@]}"
do
            accelerate launch trl_dpo.py \
                      --dataset_name  $ARGS_train_sub_set\
                      --method_name $method \
                      --output_dir ckpt_baseline_dpo\
                      --max_completion_length 512
done

cd ..

methods=("PCogAlign_DPO")
for method in "${methods[@]}"
do
    python test_generation.py --method $method --train_sub_set $ARGS_train_sub_set
done



## STEP 4: Directly test the prompt version of the PCogAlign, i.e., PCogAlign(P)
python test_generation_PCogAlign_Prompt.py
