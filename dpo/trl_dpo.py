# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
accelerate launch trl_dpo.py \
    --dataset_name HuggingFaceH4/rlaif-v_formatted \
    --model_name_or_path HCMAS-DPO_instances.json \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 32 \
    --dataset_num_proc 32 \
    --output_dir ckpt_trl_dpo \
    --bf16 \
    --torch_dtype bfloat16 \
    --gradient_checkpointing \
    --use_peft \
    --lora_target_modules=all-linear
"""
import json
import random
import sys

import torch
import wandb
from datasets import load_dataset, Dataset


from peft import get_peft_model, LoraConfig

from tqdm import tqdm
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig, HfArgumentParser, AutoModel, \
    AutoModelForCausalLM

from trl import (
    DPOConfig,
    # DPOTrainer,
    ModelConfig,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

from PIL import Image

from dpo_trainer import DPOTrainer

import torch
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor,Qwen2_5_VLForConditionalGeneration


from dataclasses import dataclass, field
from typing import Dict, Optional



# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """
    # training parameters
    method_name: Optional[str] = field(default="PO_RLCD",metadata={"help": "the method name"},)
    dataset_name: Optional[str] = field(default="HCMAS",metadata={"help": "the method name"},)

    output_dir: Optional[str] = field(default="ckpt",metadata={"help": "the ckpt"},)

    max_completion_length: Optional[int] = field(default=512,metadata={"help": "the max_completion_length"},)


    bench_root_path: Optional[str] = field(default="../PCogAlignBench/version_v4",
                                           metadata={"help": "bench_root_path"},)

    num_train_epochs: Optional[int] = field(default=1,metadata={"help": "the num_train_epochs"},)


sys.path.append('..')
from utils import import_VLM_name
VLM_path,VLM_name=import_VLM_name()

if __name__ == "__main__":
    parser = TrlParser((ScriptArguments,ModelConfig))
    script_args, model_args = parser.parse_args_and_config()

    training_args = DPOConfig(
        output_dir=f"{script_args.output_dir}/{VLM_name}/METHOD[{script_args.method_name}#{script_args.dataset_name}#AsTrain]",
        # directory to save and repository id

        num_train_epochs=script_args.num_train_epochs,
        # max_steps=,
        bf16=True,
        optim="adamw_torch_fused",  # use fused adamw optimizer
        # optim="paged_adamw_32bit",  # use fused adamw optimizer
        gradient_checkpointing=True,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        logging_steps=5,
        report_to="wandb",
        push_to_hub=False,
        save_total_limit=1,
        dataset_num_proc=32,  # tokenization will use 8 processes
        # max_length=1024,
        # max_prompt_length=,
        max_completion_length=script_args.max_completion_length,
    )



    print(training_args.output_dir)

    # print(training_args.max_length)

    ################
    # Model & Tokenizer
    ################
    # BitsAndBytesConfig int-4 config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )



    if "Qwen2-VL" in VLM_name:
        model_id = VLM_path
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
            # _attn_implementation="flash_attention_2",
            trust_remote_code=True
        )
        if "SFTDPO" in script_args.method_name:# 
            # adapter_path=f"{model_args.model_name_or_path}/Qwen2-VL-7B-Instruct/METHOD[{script_args.method_name}#{script_args.dataset_name}#AsTrain]"

            sft_method_name=script_args.method_name.split("DPO")[0]
            adapter_path=f"{script_args.output_dir}/{VLM_name}/METHOD[{sft_method_name}#{script_args.dataset_name}#AsTrain]"

            model.load_adapter(adapter_path)
            print("Pass! This is a DPO training based on the SFT phase ckpt.")

        else:
            # Not used generally
            print("Pass! This is a DPO training skipping SFT phase.")

    elif "Qwen2.5-VL" in VLM_name:
        model_id = VLM_path
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
            # _attn_implementation="flash_attention_2",
            trust_remote_code=True
        )
    elif "Phi-3.5" in VLM_name:
        model_id = VLM_path
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
            # _attn_implementation="flash_attention_2",
            trust_remote_code=True
        )


    else:
        model_id = VLM_path
        model = AutoModel.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
            # _attn_implementation="flash_attention_2",
            trust_remote_code=True
        )



    # Configure LoRA
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )




    # Apply PEFT model adaptation
    peft_model = get_peft_model(model, peft_config)
    # Print trainable parameters
    peft_model.print_trainable_parameters()

    ################
    ################
    processor = AutoProcessor.from_pretrained(model_id,trust_remote_code=True)

    try:
        tokenizer = processor.tokenizer
    except:
        raise ValueError


    # Set up the chat template
    if model.config.model_type == "minicpmv":
        processor.chat_template=tokenizer.chat_template
        # processor.chat_template= """{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"""
        processor.image_token="(<image>./</image>)"
    elif model.config.model_type == "phi3_v":
        processor.chat_template=tokenizer.chat_template
        processor.image_token="<|image_1|>"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


    ################
    # Dataset
    ################

    bench_root_path=script_args.bench_root_path
    DPO_instances_file_path=f"METHOD[{script_args.method_name}#{script_args.dataset_name}#AsTrain]-VLM[{VLM_name}]-DPO_instances.json"
    with open(DPO_instances_file_path,mode="r") as f:
        DPO_instances = json.load(f)

    random.seed(42)
    # random.shuffle(DPO_instances)
    # DPO_instances=DPO_instances[:3000]


    def process(row):
        image=row["images"][0]
        image_path = f"{bench_root_path}/{image}"
        row["images"][0]=image_path

        # conversation_for_image_inputs = [{"role": "user", "content": [{"type": "image", "image": image_path, }], }]
        # image_inputs, _ = process_vision_info(conversation_for_image_inputs)
        # row["images"]=image_inputs

        if len(tokenizer.encode(row["chosen"][0]["content"][0]["text"]+row["rejected"][0]["content"][0]["text"]))>1024:
            return None

        row["prompt"] = processor.apply_chat_template(row["prompt"], tokenize=False)
        row["chosen"] = processor.apply_chat_template(row["chosen"], tokenize=False)
        row["rejected"] = processor.apply_chat_template(row["rejected"], tokenize=False)
        return row

    # DPO_instances=DPO_instances[:100]


    processed_rows=[]
    for DPO_instance in tqdm(DPO_instances,total=len(DPO_instances)):
        processed_row=process(DPO_instance)
        if processed_row:
            processed_rows.append(processed_row)

    ds = Dataset.from_list(processed_rows)
    print(ds)


    ################
    # wandb Setting
    ################

    model_id_name=model_id.split("/")[-1]
    script_args.run_name=f"METHOD[{script_args.method_name}#{script_args.dataset_name}#AsTrain]-VLM[{VLM_name}]"
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="trl_dpo-PCogAlign-Training",
        # track hyperparameters and run metadata
        config={
            "script_args": script_args,
            "training_args": training_args,
            "model_args": model_args,
            "bnb_config": bnb_config,
            "peft_config": peft_config,

        },
        name=script_args.run_name,
    )


    ################
    # Training
    ################
    trainer = DPOTrainer(
        model,
        ref_model=None,
        args=training_args,
        train_dataset=ds,
        eval_dataset=None,
        processing_class=processor,
        peft_config=peft_config,
    )
    trainer.train()


    # Save and push to hub
    trainer.save_model(training_args.output_dir)