# Copyright 2025 The HuggingFace Team. All rights reserved.
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
pip install pillow

# Tested on 8x H100 GPUs
accelerate launch
    --config_file=examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/scripts/sft_vlm.py \
    --dataset_name HuggingFaceH4/llava-instruct-mix-vsft \
    --model_name_or_path llava-hf/llava-1.5-7b-hf \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --output_dir ckpt_baseline_sft \
    --bf16 \
    --torch_dtype bfloat16 \
    --gradient_checkpointing

For LLaVA-NeXT, use: (requires transformers>=4.45)
    --model_name_or_path llava-hf/llava-v1.6-mistral-7b-hf

For meta-llama/Llama-3.2-11B-Vision-Instruct, use: (requires transformers>=4.45.1)
    --model_name_or_path meta-llama/Llama-3.2-11B-Vision-Instruct
"""
import json
import os

import numpy as np
import torch
import wandb
from datasets import load_dataset, Dataset
from numpy import mean
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

from transformers import AutoModelForVision2Seq, AutoProcessor, LlavaForConditionalGeneration, Qwen2VLProcessor, \
    Qwen2VLForConditionalGeneration, AutoModel, AutoModelForCausalLM,Qwen2_5_VLForConditionalGeneration

from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,

    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)



from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig, HfArgumentParser

from dataclasses import dataclass, field
from typing import Dict, Optional
# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """
    # training parameters
    method_name: Optional[str] = field(default="PO_RLCD_Thought",metadata={"help": "the method name"},)
    dataset_name: Optional[str] = field(default="HCMAS",metadata={"help": "the method name"},)

    output_dir: Optional[str] = field(default="ckpt",metadata={"help": "the ckpt"},)

    bench_root_path: Optional[str] = field(default="../PCogAlignBench/version_v4",
                                           metadata={"help": "bench_root_path"},)
    max_seq_length: Optional[int] = field(default=1024,metadata={"help": "the max_seq_length"},)

    num_train_epochs: Optional[int] = field(default=1,metadata={"help": "the num_train_epochs"},)


import sys
sys.path.append('..')
from utils import import_VLM_name
VLM_path,VLM_name=import_VLM_name()
# -VLM[{VLM_name}]

if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, ModelConfig))
    script_args, model_args = parser.parse_args_and_config()

    training_args = SFTConfig(
        output_dir=f"{script_args.output_dir}/{VLM_name}/METHOD[{script_args.method_name}#{script_args.dataset_name}#AsTrain]",  # directory to save and repository id

        num_train_epochs=script_args.num_train_epochs,  # number of training epochs
        # max_steps=,
        per_device_train_batch_size=4,  # batch size per device during training
        gradient_accumulation_steps=4,  # number of steps before performing a backward/update pass
        # gradient_checkpointing=True,  # use gradient checkpointing to save memory
        optim="adamw_torch_fused",  # use fused adamw optimizer
        # optim="paged_adamw_32bit",  # use fused adamw optimizer
        logging_steps=5,  # log every n steps
        # save_strategy="epoch",  # save checkpoint every epoch
        learning_rate=2e-4,  # learning rate, based on QLoRA paper
        bf16=True,  # use bfloat16 precision
        tf32=True,  # use tf32 precision
        max_grad_norm=0.3,  # max gradient norm based on QLoRA paper
        warmup_ratio=0.03,  # warmup ratio based on QLoRA paper
        lr_scheduler_type="cosine",  # use cosine learning rate scheduler
        push_to_hub=False,  # push model to hub
        report_to="wandb",  # report metrics to tensorboard
        gradient_checkpointing_kwargs={"use_reentrant": False},  # use reentrant checkpointing
        dataset_text_field="",  # need a dummy field for collator
        dataset_kwargs={"skip_prepare_dataset": True},  # important for collator
        max_seq_length=script_args.max_seq_length,
        dataset_num_proc=32,
    )
    training_args.remove_unused_columns = False



    ################
    # Model, Tokenizer & Processor
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
        # target_modules=["q_proj", "v_proj"],
        target_modules=["k_proj"],
        task_type="CAUSAL_LM",
    )


    # Apply PEFT model adaptation
    peft_model = get_peft_model(model, peft_config)
    # Print trainable parameters
    peft_model.print_trainable_parameters()

    ################
    # processor = Qwen2VLProcessor.from_pretrained(model_id)
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
    # Create a data collator to encode text and image pairs
    ################
    from qwen_vl_utils import process_vision_info
    # Create a data collator to encode text and image pairs
    def collate_fn(examples):
        # Get the texts and images, and apply the chat template
        texts = []
        image_inputs = []
        for example in examples:
            conversation_for_image_inputs = [
                {"role": "user", "content": [{"type": "image", "image": example["images"][0],
                                              "resized_height": 280, "resized_width": 420}], }]
            example_image_inputs, _ = process_vision_info(conversation_for_image_inputs)

            image_inputs.append(example_image_inputs)
            text=processor.apply_chat_template(example["messages"], tokenize=False)
            # print(text)
            # assert 1==0
            texts.append(text)


        # Tokenize the texts and process the images
        batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)


        # The labels are the input_ids, and we mask the padding tokens in the loss computation
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100  #
        # Ignore the image token index in the loss computation (model specific)
        if "Qwen2-VL" in VLM_name or "Qwen2.5-VL" in VLM_name:
            image_tokens = [151652, 151653, 151655]
        else:
            image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]
        for image_token_id in image_tokens:
            labels[labels == image_token_id] = -100
        batch["labels"] = labels

        return batch

    bench_root_path=script_args.bench_root_path
    SFT_instances_file_path=f"METHOD[{script_args.method_name}#{script_args.dataset_name}#AsTrain]-VLM[{VLM_name}]-SFT_instances.json"
    with open(SFT_instances_file_path,mode="r") as f:
        SFT_instances = json.load(f)

    def process(row):
        image=row["image"]
        image_path = f"{bench_root_path}/{image}"
        # print(image_path)

        if model.config.model_type in ["qwen2_vl","qwen2_5_vl"]:
            return {
                "images": [image_path],
                "messages": [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": row["specified_sys_prompt"]}],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                            },
                            {
                                "type": "text",
                                "text": row["input_text"],
                            },
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": row["output_text"]}],
                    },
                ],
            }
        elif model.config.model_type == "minicpmv":
            input_text=row["input_text"]
            return {
                "images": [image_path],
                "messages": [
                    {
                        "role": "system",
                        "content": row["specified_sys_prompt"],
                    },
                    {
                        "role": "user",
                        "content": f"(<image>./</image>)\n{input_text}",
                    },
                    {
                        "role": "assistant",
                        "content": row["output_text"],
                    },
                ],
            }
        elif model.config.model_type == "internvl_chat":
            input_text=row["input_text"]
            return {
                "images": [image_path],
                "messages": [
                    {
                        "role": "system",
                        "content": row["specified_sys_prompt"],
                    },
                    {
                        "role": "user",
                        "content": f"<image>\n{input_text}",
                    },
                    {
                        "role": "assistant",
                        "content": row["output_text"],
                    },
                ],
            }
        elif model.config.model_type == "phi3_v":
            input_text=row["input_text"]
            return {
                "images": [image_path],
                "messages": [
                    {
                        "role": "system",
                        "content": row["specified_sys_prompt"],
                    },
                    {
                        "role": "user",
                        "content": f"<|image_1|>\n{input_text}",
                    },
                    {
                        "role": "assistant",
                        "content": row["output_text"],
                    },
                ],
            }
        else:
            raise ValueError


    def get_tokens_num(row):
        text=row["specified_sys_prompt"]+row["input_text"]+row["output_text"]
        tokens = tokenizer.encode(text)
        tokens_num=len(tokens)
        # if tokens_num>512:
        #     print(text)
        return tokens_num


    tokens_nums=[]

    processed_rows=[]
    for SFT_instance in tqdm(SFT_instances,total=len(SFT_instances)):
        processed_row=process(SFT_instance)

        tokens_num=get_tokens_num(SFT_instance)
        tokens_nums.append(tokens_num)

        if processed_row:
            processed_rows.append(processed_row)

    ds = Dataset.from_list(processed_rows)
    print(ds)
    print(max(tokens_nums))
    print(min(tokens_nums))
    print(mean(tokens_num))


    model_id_name=model_id.split("/")[-1]
    script_args.run_name=f"METHOD[{script_args.method_name}#{script_args.dataset_name}#AsTrain]-VLM[{VLM_name}]"
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="trl_sft-PCogAlign-Training",
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
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=ds,
        eval_dataset=None,
        tokenizer=processor.tokenizer,
        peft_config=peft_config,
    )

    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)