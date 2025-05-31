import os
from platform import processor

import torch
from peft import PeftModel
from select import error
from transformers import AutoProcessor, AutoModelForVision2Seq

import os


def import_VLM_name():
    VLM_path = os.environ.get('VLM_PATH')
    VLM_name = os.environ.get('VLM_NAME')

    return VLM_path,VLM_name




def get_vllm_input(prompt, image_path, processor,specified_sys_prompt=None):
    from qwen_vl_utils import process_vision_info
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    if specified_sys_prompt is None:
        sys_prompt = "You are a helpful assistant."
    else:
        sys_prompt=specified_sys_prompt

    if image_path is None:
        conversation = [
            {"role": "system", "content": sys_prompt},
            {"role": "user","content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                ],
            }
        ]
    else:

        if "MiniCPM" in import_VLM_name()[1]:
            conversation = [
                {"role": "system", "content": sys_prompt},
                {
                    "role": "user",
                    "content": f"(<image>./</image>)\n{prompt}"
                }
            ]
        elif "InternVL" in import_VLM_name()[1]:
            sys_prompt=f"{sys_prompt}"
            conversation = [
                {"role": "system", "content": sys_prompt},
                {
                    "role": "user",
                    "content": f"<image>\n{prompt}"
                }
            ]
        elif "Phi-3.5" in import_VLM_name()[1]:
            sys_prompt=f"{sys_prompt}"
            conversation = [
                {"role": "system", "content": sys_prompt},
                {
                    "role": "user",
                    "content": f"<|image_1|>\n{prompt}"
                }
            ]
        else:
            conversation = [
                {"role": "system", "content": sys_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            # "image": image_path,
                            # "resized_height": 280, "resized_width": 420, # the default "dynamically adjusts image size, specify dimensions" from https://github.com/QwenLM/Qwen2-VL/tree/main/qwen-vl-utils
                        },
                        {
                            "type": "text",
                            "text": prompt
                        },
                    ],
                }
            ]

    processed_prompt = processor.apply_chat_template(conversation, tokenize=False,add_generation_prompt=True)
    # processed_prompt=prompt

    if image_path is None:
        vllm_input = {
            "prompt": processed_prompt,
            "multi_modal_data": {},
        }
    else:
        conversation_for_image_inputs = [
            {"role": "user", "content": [{"type": "image", "image": image_path,
                                          "resized_height": 280, "resized_width": 420}], }]
        image_inputs, _ = process_vision_info(conversation_for_image_inputs)

        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs

        vllm_input = {
            "prompt": processed_prompt,
            "multi_modal_data": mm_data,
        }
    return vllm_input



def prepare_vllm(model_path,use_lora=False,enable_prefix_caching=False,
                 gpu_memory_utilization=0.85,
                 max_tokens=512,temperature=0,):
    from vllm import LLM, SamplingParams

    # max_model_len = 8192

    if use_lora:
        llm = LLM(
            model=model_path,
            limit_mm_per_prompt={"image": 1},
            gpu_memory_utilization=gpu_memory_utilization,
            enable_lora=True,
            trust_remote_code=True,
            # max_model_len=max_model_len,
        )
    else:
        llm = LLM(
            model=model_path,
            limit_mm_per_prompt={"image": 1},
            gpu_memory_utilization=gpu_memory_utilization,
            enable_prefix_caching=enable_prefix_caching,
            trust_remote_code=True,
            # max_model_len=max_model_len,
        )



    sampling_params = SamplingParams(
            temperature=temperature,
            top_p=0.001,
            repetition_penalty=1.05,
            max_tokens=max_tokens,
            stop_token_ids=[],
    )

    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(model_path,trust_remote_code=True)

    if "MiniCPM" in model_path:
        # processor.chat_template =processor.tokenizer.chat_template
        processor.chat_template="{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        processor.image_token="(<image>./</image>)"
    if "Phi-3.5" in model_path:
        processor.chat_template =processor.tokenizer.chat_template

    return llm, sampling_params,processor

