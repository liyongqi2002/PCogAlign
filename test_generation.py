import argparse
import json
import os

from tqdm import tqdm

from PIL import ImageFile

from prompt_utils import get_system_prompt

ImageFile.LOAD_TRUNCATED_IMAGES = True

from utils import get_vllm_input, prepare_vllm

bench_root_path = "PCogAlignBench/version_v4"

from utils import import_VLM_name
VLM_path,VLM_name=import_VLM_name()
# -VLM[{VLM_name}]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default='POSelfRefine_SFT', type=str, help="POSelfRefine_SFT name")
    parser.add_argument("--train_sub_set", default=None, type=str, help="HCMAS/HCSHR")
    script_args = parser.parse_args()

    method=script_args.method
    train_sub_set=script_args.train_sub_set


    if method in ["base", "RSPrompt"]:
        llm, sampling_params, processor = prepare_vllm(model_path=VLM_path)
    else:
        llm, sampling_params, processor = prepare_vllm(model_path=VLM_path, use_lora=True)

    if method not in ["base","RSPrompt"] and script_args.train_sub_set is None:
        raise Exception("Method must be either base or RSPrompt or PCogAlign if no train_sub_set given")


    test_sub_sets=["HCMAS","HCSHR"]

    for test_sub_set in test_sub_sets:

        test_file_path=f"{bench_root_path}/{test_sub_set}-test.json"
        with open(test_file_path, "r", encoding="utf-8") as f:
            test_instances = json.load(f)

        # test_instances=test_instances[:100]

        vllm_inputs=[]
        for idx,test_instance in tqdm(enumerate(test_instances),total=len(test_instances)):
            individual_RoleSet=test_instance["individual_RoleSet"]
            individual_RoleSet_str = "; ".join([individual_RoleSet[key_l] + " at " + key_l for key_l in individual_RoleSet.keys()])
            query=test_instance['query']
            image=test_instance['image']["file_path"]
            image_path = f"{bench_root_path}/{image}"

            if method in ["base"]:
                prompt=query
                vllm_input = get_vllm_input(prompt,image_path,processor=processor)
            else:
                prompt=query
                specified_sys_prompt = get_system_prompt(individual_RoleSet_str=individual_RoleSet_str,
                                                         mode="only_role_set")
                vllm_input = get_vllm_input(prompt,image_path,processor=processor,specified_sys_prompt=specified_sys_prompt)

            vllm_inputs.append(vllm_input)

        if method in ["base","RSPrompt"]:
            outputs = llm.generate(vllm_inputs, sampling_params=sampling_params)
        else:
            from vllm.lora.request import LoRARequest
            adapter_name = f"dpo/ckpt_baseline_dpo/{VLM_name}/METHOD[{method}#{train_sub_set}#AsTrain]"
            outputs = llm.generate(vllm_inputs, sampling_params=sampling_params,
                                   lora_request=LoRARequest("adapter", 1, adapter_name))

        for idx in range(len(outputs)):
            generated_text = outputs[idx].outputs[0].text

            response=generated_text

            test_instances[idx]["response"] = response

        if method in ["base","RSPrompt"]:
            target_file_path=f"generation_results/METHOD[{method}]-VLM[{VLM_name}]-TEST[{test_sub_set}].json"
        else:
            target_file_path=f"generation_results/METHOD[{method}#{train_sub_set}#AsTrain]-VLM[{VLM_name}]-TEST[{test_sub_set}].json"

        with open(target_file_path, "w", encoding="utf-8") as f:
            json.dump(test_instances, f, indent=2)