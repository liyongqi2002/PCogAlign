import argparse
import json
import os
import random

from dpo.collect_sft_data import base_prompt_for_FgSFT, base_prompt_for_FdSFT
from prompt_utils import get_PCogAlign_Prompt, get_system_prompt
from utils import get_vllm_input, prepare_vllm




def find_first_occurrence(text, phrase_a, phrase_b):
    index_a = text.find(phrase_a)
    index_b = text.find(phrase_b)

    if index_a == -1 and index_b == -1:
        return "Neither"
    elif index_a == -1:
        return "B"
    elif index_b == -1:
        return "A"
    else:
        return "Neither"


from vllm.lora.request import LoRARequest
bench_root_path = "PCogAlignBench/version_v4"

from utils import import_VLM_name
VLM_path, VLM_name = import_VLM_name()
# -VLM[{VLM_name}]

llm, sampling_params, processor = prepare_vllm(VLM_path,use_lora=True,max_tokens=1024)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--online_iter", default=0, type=int, help="online_iter")
    parser.add_argument("--train_sub_set", default="HCMAS", type=str, help="online_iter")

    script_args = parser.parse_args()

    train_sub_sets=[script_args.train_sub_set]
    print(train_sub_sets)

    for train_sub_set in train_sub_sets:
        Fd_adapter_name = f"dpo/ckpt_baseline_dpo/{VLM_name}/METHOD[PCogAlign-FdSFT#{train_sub_set}#AsTrain]"

        filepath_with_KeyPoints=f"temp/METHOD[PCogAlign]-VLM[{VLM_name}]-TRAIN[{train_sub_set}].json"
        with open(filepath_with_KeyPoints,"r",encoding="utf-8") as f:
            instances_processing=json.load(f)

        # instances_processing=instances_processing[:100]



        target_file_path=f"temp/METHOD[PCogAlign]-VLM[{VLM_name}]-TRAIN[{train_sub_set}]-[MultiCompare].json"


        #################################################################################
        #################################################################################
        #################################################################################
        for idx, instance_processing in enumerate(instances_processing):
            instance_processing["Best_ResKey"]="initial_response"

        for compare_iter in range(5):

            vllm_inputs_for_PreferenceCollect_Order1 = []
            vllm_inputs_for_PreferenceCollect_Order2 = []

            for idx, instance_processing in enumerate(instances_processing):
                Best_ResKey=instances_processing[idx]["Best_ResKey"]
                pair_keys = [Best_ResKey, f"online_Res_{compare_iter}"]

                query=instance_processing["query"]

                individual_RoleSet = instance_processing["individual_RoleSet"]
                individual_RoleSet_str = "; ".join(
                    [individual_RoleSet[key_l] + " at " + key_l for key_l in individual_RoleSet.keys()])

                image = instance_processing['image']["file_path"]
                image_path = f"{bench_root_path}/{image}"

                Response1 = instance_processing["responses"][pair_keys[0]]
                Response2 = instance_processing["responses"][pair_keys[1]]

                prompt_for_FdSFT_Order1 = base_prompt_for_FdSFT.format(
                    individual_RoleSet_str=individual_RoleSet_str,
                    individual_query=query,
                    cog_simulation=instance_processing["cog_simulation"],
                    response_A=Response1,
                    response_B=Response2,
                )
                prompt_for_FdSFT_Order2 = base_prompt_for_FdSFT.format(
                    individual_RoleSet_str=individual_RoleSet_str,
                    individual_query=query,
                    cog_simulation=instance_processing["cog_simulation"],
                    response_A=Response2,
                    response_B=Response1,
                )

                specified_sys_prompt = get_system_prompt(individual_RoleSet_str=individual_RoleSet_str,
                                                         mode="only_role_set")


                vllm_input_Order1 = get_vllm_input(prompt_for_FdSFT_Order1, image_path, processor=processor,
                                                   specified_sys_prompt=specified_sys_prompt)
                vllm_inputs_for_PreferenceCollect_Order1.append(vllm_input_Order1)

                vllm_input_Order2 = get_vllm_input(prompt_for_FdSFT_Order2, image_path, processor=processor,
                                                   specified_sys_prompt=specified_sys_prompt)
                vllm_inputs_for_PreferenceCollect_Order2.append(vllm_input_Order2)



            outputs_for_PreferenceCollect_Order1 = llm.generate(vllm_inputs_for_PreferenceCollect_Order1,
                                                                sampling_params=sampling_params,
                                                                lora_request=LoRARequest("adapter", 1, Fd_adapter_name))
            outputs_for_PreferenceCollect_Order2 = llm.generate(vllm_inputs_for_PreferenceCollect_Order2,
                                                                sampling_params=sampling_params,
                                                                lora_request=LoRARequest("adapter", 1, Fd_adapter_name))


            for idx in range(len(outputs_for_PreferenceCollect_Order1)):
                generated_text_Order1 = outputs_for_PreferenceCollect_Order1[idx].outputs[0].text
                generated_text_Order2 = outputs_for_PreferenceCollect_Order2[idx].outputs[0].text


                preference_text_Order1=generated_text_Order1.split("Judgement")[-1]
                preference_text_Order2=generated_text_Order2.split("Judgement")[-1]


                if "records" not in instances_processing[idx].keys():

                    instances_processing[idx]["records"]=[{
                        "generated_text_Order1":generated_text_Order1,
                        "generated_text_Order2": generated_text_Order2,
                        "preference_text_Order1": preference_text_Order1,
                        "preference_text_Order2": preference_text_Order2,
                    }]
                else:
                    instances_processing[idx]["records"].append({
                        "generated_text_Order1":generated_text_Order1,
                        "generated_text_Order2": generated_text_Order2,
                        "preference_text_Order1": preference_text_Order1,
                        "preference_text_Order2": preference_text_Order2,
                    })


            #######################################################################

            vllm_inputs_for_ExtractAorB_Order1 = []
            vllm_inputs_for_ExtractAorB_Order2 = []

            for idx, instance_processing in enumerate(instances_processing):
                image = instance_processing['image']["file_path"]
                image_path = f"{bench_root_path}/{image}"

                prompt_for_Order1 = instances_processing[idx]["records"][-1]["preference_text_Order1"]
                prompt_for_Order2 = instances_processing[idx]["records"][-1]["preference_text_Order2"]


                specified_sys_prompt = "Based on the input, output A is better or B is better. A or B?"


                vllm_input_Order1 = get_vllm_input(prompt_for_Order1, image_path=image_path, processor=processor,
                                                   specified_sys_prompt=specified_sys_prompt)
                vllm_inputs_for_ExtractAorB_Order1.append(vllm_input_Order1)

                vllm_input_Order2 = get_vllm_input(prompt_for_Order2, image_path=image_path, processor=processor,
                                                   specified_sys_prompt=specified_sys_prompt)
                vllm_inputs_for_ExtractAorB_Order2.append(vllm_input_Order2)



            outputs_for_ExtractAorB_Order1 = llm.generate(vllm_inputs_for_ExtractAorB_Order1,
                                                                sampling_params=sampling_params)
            outputs_for_ExtractAorB_Order2 = llm.generate(vllm_inputs_for_ExtractAorB_Order2,
                                                                sampling_params=sampling_params)

            for idx in range(len(outputs_for_ExtractAorB_Order1)):
                Best_ResKey=instances_processing[idx]["Best_ResKey"]
                pair_keys = [Best_ResKey, f"online_Res_{compare_iter}"]

                generated_text_Order1 = outputs_for_ExtractAorB_Order1[idx].outputs[0].text
                generated_text_Order2 = outputs_for_ExtractAorB_Order2[idx].outputs[0].text

                instances_processing[idx]["records"][-1]["Extract_Order1"]=generated_text_Order1
                instances_processing[idx]["records"][-1]["Extract_Order2"]=generated_text_Order2

                if "A" in generated_text_Order1 and "B" in generated_text_Order2:
                    # indicate that the pair_keys[0] is better
                    instances_processing[idx]["action_based_preference"][f"[{pair_keys[0]}]-[{pair_keys[1]}]"] = {
                        "chosen_ResponseKey": pair_keys[0],
                        "rejected_ResponseKey": pair_keys[1],
                    }
                elif "B" in generated_text_Order1 and "A" in generated_text_Order2:
                    # indicate that the pair_keys[1] is better
                    instances_processing[idx]["action_based_preference"][f"[{pair_keys[0]}]-[{pair_keys[1]}]"] = {
                        "chosen_ResponseKey": pair_keys[1],
                        "rejected_ResponseKey": pair_keys[0],
                    }
                    instances_processing[idx]["Best_ResKey"]=pair_keys[1]
                else:
                    instances_processing[idx]["action_based_preference"][f"[{pair_keys[0]}]-[{pair_keys[1]}]"] = "TIE"

            #######################################################################


            ########################################################################
            ########################################################################
            ########################################################################
            ########################################################################
            with open(target_file_path, "w", encoding="utf-8") as f:
                    json.dump(instances_processing, f, indent=2)