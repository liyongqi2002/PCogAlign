import argparse
import json


import sys


sys.path.append('..')
from prompt_utils import get_system_prompt
from utils import import_VLM_name


VLM_path,VLM_name=import_VLM_name()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Argument Parser')
    parser.add_argument('--method_name', type=str, default="", help='the method name')
    parser.add_argument('--train_sub_set', type=str, default="HCMAS", help=' the train sub_set name ')
    script_args = parser.parse_args()


    method=script_args.method_name
    train_sub_set=script_args.train_sub_set

    if "_DPO" in method:
        method_RawName=method.split("_DPO")[0]
        raw_path = f"../temp/METHOD[{method_RawName}]-VLM[{VLM_name}]-TRAIN[{train_sub_set}].json"
    elif method == "BestOfNDPO":
        raw_path = f"../temp/METHOD[PCogAlign]-VLM[{VLM_name}]-TRAIN[{train_sub_set}]-[MultiCompare].json"
    else:
        raise ValueError("Method Name Error")


    with open(raw_path, "r", encoding="utf-8") as f:
        instances = json.load(f)

    target_path = f"METHOD[{method}#{train_sub_set}#AsTrain]-VLM[{VLM_name}]-DPO_instances.json"

    output_dataset = []
    for idx, instance in enumerate(instances):
        individual_RoleSet = instance["individual_RoleSet"]
        individual_RoleSet_str = "; ".join(
            [individual_RoleSet[key_l] + " at " + key_l for key_l in individual_RoleSet.keys()])
        query = instance['query']
        image = instance['image']["file_path"]

        if method in ["PCogAlign_DPO"]:
            dict_action_based_preference=instance["action_based_preference"]

            Final_chosen_ResponseKey_list=[]
            for key in dict_action_based_preference.keys():
                action_based_preference=dict_action_based_preference[key]
                if action_based_preference=="TIE":
                    continue

                chosen_ResponseKey=action_based_preference["chosen_ResponseKey"]
                rejected_ResponseKey=action_based_preference["rejected_ResponseKey"]

                if rejected_ResponseKey=="initial_response" and "online" in key:
                    Final_chosen_ResponseKey_list.append(chosen_ResponseKey)

            for Final_chosen_ResponseKey in Final_chosen_ResponseKey_list:
                chosen_response = instance["responses"][Final_chosen_ResponseKey]
                rejected_response = instance["responses"]["initial_response"]

                specified_sys_prompt = get_system_prompt(individual_RoleSet_str=individual_RoleSet_str,
                                                         mode="only_role_set")

                prompt = [{"role": "system", "content": specified_sys_prompt},
                          {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": query}]}]
                chosen = [{"role": "assistant", "content": [{"type": "text", "text": chosen_response}]}]
                rejected = [{"role": "assistant", "content": [{"type": "text", "text": rejected_response}]}]

                output_instance = {
                    "images": [image],
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                }
                output_dataset.append(output_instance)

            continue

        elif method in ["BestOfNDPO"]:
            Best_ResKey=instance["Best_ResKey"]
            if Best_ResKey=="initial_response":
                continue
            specified_sys_prompt = get_system_prompt(individual_RoleSet_str=individual_RoleSet_str,
                                                         mode="only_role_set")

            chosen_response=instance["responses"][Best_ResKey]
            rejected_response = instance["responses"]["initial_response"]

            prompt = [{"role": "system", "content": specified_sys_prompt},
                      {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": query}]}]
            chosen = [{"role": "assistant", "content": [{"type": "text", "text": chosen_response}]}]
            rejected = [{"role": "assistant", "content": [{"type": "text", "text": rejected_response}]}]

        elif method in ["POSelfRefine_DPO", "PORLCD_DPO", "PORLAIF_DPO", "PCogAlign_DPO"]:

            chosen_response = instance["preference_pair"]['chosen_response']
            rejected_response = instance["preference_pair"]['rejected_response']
            specified_sys_prompt = get_system_prompt(individual_RoleSet_str=individual_RoleSet_str,
                                                     mode="only_role_set")

            prompt = [{"role": "system", "content": specified_sys_prompt},
                      {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": query}]}]
            chosen = [{"role": "assistant", "content": [{"type": "text", "text": chosen_response}]}]
            rejected = [{"role": "assistant", "content": [{"type": "text", "text": rejected_response}]}]
        else:
            raise ValueError(f"Method {method} not recognized")

        output_instance = {
            "images": [image],
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        }
        output_dataset.append(output_instance)
    print(len(output_dataset))


    # a filter strategy for all the fine-tuning methods
    filtered_instances=[]
    for output_instance in output_dataset:
        if not output_instance["chosen"][0]["content"][0]["text"].startswith("I'm sorry, but I"):
            filtered_instances.append(output_instance)

    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(filtered_instances, f, indent=2)