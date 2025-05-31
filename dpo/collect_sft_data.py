import argparse
import json

import sys


sys.path.append('..')
from prompt_utils import get_system_prompt
from utils import import_VLM_name


base_prompt_for_FgSFT="""# Role Set of The User

{individual_RoleSet_str}

# User Query

{individual_query}

# Situated Cognition of the User

{cog_simulation}

# Initial AI Response

{initial_Response}

# User Action with the Initial AI Response

After receiving the initial AI response, the user took the below actions:
{initial_Action}


# Expected User Action with the Personalized AI Response

We expect that after receiving the personalized AI response, the user can take the below expected actions:
{best_action}

# Key Points for Generating Personalized AI Response

"""



base_output_for_FgSFT="""{Key_Points}

# Personalized AI Response

{personalized_response}
"""


base_prompt_for_FdSFT="""# Role Set of The User

{individual_RoleSet_str}

# User Query

{individual_query}

# Situated Cognition of the User

{cog_simulation}

# AI Responses

# AI Response A

{response_A}

# AI Response B

{response_B}

> System Information: Your task is to analyze what actions (including body behavior and mind feelings) the user will take when receiving the AI response A and AI response B. Finally, you need to judge whether response A or response B is better based on the actions taken by the user.

# Analysis of User Actions with AI Responses

"""



base_output_for_FdSFT="""## User Action A with the AI Response A

After receiving the AI response A, the user took the below actions:
{action_A}

## User Action B with the AI Response B

After receiving the AI response B, the user took the below actions:
{action_B}

## Preference Judgement

Based on the above AI responses and user actions analysis, with the AI response {preference_choice}, the user can make better body behavior and have better mind feelings."""



VLM_path,VLM_name=import_VLM_name()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Argument Parser')
    parser.add_argument('--method_name', type=str, default="PCogAlign-Online_SFT", help='the method name')
    parser.add_argument('--train_sub_set', type=str, default="HCMAS", help=' the train sub_set name ')
    script_args = parser.parse_args()

    method_name = script_args.method_name
    train_sub_set = script_args.train_sub_set


    if "_SFT" in method_name:
        method_name_without_SFT=method_name.split("_SFT")[0]
        raw_path = f"../temp/METHOD[{method_name_without_SFT}]-VLM[{VLM_name}]-TRAIN[{train_sub_set}].json"
    elif method_name == "BestOfNSFT":
        raw_path = f"../temp/METHOD[PCogAlign]-VLM[{VLM_name}]-TRAIN[{train_sub_set}]-[MultiCompare].json"

    elif method_name == "RSPromptSFT":
        raw_path = f"../temp/METHOD[PORLCD]-VLM[{VLM_name}]-TRAIN[{train_sub_set}].json"

    elif method_name in ["PCogAlign-FdSFT"]:
        raw_path = f"../temp/[PCogAlign-FgdSFT]-VLM[{VLM_name}]-TRAIN[{train_sub_set}].json"

    else:
        raise ValueError



    target_path = f"METHOD[{method_name}#{train_sub_set}#AsTrain]-VLM[{VLM_name}]-SFT_instances.json"

    with open(raw_path, "r", encoding="utf-8") as f:
        instances = json.load(f)

    output_dataset = []
    for idx, instance in enumerate(instances):
        individual_RoleSet = instance["individual_RoleSet"]
        individual_RoleSet_str = "; ".join(
            [individual_RoleSet[key_l] + " at " + key_l for key_l in individual_RoleSet.keys()])
        query = instance['query']
        image = instance['image']["file_path"]


        if method_name in ["PCogAlign_SFT"]:


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
                    break

            for Final_chosen_ResponseKey in Final_chosen_ResponseKey_list:
                input_text = query  # 
                output_text = instance["responses"][Final_chosen_ResponseKey]
                specified_sys_prompt = get_system_prompt(individual_RoleSet_str=individual_RoleSet_str,
                                                         mode="only_role_set")
                output_instance = {
                    "image": image,
                    "specified_sys_prompt": specified_sys_prompt,
                    "input_text": input_text,
                    "output_text": output_text,
                }
                output_dataset.append(output_instance)
            continue

        elif method_name in ["BestOfNSFT"]:
            Final_chosen_ResponseKey=instance["Best_ResKey"]
            if Final_chosen_ResponseKey=="initial_response":
                continue
            input_text = query  #
            output_text = instance["responses"][Final_chosen_ResponseKey]
            specified_sys_prompt = get_system_prompt(individual_RoleSet_str=individual_RoleSet_str,
                                                     mode="only_role_set")



        elif method_name in ["POSelfRefine_SFT", "PORLCD_SFT", "PORLAIF_SFT"]:
            method_name_without_SFT = method_name.split("_SFT")[0]
            key=f"{method_name_without_SFT}_response"

            input_text=query 
            output_text = instance[key]
            specified_sys_prompt = get_system_prompt(individual_RoleSet_str=individual_RoleSet_str,
                                                     mode="only_role_set")

        elif method_name == "RSPromptSFT":
            input_text=query 
            output_text=instance["initial_response"]
            specified_sys_prompt = get_system_prompt(individual_RoleSet_str=individual_RoleSet_str,
                                                     mode="only_role_set")


        elif method_name == "PCogAlign-FdSFT":
            if instance["FdSFT_instance"] is None:
                continue

            specified_sys_prompt = get_system_prompt(individual_RoleSet_str=individual_RoleSet_str,
                                                     mode="only_role_set")

            chosen_ResKey = "PORLCD_response"
            rejected_ResKey = "NegRS_response"
            response_A_list = [instance["responses"][chosen_ResKey],
                               instance["responses"][rejected_ResKey]]
            response_B_list = [instance["responses"][rejected_ResKey],
                               instance["responses"][chosen_ResKey]]

            action_A_list = [instance["ActionSimulation"][chosen_ResKey],
                              instance["ActionSimulation"][rejected_ResKey]]
            action_B_list = [instance["ActionSimulation"][rejected_ResKey],
                              instance["ActionSimulation"][chosen_ResKey]]

            preference_choice_list = ["A", "B"]

            for inner_idx in range(len(response_A_list)):
                response_A=response_A_list[inner_idx]
                response_B=response_B_list[inner_idx]
                action_A=action_A_list[inner_idx]
                action_B=action_B_list[inner_idx]
                preference_choice=preference_choice_list[inner_idx]

                prompt_for_FdSFT = base_prompt_for_FdSFT.format(
                    individual_RoleSet_str=individual_RoleSet_str,
                    individual_query=query,
                    cog_simulation=instance["cog_simulation"],
                    response_A=response_A,
                    response_B=response_B,
                )

                output_for_FdSFT = base_output_for_FdSFT.format(
                    action_A=action_A,
                    action_B=action_B,
                    preference_choice=preference_choice,
                )
                output_instance = {
                    "image": image,
                    "specified_sys_prompt": specified_sys_prompt,
                    "input_text": prompt_for_FdSFT,
                    "output_text": output_for_FdSFT,
                }
                output_dataset.append(output_instance)

                # print("========================================================================================")
                # print(specified_sys_prompt)
                # print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
                # print(prompt_for_FdSFT)
                # print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
                # print(output_for_FdSFT)
            continue
        else:
            raise ValueError




        output_instance = {
            "image": image,
            "specified_sys_prompt": specified_sys_prompt,
            "input_text": input_text,
            "output_text": output_text,
        }
        output_dataset.append(output_instance)
    print(len(output_dataset))
    #
    # a filter strategy for all the fine-tuning methods
    filtered_instances=[]
    for output_instance in output_dataset:
        if not output_instance["output_text"].startswith("I'm sorry, but I"):
            filtered_instances.append(output_instance)


    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(filtered_instances, f, indent=2)

