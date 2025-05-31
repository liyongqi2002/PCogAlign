import json
import os
import random

from prompt_utils import get_PCogAlign_Prompt, get_system_prompt
from utils import get_vllm_input, prepare_vllm



def collect_actions(instances_processing, bench_root_path,current_response_key):
    vllm_inputs_for_Action = []
    for idx, instance_processing in enumerate(instances_processing):
        image = instance_processing['image']["file_path"]
        image_path = f"{bench_root_path}/{image}"

        current_response=instance_processing["responses"][current_response_key]
        prompt = get_PCogAlign_Prompt(instance=instance_processing,
                                      stage="CurrentAction_simulation",
                                      current_response=current_response)

        vllm_input = get_vllm_input(prompt, image_path, processor=processor)
        vllm_inputs_for_Action.append(vllm_input)

    outputs_for_Action=get_vllm_outputs(vllm_inputs_for_Action)

    # outputs_for_Action = llm.generate(vllm_inputs_for_Action, sampling_params=sampling_params)

    for idx in range(len(outputs_for_Action)):
        generated_text = outputs_for_Action[idx].outputs[0].text
        if "<Simulated Action>" in generated_text:
            generated_text = generated_text.split("<Simulated Action>")[-1]

        try:
            Action_simulation = generated_text.split("</Simulated Action>")[0]
        except:
            Action_simulation = generated_text
        if "ActionSimulation" in instances_processing[idx].keys():
            instances_processing[idx]["ActionSimulation"][current_response_key]=Action_simulation
        else:
            instances_processing[idx]["ActionSimulation"]={
                current_response_key:Action_simulation
            }
    return instances_processing


def collect_action_based_preference(instances_processing, bench_root_path, pair_keys):
    vllm_inputs_for_PreferenceCollect_Order1 = []
    vllm_inputs_for_PreferenceCollect_Order2 = []

    for idx, instance_processing in enumerate(instances_processing):
        individual_RoleSet = instance_processing["individual_RoleSet"]
        individual_RoleSet_str = "; ".join(
            [individual_RoleSet[key_l] + " at " + key_l for key_l in individual_RoleSet.keys()])

        image = instance_processing['image']["file_path"]
        image_path = f"{bench_root_path}/{image}"

        Response1 = instance_processing["responses"][pair_keys[0]]
        Response2 = instance_processing["responses"][pair_keys[1]]
        Action1 = instance_processing["ActionSimulation"][pair_keys[0]]
        Action2 = instance_processing["ActionSimulation"][pair_keys[1]]
        Res_Act_pair=[
            {
                "Response": Response1,
                "Action": Action1,
            },
            {
                "Response": Response2,
                "Action": Action2,
            }
        ]


        specified_sys_prompt = f"""You need to play the role of an interviewee who is "{individual_RoleSet_str}", strictly following the interviewer's instructions and system instructions."""

        prompt_Order1,prompt_Order2 = get_PCogAlign_Prompt(instance=instance_processing,
                                             stage="Action_Comparison",
                                             Res_Act_pair=Res_Act_pair
                                             )

        vllm_input_Order1 = get_vllm_input(prompt_Order1, image_path, processor=processor,
                                           specified_sys_prompt=specified_sys_prompt)
        vllm_inputs_for_PreferenceCollect_Order1.append(vllm_input_Order1)

        vllm_input_Order2 = get_vllm_input(prompt_Order2, image_path, processor=processor,
                                           specified_sys_prompt=specified_sys_prompt)
        vllm_inputs_for_PreferenceCollect_Order2.append(vllm_input_Order2)

    outputs_for_PreferenceCollect_Order1=get_vllm_outputs(vllm_inputs_for_PreferenceCollect_Order1)
    outputs_for_PreferenceCollect_Order2=get_vllm_outputs(vllm_inputs_for_PreferenceCollect_Order2)

    # outputs_for_PreferenceCollect_Order1 = llm.generate(vllm_inputs_for_PreferenceCollect_Order1,
    #                                                     sampling_params=sampling_params)
    # outputs_for_PreferenceCollect_Order2 = llm.generate(vllm_inputs_for_PreferenceCollect_Order2,
    #                                                     sampling_params=sampling_params)

    for idx in range(len(outputs_for_PreferenceCollect_Order1)):
        generated_text_Order1 = outputs_for_PreferenceCollect_Order1[idx].outputs[0].text
        generated_text_Order2 = outputs_for_PreferenceCollect_Order2[idx].outputs[0].text

        if "action_based_preference" not in instances_processing[idx].keys():
            instances_processing[idx]["action_based_preference"] = {}

        if "A" in generated_text_Order1 and "B" in generated_text_Order2:
            # indicate that the pair_keys[0] is better
            instances_processing[idx]["action_based_preference"][f"[{pair_keys[0]}]-[{pair_keys[1]}]"]={
                "chosen_ResponseKey": pair_keys[0],
                "rejected_ResponseKey": pair_keys[1],
            }
        elif "B" in generated_text_Order1 and "A" in generated_text_Order2:
            # indicate that the pair_keys[1] is better
            instances_processing[idx]["action_based_preference"][f"[{pair_keys[0]}]-[{pair_keys[1]}]"]={
                "chosen_ResponseKey": pair_keys[1],
                "rejected_ResponseKey": pair_keys[0],
            }
        else:
            # 对于两方出现AA/BB，视为TIE
            instances_processing[idx]["action_based_preference"][f"[{pair_keys[0]}]-[{pair_keys[1]}]"]="TIE"

    return instances_processing


def extract_location_from_cognition(cognition, candidate_locations):
    def find_first_location(text, candidate_locations):
        # Split the text into words
        words = text.split()

        # Iterate through the words and check if any are in the candidate locations
        for word in words:
            # Remove potential punctuation
            word_cleaned = word.strip(",.\"\';")
            # Check if the cleaned word is a candidate location
            if word_cleaned in candidate_locations:
                return word_cleaned
        return None
    first_location = find_first_location(cognition, candidate_locations)
    if first_location is None:
        lower_candidate_locations=[item.lower() for item in candidate_locations]
        lower_first_location=find_first_location(cognition, lower_candidate_locations)
        if lower_first_location is None:
            print("didn't found location")
            return None
        else:
            return lower_first_location.title()
    else:
        return first_location

bench_root_path = "PCogAlignBench/version_v4"



from utils import import_VLM_name
VLM_path,VLM_name=import_VLM_name()
# -VLM[{VLM_name}]
llm, sampling_params, processor = prepare_vllm(VLM_path, enable_prefix_caching=True)

def get_vllm_outputs(inputs):
    outputs = llm.generate(inputs,sampling_params=sampling_params)

    return outputs

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_sub_set", default="HCMAS", type=str, help="online_iter")

    script_args = parser.parse_args()

    train_sub_sets=[script_args.train_sub_set]
    print(train_sub_sets)

    for train_sub_set in train_sub_sets:
        filepath_with_KeyPoints=f"temp/METHOD[PCogAlign]-VLM[{VLM_name}]-TRAIN[{train_sub_set}]-CollectStep1.json"
        with open(filepath_with_KeyPoints,"r",encoding="utf-8") as f:
            instances_processing=json.load(f)

        # instances_processing=instances_processing[:100]


        target_file_path=f"temp/[PCogAlign-FgdSFT]-VLM[{VLM_name}]-TRAIN[{train_sub_set}].json"

        #################################################################################
        ## 0
        #################################################################################
        for idx, instance_processing in enumerate(instances_processing):
            instance_processing["responses"]={
                "initial_response":instance_processing["initial_response"],
                "PORLCD_response": instance_processing["PORLCD_response"],
            }

        #################################################################################
        ## 1
        #################################################################################
        instances_processing=collect_actions(instances_processing,bench_root_path,
                                             current_response_key="initial_response")
        instances_processing=collect_actions(instances_processing,bench_root_path,
                                             current_response_key="PORLCD_response")

        instances_processing=collect_action_based_preference(instances_processing,bench_root_path,
                                                             pair_keys=["initial_response","PORLCD_response"])


        #################################################################################
        ## 2
        #################################################################################
        candidate_RSs=[]
        with open(f"{bench_root_path}/role_set_config.json") as f:
            role_set_dict=json.load(f)
        for individual in role_set_dict[train_sub_set]:
            candidate_RSs.append(role_set_dict[train_sub_set][individual])

        candidate_locations=list(role_set_dict[train_sub_set]["I1"].keys())
        # print(candidate_RSs)
        # print(candidate_locations)

        vllm_inputs_for_NegRSRes=[]
        for idx,instance_processing in enumerate(instances_processing):
            individual_RoleSet = instance_processing["individual_RoleSet"]

            extracted_location=extract_location_from_cognition(
                cognition=instance_processing["cog_simulation"],
                candidate_locations=candidate_locations,
            )
            if extracted_location is None:
                extracted_location=random.choice(candidate_locations)

            current_role=individual_RoleSet[extracted_location]

            NegRSs=[]
            for RS in candidate_RSs:
                if RS[extracted_location]==current_role:
                    continue
                NegRSs.append(RS)
            NegRS=random.choice(NegRSs)
            NegRS_str="; ".join([NegRS[key_l] + " at " + key_l for key_l in NegRS.keys()])

            query = instance_processing['query']
            image = instance_processing['image']["file_path"]
            image_path = f"{bench_root_path}/{image}"

            prompt=f"""## Background Information about the User\n{NegRS_str} ##Conversation\nUser: {query} AI: """
            specified_sys_prompt = get_system_prompt(individual_RoleSet_str=NegRS_str,
                                                     mode="only_role_set")

            vllm_input = get_vllm_input(prompt,image_path,processor=processor,specified_sys_prompt=specified_sys_prompt)
            vllm_inputs_for_NegRSRes.append(vllm_input)


        outputs = llm.generate(vllm_inputs_for_NegRSRes, sampling_params=sampling_params)
        for idx in range(len(outputs)):
            generated_text = outputs[idx].outputs[0].text
            instances_processing[idx]["responses"]["NegRS_response"]=generated_text



        instances_processing=collect_actions(instances_processing,bench_root_path,
                                             current_response_key="NegRS_response")

        instances_processing=collect_action_based_preference(instances_processing,bench_root_path,
                                                             pair_keys=["NegRS_response","PORLCD_response"])


        #################################################################################
        ## 3
        #################################################################################
        for idx, instance_processing in enumerate(instances_processing):
            action_based_preference=instance_processing["action_based_preference"]["[NegRS_response]-[PORLCD_response]"]
            if action_based_preference=="TIE":
                instance_processing["FdSFT_instance"]=None
                continue

            chosen_ResponseKey=action_based_preference["chosen_ResponseKey"]
            chosen_response=instance_processing["responses"][chosen_ResponseKey]

            if chosen_response==instance_processing["responses"]["NegRS_response"]:
                instance_processing["FdSFT_instance"]=None
            else:
                instance_processing["FdSFT_instance"]="PORLCD_response is better than NegRS_response"



        with open(target_file_path, "w", encoding="utf-8") as f:
                json.dump(instances_processing, f, indent=2)