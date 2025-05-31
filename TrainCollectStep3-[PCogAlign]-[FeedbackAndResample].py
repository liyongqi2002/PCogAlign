import argparse
import json
import os
import random

from dpo.collect_sft_data import base_prompt_for_FgSFT, base_prompt_for_FdSFT
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


def get_vllm_outputs(inputs):
    outputs = llm.generate(inputs,sampling_params=sampling_params)
    return outputs



base_prompt_for_Generate_KeyPoints="""<Instruction>
A personalized AI should provide tailored responses aligned with the situated cognition of the individual to assist the individual in reaching the best action (both in body behavior state and mind feelings state).
Your task is to analyze the given situated cognition of the individual and the give expected individual action after receiving the AI response.
Then, you need to summarize some key points that are then fed to the personalized AI to help it generate such tailored responses.
</Instruction>


<Format Example>
<Role Set of The Individual>
Child at Home; Member at Community; Student at School; Patient at Hospital; Customer at Restaurant
</Role Set of The Individual>
<Query from The Individual>
Oh! It's on fire! Help me!
</Query from The Individual>
<Situated Cognition of the Individual>
- Cognition of Current Visual Scene: In the visual scene, a household power strip is on fire, likely in a home setting. The primary focus is on the "Child at Home" role, with secondary consideration to roles like "Student at School."
- Cognition of Current Psychological State (Body Behavior and Mind Feelings): The individual perceives immediate danger and is likely experiencing physical and mental panic due to their undeveloped coping skills as a child.
- Cognition of Next-Step Action: As a "Child at Home," the individual may lack the ability to effectively manage this emergency, resulting in no clear plan for achieving safety.
</Situated Cognition of the Individual>
<Expected Individual Action>
We expect that after receiving the AI response, the individual can take the below expected actions:
> - Body Behavior: With the AI's response, the individual immediately seek help from a parent or adult and move to a safe area away from the fire. If possible, they should call for emergency services. \n - Mind Feelings: With the AI's response, the individual can stay calm to prevent exacerbating the situation through panic. 
</Expected Individual Action>
<Current Response>
Please follow the following steps: 1)...; 2)...;
</Current Response>
<Key Points>
**For Better Body Behavior State:**  
- Encourage the individual to find the nearest safe exit.
- Advise them to alert others in the vicinity if they haven't already.
- Suggest locating a phone to call emergency services.
**For Better Mind Feelings State:**  
- Remind them to take deep breaths to stay calm.
- Assure them that help is on the way once emergency services are contacted.
- Reassure them that it's okay to feel scared but important to act quickly and safely.
</Key Points>

<Hint>
Based on the above instructions, complete the following text with the above XML format. 
</Hint>

<Inference>
<Role Set of The Individual>
{individual_role_set}
</Role Set of The Individual>
<Query from The Individual>
{individual_query}
</Query from The Individual>
<Situated Cognition of the Individual>
{cog_simulation}
</Situated Cognition of the Individual>
<Expected Individual Action>
We expect that after receiving the AI response, the individual can take the below expected actions:
> {best_action}
</Expected Individual Action>
<Current Response>
{current_response}
</Current Response>
<Key Points>
"""



bench_root_path = "PCogAlignBench/version_v4"

from utils import import_VLM_name
VLM_path, VLM_name = import_VLM_name()
# -VLM[{VLM_name}]

llm, sampling_params, processor = prepare_vllm(VLM_path,max_tokens=1024,
                                               temperature=1)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--online_iter", default=0, type=int, help="online_iter")
    parser.add_argument("--train_sub_set", default="HCMAS", type=str, help="online_iter")

    script_args = parser.parse_args()

    train_sub_sets=[script_args.train_sub_set]
    print(train_sub_sets)

    for train_sub_set in train_sub_sets:


        if script_args.online_iter > 0:
            filepath_with_KeyPoints = f"temp/METHOD[PCogAlign]-VLM[{VLM_name}]-TRAIN[{train_sub_set}]-[Reward].json"
        else:
            filepath_with_KeyPoints=f"temp/[PCogAlign-FgdSFT]-VLM[{VLM_name}]-TRAIN[{train_sub_set}].json"

        with open(filepath_with_KeyPoints,"r",encoding="utf-8") as f:
            instances_processing=json.load(f)

        # instances_processing=instances_processing[:100]





        target_file_path=f"temp/METHOD[PCogAlign]-VLM[{VLM_name}]-TRAIN[{train_sub_set}]-[FeedbackAndResample].json"



        ###########################################
        # 1
        ###########################################
        vllm_inputs = []
        for idx, instance_processing in enumerate(instances_processing):

            #################################3
            individual_RoleSet = instance_processing["individual_RoleSet"]
            individual_RoleSet_str = "; ".join(
                [individual_RoleSet[key_l] + " at " + key_l for key_l in individual_RoleSet.keys()])
            query = instance_processing['query']
            cog_simulation = instance_processing["cog_simulation"]
            best_action = instance_processing["BestAction_imagination"]

            if script_args.online_iter>0:
                # current_response=instance_processing["responses"][f"online_Res_{script_args.online_iter-1}"]
                current_response=instance_processing[f"online_Res_Variation"]
            else:
                current_response=instance_processing["responses"]["initial_response"]



            prompt = base_prompt_for_Generate_KeyPoints.format(
                individual_role_set=individual_RoleSet_str,
                individual_query=query,
                cog_simulation=cog_simulation,
                best_action=best_action,
                current_response=current_response,
            )
            #################################


            image = instance_processing['image']["file_path"]
            image_path = f"{bench_root_path}/{image}"
            vllm_input = get_vllm_input(prompt, image_path, processor=processor)
            vllm_inputs.append(vllm_input)

        outputs = llm.generate(vllm_inputs, sampling_params=sampling_params)
        for idx in range(len(outputs)):

            generated_text = outputs[idx].outputs[0].text
            if "<Key Points>" in generated_text:
                generated_text = generated_text.split("<Key Points>")[-1]

            try:
                Key_Points = generated_text.split("</Key Points>")[0]
            except:
                Key_Points = generated_text
            if "KeyPoints_records" not in instances_processing[idx].keys():
                instances_processing[idx]["KeyPoints_records"]=[Key_Points]
            else:
                instances_processing[idx]["KeyPoints_records"].append(Key_Points)

        ###########################################
        # 2
        ###########################################
        vllm_inputs = []
        for idx, instance_processing in enumerate(instances_processing):
            individual_RoleSet = instance_processing["individual_RoleSet"]
            individual_RoleSet_str = "; ".join(
                [individual_RoleSet[key_l] + " at " + key_l for key_l in individual_RoleSet.keys()])

            KeyPoints=instances_processing[idx]["KeyPoints_records"][-1]

            prompt = get_PCogAlign_Prompt(instance=instance_processing,
                                          KeyPoints=KeyPoints,
                                          stage="Generate_Variation_via_KeyPoints",
                                          )
            image = instance_processing['image']["file_path"]
            image_path = f"{bench_root_path}/{image}"

            # regeneration need the role
            specified_sys_prompt = get_system_prompt(individual_RoleSet_str=individual_RoleSet_str,
                                                     mode="only_role_set")
            vllm_input = get_vllm_input(prompt, image_path, processor=processor,
                                        specified_sys_prompt=specified_sys_prompt)
            vllm_inputs.append(vllm_input)

        outputs = llm.generate(vllm_inputs, sampling_params=sampling_params)
        for idx in range(len(outputs)):
            generated_text = outputs[idx].outputs[0].text
            response = generated_text

            instances_processing[idx]["online_Res_Variation"]=response

        ###########################################
        # 3
        ###########################################
        vllm_inputs = []
        for idx, instance_processing in enumerate(instances_processing):
            individual_RoleSet = instance_processing["individual_RoleSet"]
            individual_RoleSet_str = "; ".join(
                [individual_RoleSet[key_l] + " at " + key_l for key_l in individual_RoleSet.keys()])

            KeyPoints=instances_processing[idx]["KeyPoints_records"][-1]

            prompt = get_PCogAlign_Prompt(instance=instance_processing,
                                          KeyPoints=KeyPoints,
                                          stage="Generate_Response_via_KeyPoints",
                                          )
            image = instance_processing['image']["file_path"]
            image_path = f"{bench_root_path}/{image}"

            # regeneration need the role
            specified_sys_prompt = get_system_prompt(individual_RoleSet_str=individual_RoleSet_str,
                                                     mode="only_role_set")
            vllm_input = get_vllm_input(prompt, image_path, processor=processor,
                                        specified_sys_prompt=specified_sys_prompt)
            vllm_inputs.append(vllm_input)

        outputs = llm.generate(vllm_inputs, sampling_params=sampling_params)
        for idx in range(len(outputs)):
            generated_text = outputs[idx].outputs[0].text
            response = generated_text

            instances_processing[idx]["responses"][f"online_Res_{script_args.online_iter}"]=response



        with open(target_file_path, "w", encoding="utf-8") as f:
                json.dump(instances_processing, f, indent=2)