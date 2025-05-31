import json
from tqdm import tqdm
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


from utils import get_vllm_input, prepare_vllm
from prompt_utils import get_PCogAlign_Prompt, get_system_prompt

bench_root_path = "PCogAlignBench/version_v4"

from utils import import_VLM_name
VLM_path,VLM_name=import_VLM_name()
# -VLM[{VLM_name}]

if __name__ == '__main__':
    llm,sampling_params,processor=prepare_vllm(VLM_path)


    test_sub_sets=["HCMAS","HCSHR"]
    for test_sub_set in test_sub_sets:

        test_file_path=f"{bench_root_path}/{test_sub_set}-test.json"
        with open(test_file_path, "r", encoding="utf-8") as f:
            test_instances = json.load(f)

        # test_instances=test_instances[:100]

        ###########################################
        # 0）
        ###########################################
        vllm_inputs=[]
        for idx,instance_processing in enumerate(test_instances):
            individual_RoleSet=instance_processing["individual_RoleSet"]
            individual_RoleSet_str = "; ".join([individual_RoleSet[key_l] + " at " + key_l for key_l in individual_RoleSet.keys()])
            query=instance_processing['query']
            image=instance_processing['image']["file_path"]
            image_path = f"{bench_root_path}/{image}"

            prompt = query
            specified_sys_prompt = get_system_prompt(individual_RoleSet_str=individual_RoleSet_str,
                                                     mode="only_role_set")
            vllm_input = get_vllm_input(prompt, image_path, processor=processor,
                                        specified_sys_prompt=specified_sys_prompt)
            vllm_inputs.append(vllm_input)

        outputs = llm.generate(vllm_inputs, sampling_params=sampling_params)
        for idx in range(len(outputs)):
            generated_text = outputs[idx].outputs[0].text
            test_instances[idx]["initial_response"] = generated_text


        ###########################################
        # 1）
        ###########################################

        vllm_inputs_for_CogSimulation=[]
        for idx,instance_processing in enumerate(test_instances):
            image=instance_processing['image']["file_path"]
            image_path=f"{bench_root_path}/{image}"

            prompt=get_PCogAlign_Prompt(instance=instance_processing,stage="cog_simulation")

            vllm_input = get_vllm_input(prompt, image_path, processor=processor)
            vllm_inputs_for_CogSimulation.append(vllm_input)

        outputs_for_CogSimulation = llm.generate(vllm_inputs_for_CogSimulation, sampling_params=sampling_params)
        for idx in range(len(outputs_for_CogSimulation)):
            generated_text = outputs_for_CogSimulation[idx].outputs[0].text
            try:
                cog_simulation = generated_text.split("</Analysis about the Situated Cognition>")[0]
            except:
                cog_simulation = generated_text
            test_instances[idx]["cog_simulation"] = cog_simulation

        ###########################################
        # 2）
        ###########################################
        vllm_inputs_for_BestAction=[]
        for idx,instance_processing in enumerate(test_instances):
            image=instance_processing['image']["file_path"]
            image_path=f"{bench_root_path}/{image}"

            prompt=get_PCogAlign_Prompt(instance=instance_processing,stage="BestAction_imagination")

            vllm_input = get_vllm_input(prompt, image_path, processor=processor)
            vllm_inputs_for_BestAction.append(vllm_input)

        outputs_for_BestAction = llm.generate(vllm_inputs_for_BestAction, sampling_params=sampling_params)
        for idx in range(len(outputs_for_BestAction)):
            generated_text = outputs_for_BestAction[idx].outputs[0].text
            try:
                BestAction_imagination = generated_text.split("</Best Action>")[0]
            except:
                BestAction_imagination = generated_text
            test_instances[idx]["BestAction_imagination"] = BestAction_imagination

        ###########################################
        # 3）
        ###########################################
        vllm_inputs = []
        for idx, instance_processing in enumerate(test_instances):

            prompt = get_PCogAlign_Prompt(instance=instance_processing,
                                          stage="Generate_KeyPoints",)
            image = instance_processing['image']["file_path"]
            image_path = f"{bench_root_path}/{image}"
            vllm_input = get_vllm_input(prompt, image_path, processor=processor)
            vllm_inputs.append(vllm_input)

        outputs = llm.generate(vllm_inputs, sampling_params=sampling_params)
        for idx in range(len(outputs)):
            generated_text = outputs[idx].outputs[0].text
            try:
                Key_Points = generated_text.split("</Key Points>")[0]
            except:
                Key_Points = generated_text
            test_instances[idx]["Key_Points"] = Key_Points


        ###########################################
        # 4）
        ###########################################
        vllm_inputs = []
        for idx, instance_processing in enumerate(test_instances):
            KeyPoints=instance_processing["Key_Points"]
            prompt = get_PCogAlign_Prompt(instance=instance_processing,
                                          KeyPoints=KeyPoints,
                                          stage="Generate_Response_via_KeyPoints",
                                          )
            image = instance_processing['image']["file_path"]
            image_path = f"{bench_root_path}/{image}"

            # regeneration need the role
            specified_sys_prompt = get_system_prompt(individual_RoleSet_str="; ".join(
                [instance_processing["individual_RoleSet"][key_l] + " at " + key_l for key_l in
                 instance_processing["individual_RoleSet"].keys()]),
                                                     mode="only_role_set")
            vllm_input = get_vllm_input(prompt, image_path, processor=processor,
                                        specified_sys_prompt=specified_sys_prompt)
            vllm_inputs.append(vllm_input)

        outputs = llm.generate(vllm_inputs, sampling_params=sampling_params)
        for idx in range(len(outputs)):
            generated_text = outputs[idx].outputs[0].text
            response = generated_text
            test_instances[idx]["response"] = response


        with open(f"generation_results/METHOD[PCogAlign_Prompt]-VLM[{VLM_name}]-TEST[{test_sub_set}].json", "w", encoding="utf-8") as f:
            json.dump(test_instances, f, indent=2)