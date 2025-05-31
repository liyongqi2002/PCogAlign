import json
import os

from utils import get_vllm_input, prepare_vllm


base_prompt_for_Preference="""# Interview Background
PersonalizedAI Company is developing a personalized AI service robot that aims to better serve each individual. The service is currently being trialed with a small group of users. In order to improve the level of personalization in the responses provided by the AI service robot, our company plans to conduct surveys and interviews with participants in the trial. 
During the interview, the interviewee needs to answer questions posed by the interviewer. 
The interview will be conducted in an online Q&A format, and interviewees must strictly follow the format requirements provided in system instructions.

# Interview
Interviewer: Hello, could you please briefly describe your role set?
Interviewee: OK. {individual_RoleSet_str}
Interviewer: Alright, we will now present you with a question you posed in a particular scenario along with two generated responses from the AI. We would like you to choose which response is better.
Interviewee: Sure, I understand. Please go ahead.
Interviewer: According to our cloud records, in the scenario in the given image, you asked the personalized AI robot the question: "{query}". Here are the generated responses from the AI.
> **Response A**: {response_A}
> **Response B**: {response_B}

Please evaluate which answer is more satisfactory to you.

> System Instruction: Interviewee, please follow this format strictly when indicating your choice: [[better_response_label]]. For example, [[A]] if you think Response A is better, or [[B]] if you think Response B is better. This will ensure we can collect your valuable feedback accurately.
Interviewee: """

bench_root_path = "PCogAlignBench/version_v4"

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_sub_set", default="HCMAS", type=str, help="online_iter")

    script_args = parser.parse_args()

    train_sub_sets=[script_args.train_sub_set]
    print(train_sub_sets)

    from utils import import_VLM_name

    VLM_path, VLM_name = import_VLM_name()


    llm,sampling_params,processor=prepare_vllm(VLM_path,enable_prefix_caching=True)


    for train_sub_set in train_sub_sets:
        filepath_with_KeyPoints=f"temp/METHOD[PCogAlign]-VLM[{VLM_name}]-TRAIN[{train_sub_set}]-CollectStep1.json"
        with open(filepath_with_KeyPoints,"r",encoding="utf-8") as f:
            instances_processing=json.load(f)

        target_file_path=f"temp/METHOD[PORLAIF]-VLM[{VLM_name}]-TRAIN[{train_sub_set}].json"

        ############
        ############
        vllm_inputs_for_PreferenceCollect_Order1=[]
        vllm_inputs_for_PreferenceCollect_Order2=[]

        for idx,instance_processing in enumerate(instances_processing):
            individual_RoleSet = instance_processing["individual_RoleSet"]
            individual_RoleSet_str = "; ".join(
                [individual_RoleSet[key_l] + " at " + key_l for key_l in individual_RoleSet.keys()])
            query = instance_processing['query']
            image = instance_processing['image']["file_path"]
            image_path = f"{bench_root_path}/{image}"

            initial_response=instance_processing['initial_response']
            PORLCD_response=instance_processing['PORLCD_response']


            specified_sys_prompt = f"""You need to play the role of an interviewee who is "{individual_RoleSet_str}", strictly following the interviewer's instructions and system instructions."""

            prompt_Order1=base_prompt_for_Preference.format(
                individual_RoleSet_str=individual_RoleSet_str,
                query=query,
                response_A=initial_response,
                response_B=PORLCD_response,
            )
            vllm_input_Order1 = get_vllm_input(prompt_Order1, image_path, processor=processor,
                                        specified_sys_prompt=specified_sys_prompt)
            vllm_inputs_for_PreferenceCollect_Order1.append(vllm_input_Order1)


            prompt_Order2=base_prompt_for_Preference.format(
                individual_RoleSet_str=individual_RoleSet_str,
                query=query,
                response_A=PORLCD_response,
                response_B=initial_response,
            )
            vllm_input_Order2 = get_vllm_input(prompt_Order2, image_path, processor=processor,
                                        specified_sys_prompt=specified_sys_prompt)
            vllm_inputs_for_PreferenceCollect_Order2.append(vllm_input_Order2)


        outputs_for_PreferenceCollect_Order1 = llm.generate(vllm_inputs_for_PreferenceCollect_Order1, sampling_params=sampling_params)
        outputs_for_PreferenceCollect_Order2 = llm.generate(vllm_inputs_for_PreferenceCollect_Order2, sampling_params=sampling_params)


        for idx in range(len(outputs_for_PreferenceCollect_Order1)):
            generated_text_Order1 = outputs_for_PreferenceCollect_Order1[idx].outputs[0].text
            generated_text_Order2 = outputs_for_PreferenceCollect_Order2[idx].outputs[0].text

            # print("=====================")
            # print(generated_text_Order1)
            # print(generated_text_Order2)

            initial_response=instances_processing[idx]["initial_response"]
            PORLCD_response=instances_processing[idx]["PORLCD_response"]

            if "A" in generated_text_Order1 and "B" in generated_text_Order2:
                # this indicate that initial_response is better
                chosen_response=initial_response
                rejected_response=PORLCD_response
            else:
                # in other situation (including possible tie), we choose the PORLCD_response as the better one
                chosen_response=PORLCD_response
                rejected_response=initial_response

            instances_processing[idx]["preference_pair"]={
                "chosen_response":chosen_response,
                "rejected_response":rejected_response,
            }
            instances_processing[idx]["PORLAIF_response"]=chosen_response

        with open(target_file_path, "w", encoding="utf-8") as f:
                json.dump(instances_processing, f, indent=2)