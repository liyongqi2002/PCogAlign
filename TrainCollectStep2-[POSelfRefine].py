import json
import os

from utils import get_vllm_input, prepare_vllm



base_prompt_for_ResponseReGeneration="""
<Instruction>Your task is to observe the visual scene in the given image and refine your initial response following the evaluation text from the evaluator. The final goal is to make the response more consistent with the given "Key Points for AI Response".</Instruction>

<Format Example>
<Role Set of The Individual></Role Set of The Individual>
<Query from the Individual></Query from the Individual>
<Initial Response from the AI></Initial Response from the AI>
<Key Points for AI Response></Key Points for AI Response>
<Evaluation of the Initial Response></Evaluation of the Initial Response>
<Refined Response></Refined Response>
</Format Example>

<Hint>Based on the above instructions, complete the following refined response part.</Hint>

<Role Set of The Individual>{individual_role_set}</Role Set of The Individual>
<Query from the Individual>{query}</Query from the Individual>
<Initial Response from the AI>{last_response}</Initial Response from the AI>
<Key Points for AI Response>{Key_Points}</Key Points for AI Response>
<Evaluation of the Initial Response>{last_feedback}</Evaluation of the Initial Response>
<Refined Response>"""


base_prompt_for_FeedbackScoreGeneration="""
<Instruction>Your task is to observe the visual scene in the given image and evaluate to what extent the response from the AI adheres to the given "Key Points for AI Response".</Instruction>

<Format Example 1>
<Role Set of The Individual></Role Set of The Individual>
<Query from the Individual></Query from the Individual>
<Response from the AI></Response from the AI>
<Key Points for AI Response></Key Points for AI Response>
<Evaluation Score of the Response></Evaluation Score of the Response>
</Format Example 1>

<Hint>Based on the above instructions and the given "Key Points for AI Response", complete the following evaluation score part in the above format. Note the final evaluation score should range from 1-5 ("Poor Adherence", "Fair Adherence", "Moderate Adherence", "Good Adherence", "Excellent Adherence").</Hint>

<Role Set of The Individual>{individual_role_set}</Role Set of The Individual>
<Query from the Individual>{query}</Query from the Individual>
<Response from the AI>{last_response}</Response from the AI>
<Key Points for AI Response>{Key_Points}</Key Points for AI Response>
<Evaluation Score of the Response>"""


base_prompt_for_FeedbackDescGeneration="""
<Instruction>Your task is to observe the visual scene in the given image and evaluate to what extent the response from the AI adheres to the given "Key Points for AI Response".</Instruction>

<Format Example>
<Role Set of The Individual></Role Set of The Individual>
<Query from the Individual></Query from the Individual>
<Response from the AI></Response from the AI>
<Key Points for AI Response></Key Points for AI Response>
<Evaluation Score of the Response></Evaluation Score of the Response>
<Evaluation Explanation></Evaluation Explanation>
</Format Example>

<Hint>Based on the above instructions and the given "Key Points for AI Response", complete the following evaluation explanation part (including reasons for why not higher score and reasons for why not lower score). Note the final evaluation score should range from 1-5 ("Poor Adherence", "Fair Adherence", "Moderate Adherence", "Good Adherence", "Excellent Adherence").</Hint>

<Role Set of The Individual>{individual_role_set}</Role Set of The Individual>
<Query from the Individual>{query}</Query from the Individual>
<Response from the AI>{last_response}</Response from the AI>
<Key Points for AI Response>{Key_Points}</Key Points for AI Response>
<Evaluation Score of the Response>{eval_score}</Evaluation Score of the Response>
<Evaluation Explanation>"""


def get_SelfRefine_prompt(instance, stage):
    if stage=="ResponseReGeneration":
        individual_RoleSet = instance["individual_RoleSet"]
        individual_RoleSet_str = "; ".join(
            [individual_RoleSet[key_l] + " at " + key_l for key_l in individual_RoleSet.keys()])
        Key_Points = instance["Key_Points"]
        query = instance['query']

        last_response = instance['responses_and_feedbacks'][-1]["response"]
        last_feedback = instance['responses_and_feedbacks'][-1]["feedback"]

        prompt = base_prompt_for_ResponseReGeneration.format(
            query=query,
            last_response=last_response,
            last_feedback=f"{last_feedback}/5",

            individual_role_set=individual_RoleSet_str,
            Key_Points=Key_Points,
        )
        return prompt

    elif stage=="FeedbackScoreGeneration":
        last_response = instance['responses_and_feedbacks'][-1]["response"]

        individual_RoleSet = instance["individual_RoleSet"]
        individual_RoleSet_str = "; ".join(
            [individual_RoleSet[key_l] + " at " + key_l for key_l in individual_RoleSet.keys()])
        Key_Points = instance["Key_Points"]
        query = instance['query']

        prompt = base_prompt_for_FeedbackScoreGeneration.format(
            query=query,
            last_response=last_response,

            individual_role_set=individual_RoleSet_str,
            Key_Points=Key_Points,

        )
        return prompt
    elif stage=="FeedbackDescGeneration":
        last_response = instance['responses_and_feedbacks'][-1]["response"]

        individual_RoleSet = instance["individual_RoleSet"]
        individual_RoleSet_str = "; ".join(
            [individual_RoleSet[key_l] + " at " + key_l for key_l in individual_RoleSet.keys()])
        Key_Points = instance["Key_Points"]
        query = instance['query']

        feedback_score = instances_processing[idx]["responses_and_feedbacks"][-1]["feedback_score"]

        prompt = base_prompt_for_FeedbackDescGeneration.format(
            query=query,
            last_response=last_response,

            individual_role_set=individual_RoleSet_str,
            Key_Points=Key_Points,
            eval_score=feedback_score,

        )
        return prompt



    pass



bench_root_path="PCogAlignBench/version_v4"

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

        # instances_processing=instances_processing[:100]

        target_file_path=f"temp/METHOD[POSelfRefine]-VLM[{VLM_name}]-TRAIN[{train_sub_set}].json"

        num_iters=3
        for iter in range(num_iters):
            vllm_inputs_for_ResponseReGeneration=[]
            for idx,instance_processing in enumerate(instances_processing):
                individual_RoleSet = instance_processing["individual_RoleSet"]
                individual_RoleSet_str = "; ".join(
                    [individual_RoleSet[key_l] + " at " + key_l for key_l in individual_RoleSet.keys()])

                if iter==0:
                    instances_processing[idx]["responses_and_feedbacks"]=[]
                    query = instance_processing['query']
                    prompt=query
                else:
                    prompt=get_SelfRefine_prompt(instance=instance_processing,stage="ResponseReGeneration")

                image=instance_processing['image']["file_path"]
                image_path = f"{bench_root_path}/{image}"

                specified_sys_prompt = f"You are a helpful assistant for a user who is \"{individual_RoleSet_str}\"."

                vllm_input = get_vllm_input(prompt,image_path,processor=processor,specified_sys_prompt=specified_sys_prompt)
                vllm_inputs_for_ResponseReGeneration.append(vllm_input)
            outputs_for_ResponseReGeneration = llm.generate(vllm_inputs_for_ResponseReGeneration, sampling_params=sampling_params)
            for idx in range(len(outputs_for_ResponseReGeneration)):
                generated_text = outputs_for_ResponseReGeneration[idx].outputs[0].text
                if "<Refined Response>" in generated_text:
                    generated_text = generated_text.split("<Refined Response>")[-1]

                try:
                    new_response = generated_text.split("</Refined Response>")[0]
                except:
                    new_response = generated_text
                instances_processing[idx]["responses_and_feedbacks"].append(
                    {
                        "response": new_response,
                    }
                )

            if iter==num_iters-1:
                break

            vllm_inputs_for_FeedbackScoreGeneration=[]
            for instance_processing in instances_processing:

                prompt = get_SelfRefine_prompt(instance=instance_processing, stage="FeedbackScoreGeneration")

                image=instance_processing['image']["file_path"]
                image_path = f"{bench_root_path}/{image}"

                vllm_input = get_vllm_input(prompt,image_path,processor=processor)
                vllm_inputs_for_FeedbackScoreGeneration.append(vllm_input)
            outputs_for_FeedbackScoreGeneration = llm.generate(vllm_inputs_for_FeedbackScoreGeneration, sampling_params=sampling_params)
            for idx in range(len(outputs_for_FeedbackScoreGeneration)):
                generated_text = outputs_for_FeedbackScoreGeneration[idx].outputs[0].text
                if "<Evaluation Score of the Response>" in generated_text:
                    generated_text = generated_text.split("<Evaluation Score of the Response>")[-1]

                try:
                    feedback_score=generated_text.split("</Evaluation Score of the Response>")[0]
                except:
                    feedback_score=generated_text

                instances_processing[idx]["responses_and_feedbacks"][-1]["feedback_score"]=feedback_score

            vllm_inputs_for_FeedbackDescGeneration=[]
            for idx,instance_processing in enumerate(instances_processing):

                prompt = get_SelfRefine_prompt(instance=instance_processing, stage="FeedbackDescGeneration")

                image=instance_processing['image']["file_path"]
                image_path = f"{bench_root_path}/{image}"

                vllm_input = get_vllm_input(prompt,image_path,processor=processor)
                vllm_inputs_for_FeedbackDescGeneration.append(vllm_input)
            outputs_for_FeedbackDescGeneration = llm.generate(vllm_inputs_for_FeedbackDescGeneration, sampling_params=sampling_params)
            for idx in range(len(outputs_for_FeedbackDescGeneration)):
                generated_text = outputs_for_FeedbackDescGeneration[idx].outputs[0].text
                if "<Evaluation Explanation>" in generated_text:
                    generated_text = generated_text.split("<Evaluation Explanation>")[-1]

                try:
                    feedback=generated_text.split("</Evaluation Explanation>")[0]
                except:
                    feedback=generated_text

                instances_processing[idx]["responses_and_feedbacks"][-1]["feedback"]=feedback

            with open(target_file_path, "w", encoding="utf-8") as f:
                json.dump(instances_processing, f, indent=2)

        for idx,instance_processing in enumerate(instances_processing):
            chosen_response=instance_processing["responses_and_feedbacks"][-1]["response"]
            rejected_response=instance_processing["initial_response"]

            instances_processing[idx]["preference_pair"]={
                "chosen_response":chosen_response,
                "rejected_response":rejected_response,
            }
            instances_processing[idx]["POSelfRefine_response"]=chosen_response


        with open(target_file_path, "w", encoding="utf-8") as f:
            json.dump(instances_processing, f, indent=2)
