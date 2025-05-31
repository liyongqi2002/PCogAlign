import json
import os

import random
import re

from tqdm import tqdm

from api_utils import get_res_api
from eval_prompt_utils import get_eval_prompt_Rating, get_eval_prompt_Pair


def extract_evaluation_data(response):
    # Use regular expressions to capture the scores from the response
    try:
        score_pattern = re.compile(r'\[\[(\d)\]\]')
        scores = score_pattern.findall(response)
        scores=[int(score) for score in scores]
        assert len(scores)==5
    except Exception as e:
        try:
            score_pattern = re.compile(r'\[(\d)\]')
            scores = score_pattern.findall(response)
            scores = [int(score) for score in scores]
            assert len(scores) == 5
        except Exception as e:
            print(e)
    # Elements to extract from the response
    elements = [
        "Role-Set Sensitivity",
        "Body Behavior Awareness",
        "Mind Feelings Awareness",
        "Contextual Awareness",
        "Conversational Flow"
    ]

    # Extract explanations after "Evaluation Explanation ###"
    explanation_section = response.split("### Evaluation Explanation ###")[-1].strip()

    # Create a dictionary with the captured data
    evaluation_data = {
        "Evaluation Result Scores": {elements[i]: scores[i] for i in range(len(elements))},
        "Element Average Score": sum(scores)/len(scores),
        "Evaluation Explanations": explanation_section
    }

    return evaluation_data

def parse_res_text_to_score(eval_type, response_text):
    if eval_type == "RatingEval":
        # eval_score = response_text.split("]]")[0].split("[[")[1]
        # eval_score=int(eval_score)
        eval_score=extract_evaluation_data(response_text)


    elif eval_type=="PairEval":
        eval_str = response_text.split("]]")[0].split("[[")[1]
        # transform the A/B/TIE into score
        if eval_str == "A":
            eval_score = -1
        elif eval_str == "B":
            eval_score = 1
        elif eval_str == "TIE":
            eval_score = 0
        else:
            raise ValueError(f"Not A/B/TIE {eval_str}")
    else:
        raise ValueError(f"Invalid eval type: {eval_type}")
    return eval_score


def get_retry_eval_score(eval_type, prompt_eval,sys_prompt):
    temperature = 0
    flag = True
    while flag:
        try:
            res_eval = get_res_api(prompt=prompt_eval, temperature=temperature, llm_type="gpt-4o-mini",
                                   specified_sys_prompt=sys_prompt)
            eval_score = parse_res_text_to_score(eval_type, res_eval)
            flag = False
        except Exception as e:
            print(e)
            temperature = 0.5
    return eval_score


def parse_eval_result(eval_type, response_text, test_instance, ref_base_response_instance):
    if eval_type=="RatingEval":
        try:
            eval_score = parse_res_text_to_score(eval_type, response_text)
        except:
            prompt_eval, sys_prompt = get_eval_prompt_Rating(test_instance)
            eval_score = get_retry_eval_score(eval_type, prompt_eval, sys_prompt)

    elif eval_type=="PairEval":
        try:
            eval_score = parse_res_text_to_score(eval_type, response_text)
        except:
            prompt_eval, sys_prompt = get_eval_prompt_Pair(test_instance,
                                                           response_A=ref_base_response_instance["response"],
                                                           response_B=test_instance["response"])
            eval_score = get_retry_eval_score(eval_type, prompt_eval, sys_prompt)
    else:
        raise ValueError(f"Invalid eval type: {eval_type}")
    return eval_score


import sys
sys.path.append('..')

VLM_name = None

if __name__ == '__main__':
    #########################
    # eval engine
    #########################
    eval_engine="gpt-4o-mini"


    batch_work_ids=[

        "batch_67af5eb2bc348190862bd99f1e461df0",


    ]

    results_requests_gpt4 = []
    for batch_work_id in batch_work_ids:
        results_requests_gpt4_file=f"batches_gpt4_requests/{batch_work_id}_output.jsonl"
        with open(results_requests_gpt4_file, mode='r') as f:
            lines=f.readlines()
            for line in lines:
                results_requests_gpt4.append(json.loads(line))

    print(len(results_requests_gpt4))

    detailed_method_names=[]
    test_sub_sets=[]
    for result_object in tqdm(results_requests_gpt4):
        custom_id=result_object["custom_id"]
        pattern = r"request-EvalType\[(.*?)\]-Method\[(.*?)\]-VLM\[(.*?)\]-SubSet\[(.*?)\]-Idx\[(.*?)\]"
        match = re.match(pattern, custom_id)
        method_name = match.group(2)
        # method_name=method_name.replace("]-VLM[Qwen2-VL-7B-Instruct","")
        test_sub_set = match.group(4)

        # reload
        VLM_name=match.group(3)

        if method_name not in detailed_method_names:
            detailed_method_names.append(method_name)
        if test_sub_set not in test_sub_sets:
            test_sub_sets.append(test_sub_set)

    print(detailed_method_names)

    dict_ref_base_responses={}
    dict_generation_results={}
    dict_eval_results={}
    for method_name in detailed_method_names:
        dict_ref_base_responses[method_name]={}
        dict_generation_results[method_name]={}

        dict_eval_results[method_name]={}

        for test_sub_set in test_sub_sets:
            ref_base_response_path=f"../generation_results/METHOD[base]-VLM[{VLM_name}]-TEST[{test_sub_set}].json"
            with open(ref_base_response_path, "r", encoding="utf-8") as f:
                ref_base_responses = json.load(f)

            generation_results_path=f"../generation_results/METHOD[{method_name}]-VLM[{VLM_name}]-TEST[{test_sub_set}].json"
            with open(generation_results_path, "r", encoding="utf-8") as f:
                test_instances = json.load(f)

            dict_ref_base_responses[method_name][test_sub_set]=ref_base_responses
            dict_generation_results[method_name][test_sub_set]=test_instances

            dict_eval_results[method_name][test_sub_set]=[]

    eval_type="" # only one eval_type for each output.jsonl
    for result_object in tqdm(results_requests_gpt4):
        custom_id=result_object["custom_id"]
        # print(custom_id)

        pattern = r"request-EvalType\[(.*?)\]-Method\[(.*?)\]-VLM\[(.*?)\]-SubSet\[(.*?)\]-Idx\[(.*?)\]"
        match = re.match(pattern, custom_id)

        eval_type=match.group(1)
        method_name = match.group(2)
        # method_name=method_name.replace("]-VLM[Qwen2-VL-7B-Instruct","")

        test_sub_set = match.group(4)
        idx = int(match.group(5))


        ref_base_response_instance=dict_ref_base_responses[method_name][test_sub_set][idx]
        test_instance=dict_generation_results[method_name][test_sub_set][idx]

        response_text=result_object["response"]["body"]["choices"][0]["message"]["content"]

        eval_score=parse_eval_result(eval_type,response_text,
                                     test_instance,ref_base_response_instance)


        test_instance["eval_score"] = {
            f"eval_engine-{eval_engine}": eval_score
        }

        if eval_type == "PairEval":
            test_instance["eval_pairs"]={
                "response_A": ref_base_response_instance["response"],
                "response_B": test_instance["response"],
            }

        dict_eval_results[method_name][test_sub_set].append(test_instance)



    for method_name in detailed_method_names:
        for test_sub_set in test_sub_sets:
            test_instances=dict_eval_results[method_name][test_sub_set]
            print(f"EvalType {eval_type} METHOD[{method_name}]-VLM[{VLM_name}]-TEST[{test_sub_set}]")
            if eval_type == "RatingEval":
                elements = [
                    "Role-Set Sensitivity",
                    "Body Behavior Awareness",
                    "Mind Feelings Awareness",
                    "Contextual Awareness",
                    "Conversational Flow"
                ]
                for element in elements:

                    eval_scores=[item["eval_score"][f"eval_engine-{eval_engine}"]["Evaluation Result Scores"][element] for item in test_instances]
                    avg_eval_score=sum(eval_scores)/len(eval_scores)
                    print(f"{element}: {avg_eval_score}")

                ElementAvg_eval_scores=[item["eval_score"][f"eval_engine-{eval_engine}"]["Element Average Score"] for item in test_instances]
                avg_ElementAvg_eval_scores = sum(ElementAvg_eval_scores) / len(ElementAvg_eval_scores)
                print(f"ElementAvg_eval_scores: {avg_ElementAvg_eval_scores}")

            elif eval_type == "PairEval":
                eval_scores=[item["eval_score"][f"eval_engine-{eval_engine}"] for item in test_instances]
                # print(eval_scores)
                count_1 = eval_scores.count(1)
                count_0 = eval_scores.count(0)
                count_neg1 = eval_scores.count(-1)

                total_count = len(eval_scores)

                ratio_1 = count_1 / total_count
                ratio_0 = count_0 / total_count
                ratio_neg1 = count_neg1 / total_count
                print(f"win: {ratio_1} tie: {ratio_0} lose: {ratio_neg1}")




            eval_results_path=f"eval_results/{eval_type}_METHOD[{method_name}]-VLM[{VLM_name}]-TEST[{test_sub_set}].json"
            with open(eval_results_path, "w", encoding="utf-8") as f:
                json.dump(test_instances, f, indent=2)