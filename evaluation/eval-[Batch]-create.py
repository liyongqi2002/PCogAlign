import json
import os

from tqdm import tqdm

from eval_prompt_utils import get_eval_prompt_Rating, get_eval_prompt_Pair


import sys
sys.path.append('..')
from utils import import_VLM_name

# VLM_name = "Qwen2.5-VL-3B-Instruct"
# VLM_name = "Qwen2.5-VL-7B-Instruct"
VLM_name = "Qwen2-VL-7B-Instruct"


if __name__ == '__main__':



    detailed_method_names=[]
    # detailed_method_names+=["base"]
    # detailed_method_names+=["RSPrompt"]
    # detailed_method_names+=["PCogAlign_Prompt"]
    #
    #
    # detailed_method_names+=["RSPromptSFT#HCMAS#AsTrain"]
    # detailed_method_names+=["POSelfRefine_SFT#HCMAS#AsTrain"]
    # detailed_method_names+=["PORLCD_SFT#HCMAS#AsTrain"]
    # detailed_method_names+=["PORLAIF_SFT#HCMAS#AsTrain"]



    # detailed_method_names+=["POSelfRefine_DPO#HCMAS#AsTrain"]
    # detailed_method_names+=["PORLCD_DPO#HCMAS#AsTrain"]
    # detailed_method_names+=["PORLAIF_DPO#HCMAS#AsTrain"]


    detailed_method_names+=["BestOfNSFT#HCMAS#AsTrain"]
    # detailed_method_names+=["PCogAlign_SFT#HCMAS#AsTrain"]
    # detailed_method_names+=["PCogAlign_DPO#HCMAS#AsTrain"]





    test_sub_sets=["HCMAS","HCSHR"]


    eval_engine="gpt-4o-mini"

    # PairEval here is not recommended because: 1) costly (two orders are needed to consider); 2) more easily influenced by length of response
    # We recommend use the "indirect PairEval"

    # if detailed_method_names==["base"]:
    #     eval_types = ["RatingEval"]
    # else:
    #     eval_types=["RatingEval","PairEval"]

    eval_types=["RatingEval"]
    # eval_types=["PairEval"]


    for eval_type in eval_types:

        request_objects=[]
        for method_name in detailed_method_names:
            for test_sub_set in test_sub_sets:
                ref_base_response_path=f"../generation_results/METHOD[base]-VLM[{VLM_name}]-TEST[{test_sub_set}].json"
                with open(ref_base_response_path, "r", encoding="utf-8") as f:
                    ref_base_responses = json.load(f)

                generation_results_path=f"../generation_results/METHOD[{method_name}]-VLM[{VLM_name}]-TEST[{test_sub_set}].json"
                with open(generation_results_path, "r", encoding="utf-8") as f:
                    test_instances = json.load(f)

                # test_instances=test_instances[:100]

                for idx,test_instance in tqdm(enumerate(test_instances),total=len(test_instances),desc=f"METHOD[{method_name}]-VLM[{VLM_name}]-TEST[{test_sub_set}]"):

                    if eval_type=="RatingEval":
                        # prompt_eval,sys_prompt = get_eval_prompt_Rating(test_instance,role_set_EvalHelp)
                        prompt_eval,sys_prompt = get_eval_prompt_Rating(test_instance)

                    elif eval_type=="PairEval":
                        # prompt_eval,sys_prompt = get_eval_prompt_Pair(test_instance,role_set_EvalHelp,
                        #                                               response_A=ref_base_responses[idx]["response"],
                        #                                               response_B=test_instances[idx]["response"])
                        prompt_eval,sys_prompt = get_eval_prompt_Pair(test_instance,
                                                                      response_A=ref_base_responses[idx]["response"],
                                                                      response_B=test_instances[idx]["response"])
                    else:
                        raise ValueError(f"Invalid eval type: {eval_type}")


                    # print("==============================")
                    # print(sys_prompt)
                    # print(prompt_eval)
                    # assert 1==0

                    request_object = {
                        "custom_id": f"request-EvalType[{eval_type}]-Method[{method_name}]-VLM[{VLM_name}]-SubSet[{test_sub_set}]-Idx[{idx}]",
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {"model": eval_engine,
                                 "messages": [{"role": "system",
                                               "content": sys_prompt},
                                              {"role": "user", "content": str(prompt_eval)}],
                                 "max_tokens": 512}}

                    request_objects.append(request_object)


        print(len(request_objects))
        dirs = ["batches_gpt4_requests"]
        for dir in dirs:
            if not os.path.exists(dir):
                os.mkdir(dir)

        TAGS="-".join([item.replace("#","")[:5] for item in detailed_method_names])
        judge_requests_file=f"batches_gpt4_requests/{eval_type}_Eval_requests_{eval_engine}-TAGS_{TAGS}.jsonl"

        import jsonlines
        with jsonlines.open(judge_requests_file, mode='w') as writer:
            writer.write_all(request_objects)