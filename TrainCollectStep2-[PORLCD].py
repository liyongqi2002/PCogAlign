import json
import os

from utils import get_vllm_input, prepare_vllm





bench_root_path="PCogAlignBench/version_v4"

if __name__ == '__main__':
    from utils import import_VLM_name

    VLM_path, VLM_name = import_VLM_name()

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

        target_file_path=f"temp/METHOD[PORLCD]-VLM[{VLM_name}]-TRAIN[{train_sub_set}].json"

        for idx,instance_processing in enumerate(instances_processing):
            chosen_response=instance_processing["PORLCD_response"]
            rejected_response=instance_processing["initial_response"]

            instances_processing[idx]["preference_pair"]={
                "chosen_response":chosen_response,
                "rejected_response":rejected_response,
            }

        with open(target_file_path, "w", encoding="utf-8") as f:
            json.dump(instances_processing, f, indent=2)