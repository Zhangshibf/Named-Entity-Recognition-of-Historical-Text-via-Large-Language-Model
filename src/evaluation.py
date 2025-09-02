import sys
sys.path.append("D:/HIPE2022/hipe_llm")
sys.path.append("D:/HIPE2022/hipe_llm/HIPE_scorer")
import argparse
from HIPE_scorer.hipe_evaluation.ner_eval import Evaluator
import json
import numpy as np
import os
from src.settings import BASE_DIR,DATASET_DIR,EVALUATION_DIR,PREDICTION_voted_DIR, PREDICTION_DIR
from pathlib import Path
import pandas as pd



def find_output_path(args,task_suffix):
    folder = os.path.join(
        EVALUATION_DIR,
        args.subfolder,
        args.prompt,
        args.model,
        f"temperature_{args.temperature}",
        args.dataset_name,
        args.language,
        args.split
    )

    filename = f"{args.model}-{task_suffix}-{args.dataset_name}-{args.language}-{args.split}-result.tsv"
    new_path = os.path.join(folder, filename)
    directory = os.path.dirname(new_path)
    os.makedirs(directory, exist_ok=True)

    return new_path






if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Compare reference and prediction files.")
    parser.add_argument("dataset_name")
    parser.add_argument("language")
    parser.add_argument("split")
    parser.add_argument("prompt")
    parser.add_argument("model")
    parser.add_argument("temperature")
    parser.add_argument("subfolder")
    parser.add_argument("task", help="choose nerc_fine or nerc_coarse")
    parser.add_argument("--voted", action="store_true", help="Use voted setting (default: False)")

    args = parser.parse_args()

    COARSE_COLUMNS_HIPE2022 = ["NE-COARSE-LIT"]
    FINE_COLUMNS_HIPE2022 = ["NE-FINE-LIT", "NE-NESTED"]

    task_suffix = "coarse" if "coarse" in args.task else "fine"

    if args.voted:
        pred_folder = PREDICTION_voted_DIR

    else:
        pred_folder = PREDICTION_DIR


    def get_file_paths(split):
        dataset_folder = os.path.join(DATASET_DIR, args.dataset_name, args.language)
        dataset_file_name = f"HIPE-2022-v2.1-{args.dataset_name}-{split}-{args.language}.tsv"
        f_reference = os.path.join(dataset_folder, dataset_file_name)

        temp = f"temperature_{args.temperature}"
        prediction_folder = os.path.join(pred_folder, args.subfolder,args.prompt, args.model, temp, args.dataset_name,
                                         args.language, split)
        prediction_file_name = f"{args.model}-{task_suffix}-{args.dataset_name}-{args.language}-{split}.tsv"
        f_prediction = os.path.join(prediction_folder, prediction_file_name)

        return f_reference, f_prediction


    if "coarse" in args.task:
        ner_columns = COARSE_COLUMNS_HIPE2022
    elif "fine" in args.task:
        ner_columns = FINE_COLUMNS_HIPE2022

    if args.split == "traindev":
        def concatenate_files(input_paths, output_path):
            with open(output_path, "w", encoding="utf-8") as outfile:
                for path in input_paths:
                    if os.path.exists(path):
                        with open(path, "r", encoding="utf-8") as infile:
                            outfile.write(infile.read())
                    else:
                        print(f"Warning: {path} does not exist and will be skipped.")

        subsets = ["train", "dev", "dev2"]
        ref_paths = []
        pred_paths = []

        for subset in subsets:
            try:
                ref_path, pred_path = get_file_paths(subset)
                print(ref_path)
                print(pred_path)
                if os.path.exists(ref_path) and os.path.exists(pred_path):
                    ref_paths.append(ref_path)
                    pred_paths.append(pred_path)
                    print(f"found {subset}")
                else:
                    print(f"Skipping missing subset '{subset}'")
            except Exception as e:
                print(f"Error getting paths for subset '{subset}': {e}")

        # Concatenate available files
        concatenate_files(ref_paths, "combined_ref.tsv")
        concatenate_files(pred_paths, "combined_pred.tsv")

        f_reference, f_prediction = "combined_ref.tsv", "combined_pred.tsv"
    else:
        f_reference, f_prediction = get_file_paths(args.split)


    evaluator = Evaluator(f_reference,f_prediction)
    results, _ = evaluator.evaluate(ner_columns, eval_type="nerc")

    out_path = find_output_path(args,task_suffix)



    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)





