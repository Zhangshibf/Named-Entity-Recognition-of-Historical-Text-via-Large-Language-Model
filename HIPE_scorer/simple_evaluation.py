import logging
import csv
import pathlib
import json
import sys

import itertools
from collections import defaultdict

from datetime import datetime
from docopt import docopt
import argparse
from hipe_evaluation.ner_eval import Evaluator
import os

def find_output_path(path):
    # Normalize and split the path into parts
    path = os.path.normpath(path)
    parts = path.split(os.sep)

    # Replace "prediction" with "eval_results"
    parts = ["eval_results" if part == "prediction" else part for part in parts]

    # Modify the filename
    filename = parts[-1]
    name, ext = os.path.splitext(filename)
    new_filename = name + "-result" + ext
    parts[-1] = new_filename

    # Join everything back into a path
    new_path = os.sep.join(parts)

    # Create the directory if it doesn't exist
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
    parser.add_argument("task", help="choose nerc_fine or nerc_coarse")


    args = parser.parse_args()
    import os


    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    DATASET_DIR = os.path.join(BASE_DIR, "dataset")
    dataset_folder = os.path.join(DATASET_DIR, args.dataset_name, args.language)
    dataset_file_name = "HIPE-2022-v2.1-" + args.dataset_name + "-" + args.split + '-' + args.language + '.tsv'
    f_reference = os.path.join(dataset_folder, dataset_file_name)
    temp = "temperature_"+str(args.temperature)
    prediction_folder = os.path.join(BASE_DIR, "prediction",args.prompt,args.model,temp,args.dataset_name,args.language,args.split)
    prediction_file_name = args.model +"-coarse-"+args.dataset_name+"-"+args.language+"-"+args.split+".tsv"
    f_prediction = os.path.join(prediction_folder,prediction_file_name)

    COARSE_COLUMNS_HIPE2022 = ["NE-COARSE-LIT"]
    FINE_COLUMNS_HIPE2022 = ["NE-FINE-LIT", "NE-NESTED"]
    if "coarse" in args.task:
        ner_columns = COARSE_COLUMNS_HIPE2022
    elif "fine" in args.task:
        ner_columns = FINE_COLUMNS_HIPE2022


    evaluator = Evaluator(f_reference,f_prediction)
    results, results_pertype = evaluator.evaluate(ner_columns, eval_type="nerc")
    out_path = find_output_path(f_prediction)
    print(out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
