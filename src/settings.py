from pathlib import Path
import os
import re
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TOOLS_DIR = os.path.join(BASE_DIR, "tools")
SIMILARITY_DIR =os.path.join(BASE_DIR, "similarity")
EVALUATION_DIR = os.path.join(BASE_DIR, "eval_results")
DATASET_INFO = {"ajmc": {"de", "en", "fr"}, "hipe2020": {"de", "en", "fr"}, "letemps": {"fr"},
                    "newseyechunked": {"de", "fi", "fr", "sv"}, "sonar": {"de"},
                    "topres19th": {"en"}}
PREDICTION_DIR = os.path.join(BASE_DIR, "prediction")
PREDICTION_voted_DIR = os.path.join(BASE_DIR, "voted_prediction")

import os

def find_voted_output_file(input_files,column,create=True):

    subfolders = []
    filenames = set()
    for path in input_files:
        parts = path.split(os.sep)
        pred_index = parts.index("prediction")
        subfolder = parts[pred_index + 1]
        filename = parts[-1]

        subfolders.append(subfolder)
        filenames.add(filename)

    if len(filenames) != 1:
        raise ValueError("All input files must have the same filename.")

    joined_subfolders = "_".join(subfolders)+"_"+column
    # Split the path into parts
    path_parts = input_files[0].split(os.sep)

    # Find the index of PREDICTION_DIR in the path
    prediction_dir_parts = PREDICTION_DIR.split(os.sep)
    prediction_dir_depth = len(prediction_dir_parts)

    # The next folder after PREDICTION_DIR is the one to replace
    if path_parts[prediction_dir_depth:]:
        path_parts[prediction_dir_depth] = joined_subfolders


    new_path = os.sep.join(path_parts)
    path_parts = new_path.split(os.sep)

    # Replace 'prediction' with 'voted_prediction' (case-sensitive)
    path_parts = [part if part != 'prediction' else 'voted_prediction' for part in path_parts]
    new_path = os.sep.join(path_parts)


    if create:
        output_dir = os.path.dirname(new_path)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    return new_path

def find_similarity_file(dataset_name, language, split, method):
    dataset_dir = os.path.join(SIMILARITY_DIR, dataset_name,language)
    ensure_dir(dataset_dir)

    filename = f"{dataset_name}_{split}_{method}_similarity.json"
    filepath = os.path.join(dataset_dir, filename)
    return filepath

def ensure_dir(path):
    """Ensure directory exists, create if not"""
    os.makedirs(path, exist_ok=True)

def smart_convert(x):
    if x == int(x):
        return int(x)
    else:
        return x
def safe_filename(name):
    return re.sub(r'[<>:"/\\|?*]', '_', name)
def find_output_folder(prompt_type,dataset,system,temperature,subfolder):

    temperature_string = "temperature_"+str(smart_convert(temperature))
    output_folder = os.path.join(BASE_DIR,"prediction",subfolder,prompt_type,system,temperature_string,dataset.dataset_name,dataset.language,dataset.split)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"made folder {output_folder}")

    return output_folder

def find_raw_output_file(prompt_type, dataset, system, task,temperature,subfolder):
    system = safe_filename(system)

    raw_file_name = "raw-"+system+"-"+task+"-"+dataset.dataset_name+"-"+dataset.language+"-"+dataset.split+".tsv"
    output_folder = find_output_folder(prompt_type, dataset, system,temperature,subfolder)
    raw_output_path = os.path.join(output_folder, raw_file_name)

    return raw_output_path

def find_parsed_output_file(prompt_type, dataset,system,task,temperature,subfolder):
    system = safe_filename(system)
    parsed_file_name = system + "-" + task + "-" + dataset.dataset_name + "-" + dataset.language + "-" + dataset.split + ".tsv"
    output_folder = find_output_folder(prompt_type, dataset, system,temperature,subfolder)
    parsed_output_path = os.path.join(output_folder, parsed_file_name)

    return parsed_output_path



def find_dataset(dataset_name,language,split):
    dataset_folder = os.path.join(DATASET_DIR, dataset_name, language)
    dataset_file_name = "HIPE-2022-v2.1-" + dataset_name + "-" + split + '-' + language + '.tsv'
    dataset_file = os.path.join(dataset_folder, dataset_file_name)

    return dataset_file
