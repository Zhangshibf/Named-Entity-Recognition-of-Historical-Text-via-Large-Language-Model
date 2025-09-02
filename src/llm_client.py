import os
from src.settings import BASE_DIR,DATASET_DIR
from src.prompt_generate import prompt_generator
from src.utils import save_response, query, parse_saved_response, find_raw_output_file,safe_filename
import json
import re
class llm_client():
    def __init__(self):
        pass


    def run(self, prompt_type,dataset, task, model, key,temperature,subfolder,resume=False):
        prompter = prompt_generator()
        label_set = dataset.dataset_name + "_" + task
        document_ids_done = []

        raw_output_path =find_raw_output_file(prompt_type,dataset,model,task,temperature,subfolder)
        pattern = r'"document_id"\s*:\s*"([^"]+)"'
        if os.path.exists(raw_output_path) and resume:
            with open(raw_output_path, 'r', encoding='utf-8') as f:
                content = f.read()
            matches = re.findall(pattern, content)
            document_ids_done.extend(matches)
        elif os.path.exists(raw_output_path) and not resume:
            raise ValueError(f"Not resume, but file {raw_output_path} already exists")

        for i, (document_id, document) in enumerate(dataset.documents.items()):
            if document_id not in document_ids_done:
                text = dataset.texts[document_id]
                if prompt_type == "b":
                    prompt = prompter.baseline_prompt(text, dataset.entity_label_set[label_set], task)

                elif prompt_type.startswith("r"):
                    prompt = prompter.random_example_prompt(dataset, text, dataset.entity_label_set[label_set],
                                                            task, example_num=int(prompt_type[-1]))
                elif prompt_type.startswith("s"):
                    mode = prompt_type.split("_")[-2]
                    prompt = prompter.similar_example_prompt(dataset, document_id, dataset.entity_label_set[label_set],
                                                            task, example_num=int(prompt_type[-1]),mode=mode)
                else:
                    pass
                response = query(key, prompt, temperature,model)
                response['document_id'] = document_id
                save_response(prompt_type,dataset, model, task,temperature, response,subfolder)
                print(f"got {i}")



        parse_saved_response(prompt_type,dataset,model,task,temperature,subfolder)



