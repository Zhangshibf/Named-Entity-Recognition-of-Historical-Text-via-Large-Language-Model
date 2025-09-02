from src.datasetloader import Dataset_loader
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import nltk
from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords
import regex as regexx
import numpy as np
from collections import Counter
from src.settings import BASE_DIR,SIMILARITY_DIR,ensure_dir,DATASET_INFO,find_similarity_file
import json
class prompt_generator():
    def __init__(self):
        pass

    def baseline_prompt(self, text, entity_label_set, task):
        if task=="coarse":
            prompt =[
            {"role": "system", "content": "You are a helpful and accurate Named Entity Recognition (NER) system.\n"},
            {"role": "user", "content": f"Your task is to identify and label named entities in the passage below using the following entity label set: {entity_label_set}.\n"
              "Important guidelines:\n"
              "- There should be no overlap between different entities (i.e., no nested or intersecting spans).\n"
              "- Only include spans that match one of the specified labels.\n"
              "- Be precise and only extract valid named entities.\n"
              "- Do not return an empty list. There are always some entities in the passage.\n"
              "Output format:\n"
              'A Python list of tuples, where each tuple is of the form: ("entity text", "entity label").\n'
              'Do not include any explanation or introductory text. Your output must be *only* a valid Python list of tuples.'
              f"\nPassage:\n{text}"}]


        elif task=="fine":
            prompt = [
                {"role": "system",
                 "content": "You are a helpful and accurate Named Entity Recognition (NER) system.\n"},
                {"role": "user",
                 "content": f"Your task is to identify and label named entities in the passage below using the following entity label set: {entity_label_set}.\n"
                            "Important guidelines:\n"
                            "- Entities may occasionally be nested. In such cases, include *both* the nested and enclosing entities as separate entries.\n"
                            "- Only include spans that match one of the specified labels.\n"
                            "- Be precise and only extract valid named entities.\n"
                            "- Do not return an empty list. There are always some entities in the passage.\n"
                            "Output format:\n"
                            'A Python list of tuples, where each tuple is of the form: ("entity text", "entity label").\n'
                            "For nested entities, include separate tuples for each.\n"
                            'Do not include any explanation or introductory text. Your output must be *only* a valid Python list of tuples.'
                            f"\nPassage:\n{text}"}]

        return prompt

    def format_prompt_with_example(self,example_texts,example_annotations,text,entity_label_set):
        prompt = [
            {"role": "system", "content": "You are a helpful and accurate Named Entity Recognition (NER) system.\n"}]
        for example_text, example_annotation in zip(example_texts, example_annotations):
            # {"role": , "content":}
            prompt.append({"role": "user",
                           "content": f"Your task is to identify and label named entities in the passage below using the following entity label set: {entity_label_set}.\n"
                                      "Important guidelines:\n"
                                      "- There should be no overlap between different entities (i.e., no nested or intersecting spans).\n"
                                      "- Only include spans that match one of the specified labels.\n"
                                      "- Be precise and only extract valid named entities.\n"
                                      "- Do not return an empty list. There are always some entities in the passage.\n"
                                      "Output format:\n"
                                      'A Python list of tuples, where each tuple is of the form: ("entity text", "entity label").\n'
                                      'Do not include any explanation or introductory text. Your output must be *only* a valid Python list of tuples.'
                                      f"\nPassage:\n{example_text}"})
            prompt.append({"role": "assistant", "content": f"{example_annotations}"})

        prompt.append({"role": "user",
                       "content": f"Your task is to identify and label named entities in the passage below using the following entity label set: {entity_label_set}.\n"
                                  "Important guidelines:\n"
                                  "- There should be no overlap between different entities (i.e., no nested or intersecting spans).\n"
                                  "- Only include spans that match one of the specified labels.\n"
                                  "- Be precise and only extract valid named entities.\n"
                                  "- Do not return an empty list. There are always some entities in the passage.\n"
                                  "Output format:\n"
                                  'A Python list of tuples, where each tuple is of the form: ("entity text", "entity label").\n'
                                  'Do not include any explanation or introductory text. Your output must be *only* a valid Python list of tuples.'
                                  f"\nPassage:\n{text}"})

        return prompt

    def random_example_prompt(self,dataloader:Dataset_loader, text, entity_label_set,task,example_num):

        if task=="coarse":
            example_texts,example_annotations = self.retrieve_random_example(dataloader,task,example_num)
            prompt = self.format_prompt_with_example(example_texts,example_annotations,text, entity_label_set)

        elif task == "fine":
            example_texts, example_annotations,example_nested = self.retrieve_examples(dataloader,nested = True)
            prompt = self.format_prompt_with_example(example_texts, (example_annotations,example_nested), text, entity_label_set)
            pass

        return prompt

    def similar_example_prompt(self, dataloader: Dataset_loader, document_id, entity_label_set, task,example_num,mode):
        if task == "coarse":
            text = dataloader.texts[document_id]
            example_texts, example_annotations = self.retrieve_similar_example(document_id,dataloader, task,example_num,mode)
            prompt = self.format_prompt_with_example(example_texts, example_annotations, text, entity_label_set)

        elif task == "fine":
            # examples, nested_example = self.retrieve_examples(dataloader,nested = True)
            pass

        return prompt


    def retrieve_random_example(self,dataloader:Dataset_loader,task,example_num):
        #retrieve at least one instance per entity type
        #when nested is True, retrieve an example of nested as well

        traindev_data = Dataset_loader(dataloader.dataset_name, dataloader.language, split="traindev")
        if task=="coarse":

            document_ids = list(traindev_data.annotations_coarse.keys())
            import random
            document_count = len(document_ids)
            nums = random.sample(range(document_count), example_num)
            example_ids = [document_ids[num] for num in nums]

            example_texts = [traindev_data.texts[example_id] for example_id in example_ids]
            example_annotations = [traindev_data.annotations_coarse[example_id] for example_id in example_ids]


            return example_texts,example_annotations
        elif task=="fine":
            document_ids = traindev_data.annotations_fine.keys()
            import random
            empty_doc_id = [key for key in traindev_data.annotations_fine if traindev_data.annotations_fine[key] == [] or traindev_data.annotation_nested[key] == []]
            example_id=empty_doc_id[0]
            document_count = len(document_ids)
            while example_id in empty_doc_id:
                num = random.randint(0, document_count-1)
                example_id = document_ids[num]

            example_text = traindev_data.texts[example_id]
            example_annotation_fine = traindev_data.annotations_fine[example_id]
            example_annotation_nested=traindev_data.annotation_nested[example_id]

            return example_text,example_annotation_fine,example_annotation_nested

    def retrieve_similar_example(self,document_id,dataloader:Dataset_loader,task,example_num,mode):
        #retrieve an example based on similarity
        if task=="coarse":
            selected_ids = self.find_similar(document_id,dataloader.dataset_name, dataloader.language, dataloader.split, example_num,mode)
            example_texts,example_annotations = self.retrieve_example_content(dataloader.dataset_name, dataloader.language,selected_ids)

            return example_texts, example_annotations

        elif task=="fine":
            pass

    def retrieve_example_content(self,dataset_name, language,selected_ids):
        possible_splits = ["train","dev","dev2"]

        example_texts=[]
        example_annotations=[]
        order = []
        for split in possible_splits:
            try:
                data = Dataset_loader(dataset_name, language=language, split=split)
                for idx,id in enumerate(selected_ids):

                    if id in data.texts.keys():
                        order.append(idx)
                        example_texts.append(data.texts[id])
                        example_annotations.append(data.annotations_coarse[id])
            except:
                pass
        combined = list(zip(order, example_texts, example_annotations))
        combined.sort()  # sort by the order index (first element of each tuple)
        _, example_texts, example_annotations = zip(*combined) if combined else ([], [], [])
        return list(example_texts), list(example_annotations)

    def find_similar(self,target_id,dataset_name, language, split, n,mode):
        similarity_file = find_similarity_file(dataset_name, language, split,mode)
        with open(similarity_file, 'r') as f:
            data = json.load(f)
        top_ids = data[target_id][:n]

        return top_ids
