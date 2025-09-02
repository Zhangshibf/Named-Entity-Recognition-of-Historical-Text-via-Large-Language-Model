import os
import re
import copy
import requests
import json
import ast
from src.settings import safe_filename,find_raw_output_file,find_parsed_output_file
import copy
from difflib import SequenceMatcher

def save_response(prompt_type,dataset,system,task,temperature,response,subfolder):
    system = safe_filename(system)
    raw_output_path = find_raw_output_file(prompt_type,dataset,system,task,temperature,subfolder)

    with open(raw_output_path, "a", encoding='utf-8') as f:
        f.write(json.dumps(response, indent=2))
        f.write("\n\n")
        f.close()


def parse_custom_string(s):
    #sometimes ast.literal_eval does not work. it seems to have problem with certain symbols
    #I have tried different methods but couldn't resolve it
    #in case ast.literal_eval does not work, we convert string to list of tuples using regex
    pattern = r'\(\s*(["\'])(.*?)\1\s*,\s*(["\'])(.*?)\3\s*\)'
    tuples = re.findall(pattern, s)
    return [(t[1], t[3]) for t in tuples]

def parse_saved_response(prompt_type,dataset,system,task,temperature,subfolder):
    system = safe_filename(system)
    raw_output_path = find_raw_output_file(prompt_type,dataset, system, task,temperature,subfolder)
    parsed_output_path = find_parsed_output_file(prompt_type,dataset,system,task,temperature,subfolder)

    responses = []
    with open(raw_output_path, "r", encoding="utf-8") as f:
        raw = f.read()

    chunks = raw.strip().split("\n\n")
    for chunk in chunks:
        if chunk.strip():
            responses.append(json.loads(chunk))

    documents = copy.deepcopy(dataset.documents)
    annotated_documents = dict()

    for id, document in documents.items():

        response = next((d for d in responses if d.get("document_id") == id), None)
        content = response['choices'][0]['message']['content']
        # print(content)
        try:
            content = ast.literal_eval(content)
        except:
            match = re.search(r"\[.*\]", content, re.DOTALL)
            content = match.group(0)
            try:
                content = ast.literal_eval(content)
            except:
                content = parse_custom_string(content)


        annotated_document = parse_annotations(document,content)
        annotated_documents[id] = annotated_document
    save_parsed_prediction(parsed_output_path, annotated_documents)

def save_parsed_prediction(parsed_output_path,annotated_documents):
    with open(parsed_output_path,"w", encoding='utf-8') as f:
        column_line = "TOKEN	NE-COARSE-LIT	NE-COARSE-METO	NE-FINE-LIT	NE-FINE-METO	NE-FINE-COMP	NE-NESTED	NEL-LIT	NEL-METO	MISC"
        f.write(column_line)
        f.write("\n")
        for id,annotated_document in annotated_documents.items():
            id_line = "# hipe2022:document_id = " + id
            f.write(id_line)
            f.write("\n")
            for token in annotated_document:
                f.write('\t'.join(token.values()) + '\n')

        f.close()
    print(f"saved {parsed_output_path}")


def tokenize_like_annotation(text):
    # Split into words and punctuation as separate tokens
    return re.findall(r"\w+|[^\w\s]", text.strip().lower())






def parse_annotations(tokens, span_annotations):

    span_annotations_copy = copy.deepcopy(span_annotations)

    for token_dict in tokens:
        for key in token_dict:
            if key != "token":
                token_dict[key] = "_"

    def is_slice_unlabeled(start, length):
        return all(tokens[i].get('ne_coarse_lit', '_') == "_" for i in range(start, start + length))

    def string_similarity(a, b):
        return SequenceMatcher(None, a, b).ratio()

    def slice_to_str(token_slice):
        return " ".join(token_slice)

    unmatched_spans = span_annotations_copy.copy()

    #i got entries like ('Vevey', 'latis', 'loc') and it broke my code, surprise!
    removed_items = [item for item in unmatched_spans if len(item) != 2]
    unmatched_spans = [item for item in unmatched_spans if len(item) == 2]

    if removed_items:
        print("Removed malformed entries (not in (text, label) format):")
        for item in removed_items:
            print(f"  - {item}")

    #match based on similarity score
    matched_any = True
    similarity_threshold = 0.8

    while unmatched_spans and matched_any:
        matched_any = False
        # print(unmatched_spans)
        for span_text, label in unmatched_spans[:]:
            span_tokens = tokenize_like_annotation(span_text)
            span_len = len(span_tokens)
            span_str = span_text.lower()

            best_match = None
            best_score = 0


            for alt_len in range(max(1, span_len - 2), span_len + 3):
                for i in range(len(tokens) - alt_len + 1):
                    if not is_slice_unlabeled(i, alt_len):
                        continue

                    token_slice = [t['token'].lower() for t in tokens[i:i + alt_len]]
                    slice_str = slice_to_str(token_slice)
                    score = string_similarity(span_str, slice_str)

                    if score > best_score:
                        best_score = score
                        best_match = (i, alt_len)

            if best_match and best_score >= similarity_threshold:
                start, length = best_match
                # print(f"Best match: '{slice_to_str([t['token'] for t in tokens[start:start+length]])}' at position {start}, score={best_score:.3f}")
                tokens[start]['ne_coarse_lit'] = f'B-{label}'
                for j in range(1, length):
                    tokens[start + j]['ne_coarse_lit'] = f'I-{label}'

                unmatched_spans.remove((span_text, label))
                matched_any = True

    # print(f"\nNot matched spans: {unmatched_spans}")
    return tokens



from openai import OpenAI
def query(key, prompt, temperature,model="deepseek-chat"):

        client = OpenAI(api_key=key, base_url="https://api.deepseek.com")
        if model == "deepseek-chat":
            response = client.chat.completions.create(
                model=model,
                messages=prompt,
                stream=False,
                temperature = temperature
            )
        else:
            raise ValueError(f"{model} not implemented yet")
        # print(response.choices[0].message.content)

        #the model is told to generate a python list of tuples, but sometimes it might still generate some kinds of introductory text at the beginning, or unclosed list/ tuple
        #this step is to make sure that the answer can be parsed correctly later
        response = response.model_dump()
        if response['choices'][0]["finish_reason"]=="length" or "]" not in response['choices'][0]['message']['content']:
            response['choices'][0]['message']['content']=response['choices'][0]['message']['content'].rsplit(')', 1)[0]+")]"
        response['choices'][0]['message']['content'] = response['choices'][0]['message']['content'].replace("[[","[")
        response['choices'][0]['message']['content'] = response['choices'][0]['message']['content'].replace("[ [", "[")
        response['choices'][0]['message']['content'] = response['choices'][0]['message']['content'].replace("]]", "]")
        response['choices'][0]['message']['content'] = response['choices'][0]['message']['content'].replace("] ]", "]")
        return response




    # response = requests.post(
    #     url="https://openrouter.ai/api/v1/chat/completions",
    #     headers={
    #         "Authorization": f"Bearer {key}",
    #         "Content-Type": "application/json"
    #     },
    #     data=json.dumps({
    #         "model": model,
    #         "messages": [
    #             {
    #                 "role": "user",
    #                 "content": f"{prompt}"
    #             }],}))




#this version of parse_annotations is quicker but with slightly lower performance. it performs first exact match, then match based on similarity score
# def parse_annotations(tokens, span_annotations):
#     span_annotations_copy = copy.deepcopy(span_annotations)
#
#     for token_dict in tokens:
#         for key in token_dict:
#             if key != "token":
#                 token_dict[key] = "_"
#
#     def is_slice_unlabeled(start, length):
#         """Check if all tokens in the slice are unlabeled"""
#         return all(tokens[i].get('ne_coarse_lit', '_') == "_" for i in range(start, start + length))
#
#     def string_similarity(a, b):
#         """Return similarity score between 0 and 1 (1 = exact match)"""
#         return SequenceMatcher(None, a, b).ratio()
#
#     def slice_to_str(token_slice):
#         return " ".join(token_slice)
#
#     unmatched_spans = span_annotations_copy.copy()
#
#     #
#     matched_any = True
#     while unmatched_spans and matched_any:
#         matched_any = False
#
#         for span_text, label in unmatched_spans[:]:
#             span_tokens = tokenize_like_annotation1(span_text)
#             span_len = len(span_tokens)
#
#             for i in range(len(tokens) - span_len + 1):
#                 token_slice = [t['token'].lower() for t in tokens[i:i + span_len]]
#
#                 if token_slice == span_tokens and is_slice_unlabeled(i, span_len):
#                     tokens[i]['ne_coarse_lit'] = f'B-{label}'
#                     for j in range(1, span_len):
#                         tokens[i + j]['ne_coarse_lit'] = f'I-{label}'
#
#                     unmatched_spans.remove((span_text, label))
#                     matched_any = True
#                     break
#
#     # === Second Pass: Fuzzy Matching with Best Match Selection ===
#     matched_any = True
#     similarity_threshold = 0.8  # You can tune this
#     while unmatched_spans and matched_any:
#         matched_any = False
#
#         for span_text, label in unmatched_spans[:]:
#             span_tokens = tokenize_like_annotation1(span_text)
#             span_len = len(span_tokens)
#             span_str = span_text.lower()
#
#             best_match = None
#             best_score = 0
#
#             # print(f"\n Fuzzy matching span: '{span_text}' [{label}]")
#
#             for alt_len in range(max(1, span_len - 2), span_len + 3):
#                 for i in range(len(tokens) - alt_len + 1):
#                     if not is_slice_unlabeled(i, alt_len):
#                         continue
#
#                     token_slice = [t['token'].lower() for t in tokens[i:i + alt_len]]
#                     slice_str = slice_to_str(token_slice)
#                     score = string_similarity(span_str, slice_str)
#                     # if score >= similarity_threshold:
#                     #
#                     #     print(f"   Candidate slice: '{slice_str}' (len={alt_len}) → similarity: {score:.3f}")
#
#                     if score > best_score:
#                         best_score = score
#                         best_match = (i, alt_len)
#
#             if best_match and best_score >= similarity_threshold:
#                 start, length = best_match
#                 # print(f"Best match: '{slice_to_str([t['token'] for t in tokens[start:start+length]])}' at position {start}, score={best_score:.3f}")
#                 tokens[start]['ne_coarse_lit'] = f'B-{label}'
#                 for j in range(1, length):
#                     tokens[start + j]['ne_coarse_lit'] = f'I-{label}'
#
#                 unmatched_spans.remove((span_text, label))
#                 matched_any = True
#             # else:
#             #     print("No suitable fuzzy match found.")
#     #
#     # print(f"\nNot matched spans: {unmatched_spans}")
#     return tokens
