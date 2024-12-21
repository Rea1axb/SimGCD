import os
import torch
import torch.cuda
import yaml
import numpy as np
import random
import json

def get_distinguish_prompt(super_class_set):
    prompt_howto = f"""
    Your task is to tell me what are the useful attributes for distinguishing [__SUPERCLASS__] categories in a photo of a [__SUPERCLASS__].

    Specifically, you can complete the task by following the instructions below:
    1 - I give you an example delimited by <> about what are the useful attributes for distinguishing bird species in 
    a photo of a bird. You should understand and learn this example carefully.
    2 - List the 5 most useful attributes for distinguishing [__SUPERCLASS__] categories in a photo of a [__SUPERCLASS__].
    3 - Output a Python list object that contains the listed useful attributes.

    ===
    example:
    <bird species>
    The useful attributes for distinguishing bird species in a photo of a bird:
    ['bill shape', 'wing color', 'upperparts color', 'underparts color', 'breast pattern'
    ]
    ===

    ===
    <[__SUPERCLASS__] categories>
    The useful attributes for distinguishing [__SUPERCLASS__] categories in a photo of a [__SUPERCLASS__]:
    ===
    """
    prompt_distinguish_dict = dict()
    for super_class in super_class_set:
        prompt_distinguish_dict[super_class] = prompt_howto.replace('[__SUPERCLASS__]', super_class)
    return prompt_distinguish_dict

def get_attr_prompt(super_class, attr):
    return f"Describe the {attr} of the {super_class} in this photo."

def get_guess_prompt(super_class, attribute_list):
    prompt = f"""
    I have a photo of a {super_class}. 
    Your task is to perform the following actions:
    1 - Summarize the information you get about the {super_class} from the general description and attribute description \
    delimited by triple backticks with five sentences.
    2 - Infer and list three possible breed names of the {super_class} in this photo based on the information you get.
    3 - Output a JSON object that uses the following format
    <three possible {super_class} breeds>: [
            <first sentence of the summary>,
            <second sentence of the summary>,
            <third sentence of the summary>,
            <fourth sentence of the summary>,
            <fifth sentence of the summary>,
    ]

    Use the following format to perform the aforementioned tasks:
    General Description: '''general description of the photo'''
    Attributes List:
    - '''attribute name''': '''attribute description'''
    - '''attribute name''': '''attribute description'''
    - ...
    - '''attribute name''': '''attribute description'''
    Summary: <summary>
    Three possible {super_class} breed names: <three possible {super_class} breed names>
    Output JSON: <output JSON object>

    Attributes List:
    """
    for attr, attr_val in attribute_list:
        prompt += f"""
        - '''{attr}''': '''{attr_val}'''
        """
    return prompt


def mkdir_if_missing(directory: str):
    if not os.path.exists(directory):
        os.makedirs(directory)

def dump_json(filename: str, in_data):
    if not filename.endswith('.json'):
        filename += '.json'

    with open(filename, 'w') as fbj:
        if isinstance(in_data, dict):
            json.dump(in_data, fbj, indent=4)
        elif isinstance(in_data, list):
            json.dump(in_data, fbj)
        else:
            raise TypeError(f"in_data has wrong data type {type(in_data)}")
        
def load_json(filename: str):
    if not filename.endswith('.json'):
        filename += '.json'
    if not os.path.exists(filename):
        with open(filename, 'w') as f:
            json.dump({}, f)
    with open(filename, 'r') as fp:
        return json.load(fp)

def setup_config(config_file_env: str, config_file_expt: str):
    with open(config_file_env, 'r') as stream:
        config_env = yaml.safe_load(stream)

    with open(config_file_expt, 'r') as stream:
        config_expt = yaml.safe_load(stream)

    cfg_env = dict()
    cfg_expt = dict()

    # Copy
    for k, v in config_env.items():
        cfg_env[k] = v

    for k, v in config_expt.items():
        cfg_expt[k] = v

    

    # for Stage Discovery
    cfg_expt['expt_dir_describe'] = os.path.join(cfg_expt['expt_dir'], "describe")
    mkdir_if_missing(cfg_expt['expt_dir_describe'])
    cfg_expt['path_vqa_questions'] = os.path.join(cfg_expt['expt_dir_describe'],
                                                  f"{cfg_expt['dataset_name']}_vqa_questions")
    cfg_expt['path_vqa_answers'] = os.path.join(cfg_expt['expt_dir_describe'],
                                                f"{cfg_expt['dataset_name']}_attributes_pairs")
    cfg_expt['path_llm_prompts'] = os.path.join(cfg_expt['expt_dir_describe'],
                                                f"{cfg_expt['dataset_name']}_llm_prompts")

    #for Stage Guess
    cfg_expt['expt_dir_guess'] = os.path.join(cfg_expt['expt_dir'], "guess")
    mkdir_if_missing(cfg_expt['expt_dir_guess'])
    cfg_expt['path_llm_replies_raw'] = os.path.join(cfg_expt['expt_dir_guess'],
                                                    f"{cfg_expt['dataset_name']}_llm_replies_raw")
    cfg_expt['path_llm_replies_jsoned'] = os.path.join(cfg_expt['expt_dir_guess'],
                                                       f"{cfg_expt['dataset_name']}_llm_replies_jsoned")
    cfg_expt['path_llm_gussed_names'] = os.path.join(cfg_expt['expt_dir_guess'],
                                                     f"{cfg_expt['dataset_name']}_llm_gussed_names")


    # for Stage Grouping evaluation
    cfg_expt['expt_dir_grouping'] = os.path.join(cfg_expt['expt_dir'], "grouping")
    mkdir_if_missing(cfg_expt['expt_dir_grouping'])

    #   |- model
    if cfg_expt['model_size'] == 'ViT-L/14@336px' and cfg_expt['image_size'] != 336:
        print(
            f'Model size is {cfg_expt["model_size"]} but image size is {cfg_expt["image_size"]}. Setting image size to 336.')
        cfg_expt['image_size'] = 336
    elif cfg_expt['model_size'] == 'RN50x4' and cfg_expt['image_size'] != 288:
        print(
            f'Model size is {cfg_expt["model_size"]} but image size is {cfg_expt["image_size"]}. Setting image size to 288.')
        cfg_expt['image_size'] = 288
    elif cfg_expt['model_size'] == 'RN50x16' and cfg_expt['image_size'] != 384:
        print(
            f'Model size is {cfg_expt["model_size"]} but image size is {cfg_expt["image_size"]}. Setting image size to 288.')
        cfg_expt['image_size'] = 384
    elif cfg_expt['model_size'] == 'RN50x64' and cfg_expt['image_size'] != 448:
        print(
            f'Model size is {cfg_expt["model_size"]} but image size is {cfg_expt["image_size"]}. Setting image size to 288.')
        cfg_expt['image_size'] = 448

    #   |- data augmentation
    return cfg_expt


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True