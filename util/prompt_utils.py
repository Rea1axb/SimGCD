import os
import torch
import torch.cuda
import yaml
import numpy as np
import random
import json

def get_distinguish_prompt(super_class_set):
    """
    Generate a dictionary of prompts for distinguishing attributes of different superclasses.

    Args:
        super_class_set (set): A set of superclass names.

    Returns:
        dict: A dictionary where the keys are the superclass names and the values are the corresponding prompts.

    """
    prompt_howto = """
    Your task is to tell me what are the useful attributes for distinguishing [__SUPERCLASS__] categories in a photo of a [__SUPERCLASS__].

    Specifically, you can complete the task by following the instructions below:
    1 - I give you an example about what are the useful attributes for distinguishing dog species in 
    a photo of a dog. You should understand and learn this example carefully.
    2 - List the most useful attributes for distinguishing [__SUPERCLASS__] class in [__SUPERCLASS__] photos, sorted by importance from most to least.
    3 - Output a JSON object that contains the listed useful attributes.(filling in the content between the angle brackets <> in 'Output')

    === Example:
    {'dog': ['ear shape', 'tail length', 'fur color', 'snout length', 'body size', 'pattern']}
    ===

    === Output:
    {
        [__SUPERCLASS__]:
        <The list of useful attributes for distinguishing [__SUPERCLASS__] categories in a photo of a [__SUPERCLASS__]>
    }
    ===
    """
    prompt_distinguish_dict = dict()
    for super_class in super_class_set:
        prompt_distinguish_dict[super_class] = prompt_howto.replace('[__SUPERCLASS__]', super_class)
    return prompt_distinguish_dict

def get_attr_prompt(super_class, attr):
    return f"Describe the {attr} of the {super_class} in this photo."

def get_guess_prompt(super_class, attribute_list):
    """
    Generates a prompt for guessing the attributes and summary of a given superclass based on a photo.

    Args:
        super_class (str): The name of the superclass.
        attribute_list (list): A list of tuples containing attribute names and their corresponding values.

    Returns:
        str: The generated prompt for guessing the attributes and summary.

    Example Usage:
        super_class = "Dog"
        attribute_list = [("Ears", "Semi-erect ears with the tips folding forward slightly."),
                          ("Tail", "Medium length with a slight curve."),
                          ("Snout", "Moderately long."),
                          ("Fur Color", "Black and white."),
                          ("Size", "Medium-sized.")]

        prompt = get_guess_prompt(super_class, attribute_list)
        print(prompt)
    """

    prompt = """
    I have a photo of a [__SUPERCLASS__]. 
    Your task is to perform the following actions:
    1 - Summarize the information you get about the [__SUPERCLASS__] from the Attributes List delimited by triple backticks with five sentences.
    2 - Infer three possible breed names and five sentence of summary of the [__SUPERCLASS__] in this photo based on the information you get.
    3 - Output a JSON object that uses the 'Example' and 'Output' format.(filling in the content between the angle brackets <> in 'Output')

    === Example:
    {
        'three possible names': ['Border Collie', 'Cocker Spaniels', 'Dobermans']
        'information summary': [
            'It typically have semi-erect ears with the tips folding forward slightly.', 
            'The tail is of medium length and usually have a slight curve.', 
            'It have a moderately long snout.', 
            'The fur color is black and white.', 
            'It is a medium-sized dog.',
        ]
    }
    ===

    === Output:
    {
        'three possible names': [<name_1>, <name_2>, <name_3>]
        'information summary': [
            <summary_sentence_1>, 
            <summary_sentence_2>,
            <summary_sentence_3>,
            <summary_sentence_4>,
            <summary_sentence_5>,
        ]
    }
    ===
    
    Use the following format to perform the aforementioned tasks:
    Attributes List:
    """
    for attr, attr_val in attribute_list:
        prompt += f"""
        - '''{attr}''': '''{attr_val}'''
        """
    prompt = prompt.replace('[__SUPERCLASS__]', super_class)
    return prompt

def get_distinguish_two_class_prompt(super_class, class_1, class_2, attribute_list):
    prompt = """
    I have a photo of a [__SUPERCLASS__]. 
    Your task is to perform the following actions:
    1 - Learn the information you get about the [__SUPERCLASS__] from Attributes List delimited by triple backticks.
    2 - Determine whether the [__SUPERCLASS__] in this photo based on the information you get is more likely to be [__CLASS1__] or [__CLASS1__] ?
    3 - Output a JSON object that uses the following format
    
    === Example:
    {
        'class': 'Border Collie'
    }
    ===

    === Output:
    {
        'class': <[__CLASS1__](or [__CLASS2__])>
    }
    ===

    Use the following format to perform the aforementioned tasks:
    Attributes List:
    """
    for attr, attr_val in attribute_list:
        prompt += f"""
        - '''{attr}''': '''{attr_val}'''
        """
    prompt = prompt.replace('[__SUPERCLASS__]', super_class)
    prompt = prompt.replace('[__CLASS1__]', class_1)
    prompt = prompt.replace('[__CLASS2__]', class_2)
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