import torch
import os
import argparse
import json.decoder
from tqdm import tqdm
from termcolor import colored
from collections import Counter

# from utils.fileios import dump_json, load_json, dump_txt
from util.prompt_utils import seed_everything, setup_config, get_distinguish_prompt, get_attr_prompt, get_guess_prompt, get_distinguish_two_class_prompt, dump_json, load_json

# from data import DATA_STATS, PROMPTERS, DATA_DISCOVERY
# from data.prompt_identify import prompts_howto
from agents.vqa_bot import VQABot
from agents.llm_bot import LLMBot
import re

# Debugging knob
DEBUG = False

#TODO: valid json

def is_json(reply):
    """
    检查接口返回内容是否为有效的 JSON 格式。
    Args:
        reply (str): 接口返回的字符串。
    Returns:
        bool: 如果是 JSON 格式，返回 True,否则返回 False。
    """
    reply = reply.strip()
    if not reply.startswith("{") or not reply.endswith("}"):
        return False
    try:
        # 尝试解析为 JSON
        data = json.loads(reply)
        return True
    except json.JSONDecodeError:
        return False

def is_guess_json(reply):
    reply = reply.strip()
    if not reply.startswith("{") or not reply.endswith("}"):
        return False
    try:
        required_keys = {"three possible names", "information summary"}
        
        # 尝试解析为 JSON
        data = json.loads(reply)

        # 检查是否是字典类型
        if not isinstance(data, dict):
            return False
        
        # 检查是否包含所有必需的键
        if not required_keys.issubset(data.keys()):
            return False
        
        # 检查每个键的值是否是列表
        for key in required_keys:
            if not isinstance(data[key], list):
                return False

        return True
    except json.JSONDecodeError:
        return False

def clean_name(name: str):
    name = name.title()
    name = name.replace("-", " ")
    name = name.replace("'s", "")
    return name


def extract_names(gussed_names, clean=True):
    gussed_names = [name.strip() for name in gussed_names]
    if clean:
        gussed_names = [clean_name(name) for name in gussed_names]
    gussed_names = list(set(gussed_names))
    return gussed_names


def how_to_distinguish(bot, prompt_dict, prompt_dir):
    """
    Distinguishes super classes using a bot and prompts.

    Args:
        bot (Bot): The bot used for inference.
        prompt_dict (dict): A dictionary containing the prompts for each super class.
        prompt_dir (str): The directory path where the prompt files are stored.

    Returns:
        tuple: A tuple containing two dictionaries:
            - super_class_distinguish_attr: A dictionary mapping each super class to its distinguishing attributes.
            - super_class_valid: A dictionary indicating whether each super class was successfully distinguished.
    """
    pattern = r'\[([^\]]*)\]'
    super_class_distinguish_attr_path = os.path.join(prompt_dir, 'super_class_distinguish_attr.json')
    super_class_distinguish_attr = load_json(super_class_distinguish_attr_path)
    super_class_valid = dict()
    
    for super_class, prompt in prompt_dict.items():
        if super_class in super_class_distinguish_attr:
            super_class_valid[super_class] = True
            continue
        reply = bot.infer(prompt, temperature=0.1, return_json=True)
        used_tokens = bot.get_used_tokens()

        # Check if the reply is a valid JSON
        if not is_json(reply):
            super_class_valid[super_class] = False
            continue
        
        jsoned_reply = json.loads(reply)
        super_class_distinguish_attr[super_class] = list(jsoned_reply.values())[0]
        super_class_valid[super_class] = True
    
    dump_json(super_class_distinguish_attr_path, super_class_distinguish_attr)
    return super_class_distinguish_attr, super_class_valid


def main_identify(bot, data_disco, prompt_dir):
    """
    Identify the main object category in each image and store the results.

    Args:
        bot: VQA model.
        data_disco: A list of tuples containing image data.
        prompt_dir: The directory path where prompt files are stored.

    Returns:
        img_super_classes: A dictionary containing the image super classes.
        super_classes_set: A set containing the super classes.
    """
    img_super_classes_path = os.path.join(prompt_dir, 'img_super_classes_result.json')
    img_super_classes = load_json(img_super_classes_path)
    # img_super_classes = {}             # img: [attr1, attr2, ..., attrN]
    super_classes_set = set()

    for img, label, uq_idxs, mask_lab in tqdm(data_disco, desc='identify'):
        uq_idxs = uq_idxs.item()
        if str(uq_idxs) in img_super_classes:
            super_classes_set.add(img_super_classes[str(uq_idxs)])
            continue
        # prompt_identify = "Question: What is the main object in this image (choose from: Car, Flower, or Pokemon)? Answer:"
        prompt_identify = "Question: What is the category of the main object in this image? Answer:"

        reply, trimmed_reply = bot.describe_attribute(img, prompt_identify)
        trimmed_reply = trimmed_reply.lower()
        img_super_classes[str(uq_idxs)] = trimmed_reply
        super_classes_set.add(trimmed_reply)

        # DEBUG mode
        if DEBUG:
            break
    dump_json(img_super_classes_path, img_super_classes)
    return img_super_classes, super_classes_set

def describe_attr(bot, img, attr_list, super_class):
    pair_attr_reply = []
    for attr in attr_list:
        attr_prompt = get_attr_prompt(super_class, attr)
        re_attr, trimmed_re_attr = bot.describe_attribute(img, attr_prompt)
        pair_attr_reply.append([attr, trimmed_re_attr])
    return pair_attr_reply

def main_describe(bot, data_disco, img_superclass, superclass_attr_dict, prompt_dir, super_class_valid):
    """
    Describe the attributes of images in the given dataset.

    Args:
        bot (Bot): The bot object.
        data_disco (list) : The list of image data.
        img_superclass (dict): The dictionary mapping image indices to superclass labels.
        superclass_attr_dict (dict): The dictionary mapping superclass labels to attribute lists.
        prompt_dir (str): The directory path for prompt files.
        super_class_valid (list): List of boolean values indicating the validity of each superclass.

    Returns:
        img_describe: A dictionary containing the descriptions of the images.

    """
    img_describe_path = os.path.join(prompt_dir, 'img_describe.json')
    img_describe = load_json(img_describe_path)
    # img_describe_result = dict()
    is_valid = [True] * len(data_disco)
    for i, (img, label, uq_idxs, mask_lab) in enumerate(tqdm(data_disco, desc='describe')):
        uq_idxs = uq_idxs.item()
        if str(uq_idxs) in img_describe:
            continue
        super_cls = img_superclass[str(uq_idxs)]
        if super_cls not in super_class_valid or not super_class_valid[super_cls]:
            is_valid[i] = False
            continue

        des_res = describe_attr(bot, img, superclass_attr_dict[super_cls], super_cls)
        img_describe[str(uq_idxs)] = des_res

        # DEBUG mode
        if DEBUG:
            break
    dump_json(img_describe_path, img_describe)
    return img_describe, is_valid


def main_guess(bot, data_disco, img_superclass, img_describe, prompt_dir, super_class_valid, is_valid):
    """
    Perform guessing for a given set of images.

    Args:
        bot (Bot): The bot used for inference.
        data_disco (list): List of image data.
        img_superclass (dict): Dictionary mapping image indices to superclass labels.
        img_describe (dict): Dictionary mapping image indices to descriptions.
        prompt_dir (str): Directory path for prompt files.
        super_class_valid (list): List of boolean values indicating the validity of each superclass.
        is_valid (list): List of boolean values indicating the validity of each image.
    Returns:
        tuple: A tuple containing the following:
            - guess_json (dict): Dictionary mapping image indices to generated guesses.
            - img_describe_result (dict): Dictionary mapping image indices to descriptions.
            - img_guess_json_result (dict): Dictionary mapping image indices to generated guesses.
            - is_valid (list): List of boolean values indicating the validity of each image.
    """
    guess_json_path = os.path.join(prompt_dir, 'guess_json.json')
    guess_json = load_json(guess_json_path)

    img_describe_result = dict()
    img_guess_json_result = dict()

    assert len(is_valid) == len(data_disco), "Length of is_valid and data_disco do not match."

    for i, (img, label, uq_idxs, mask_lab) in enumerate(tqdm(data_disco, desc='guess')):
        uq_idxs = uq_idxs.item()
        if not is_valid[i]:
            continue
        describe = img_describe[str(uq_idxs)]
        if str(uq_idxs) in guess_json:
            img_describe_result[str(uq_idxs)] = describe
            img_guess_json_result[str(uq_idxs)] = guess_json[str(uq_idxs)]
            continue
        
        # super_class，非法
        super_cls = img_superclass[str(uq_idxs)]
        if super_cls not in super_class_valid or not super_class_valid[super_cls]:
            is_valid[i] = False
            continue

        guess_prompt = get_guess_prompt(super_cls, describe)
        jsoned_reply = bot.infer(guess_prompt, temperature=0.9, return_json=True)

        # 返回的不是json，非法
        if not is_guess_json(jsoned_reply):
            is_valid[i] = False
            continue
        
        guess_json[str(uq_idxs)] = jsoned_reply
        img_describe_result[str(uq_idxs)] = describe
        img_guess_json_result[str(uq_idxs)] = jsoned_reply

    dump_json(guess_json_path, guess_json)
    return guess_json, img_describe_result, img_guess_json_result, is_valid

def main_distinguish_two_class(bot, data_disco, two_class_dict, img_superclass, img_describe):
    """
    Perform the main process of distinguishing two classes for each image in the given dataset.

    Args:
        bot (Bot): The bot object used for inference.
        data_disco (list): The list of data containing images, labels, unique indices, and masks.
        two_class_dict (dict): A dictionary mapping unique indices to two classes.
        img_superclass (dict): A dictionary mapping unique indices to super classes.
        img_describe (dict): A dictionary mapping unique indices to image descriptions.

    Returns:
        dict: A dictionary mapping unique indices to the inference results.

    """
    distinguish_two_class_result = dict()
    for img, label, uq_idxs, mask_lab in tqdm(data_disco, desc='distiguish_two_class'):
        uq_idxs = uq_idxs.item()
        class1, class2 = two_class_dict[str(uq_idxs)]
        describe = img_describe[str(uq_idxs)]
        super_class = img_superclass[str(uq_idxs)]
        distinguish_two_class_prompt = get_distinguish_two_class_prompt(super_class, class1, class2, describe)
        jsoned_reply = bot.infer(distinguish_two_class_prompt, temperature=0.1, return_json=True)
        distinguish_two_class_result[str(uq_idxs)] = jsoned_reply
    return distinguish_two_class_result



def get_query_result(img_data, model_size_vqa='FlanT5-XL', model_type_llm='moonshot-v1-8k', device='cuda', device_id=0, prompt_dir=None, mode='get_pseudo_labels', two_class_dict=None):
    """
    Retrieves the query result based on the provided image data and parameters.

    Args:
        img_data (list): List of image data.
        model_size_vqa (str, optional): Size of the VQA model. Defaults to 'FlanT5-XL'.
        model_type_llm (str, optional): Type of the LLM model. Defaults to 'moonshot-v1-8k'.
        device (str, optional): Device to use for computation. Defaults to 'cuda'.
        device_id (int, optional): ID of the device to use. Defaults to 0.
        prompt_dir (str, optional): Directory to store prompts. Must be specified. Defaults to None.
        mode (str, optional): Mode of operation. Defaults to 'get_pseudo_labels'.
        two_class_dict (dict, optional): Dictionary containing two classes. Must be specified if mode is 'get_distinguish_two_class'. Defaults to None.

    Returns:
        tuple: A tuple containing the following elements:
            - img_describe_result (list): List of image descriptions.
            - img_guess_json_result (list): List of image guess JSON results.
            - is_valid (bool): Flag indicating if the result is valid.

    Raises:
        AssertionError: If prompt_dir is not specified when mode is 'get_pseudo_labels'.
        AssertionError: If two_class_dict is not specified when mode is 'get_distinguish_two_class'.
    """

    assert prompt_dir is not None, "prompt_dir is not specified"

    if not os.path.exists(prompt_dir):
        os.makedirs(prompt_dir)

    # step 1
    vqa_bot = VQABot(model_tag=model_size_vqa, device=device, device_id=device_id, bit8=False)
    # img_superclass: all data
    # superclass_set: selected data
    img_superclass, superclass_set = main_identify(vqa_bot, img_data, prompt_dir)

    # step 2
    llm_bot = LLMBot(model=model_type_llm, temperature=0.1)
    distinguish_prompt_dict = get_distinguish_prompt(superclass_set)
    superclass_distinguish_attr, superclass_valid = how_to_distinguish(llm_bot, distinguish_prompt_dict, prompt_dir)

    # step 3
    # img_describe: all data
    img_describe, is_valid = main_describe(vqa_bot, img_data, img_superclass, superclass_distinguish_attr, prompt_dir, superclass_valid) 

    # step 4
    if mode == 'get_pseudo_labels':
        # guess_json: all data
        # img_describe_result: selected data
        # img_guess_json_result: selected data
        guess_json, img_describe_result, img_guess_json_result, is_valid = main_guess(llm_bot, img_data, img_superclass, img_describe, prompt_dir, superclass_valid, is_valid)

        # get class name
        guess_result_path = os.path.join(prompt_dir, 'guess_result.json')

        dump_json(guess_result_path, img_guess_json_result)        
        jsoned_replies = load_json(guess_result_path)


        return img_describe_result, img_guess_json_result, is_valid
    elif mode == 'get_distinguish_two_class':
        assert two_class_dict is not None, "two_class_dict is not specified"
        distinguish_two_class_result = main_distinguish_two_class(llm_bot, img_data, two_class_dict, img_superclass, img_describe)
        return distinguish_two_class_result

def get_pseudo_labels(img_data, clip_model, clip_processor, label2name, model_size_vqa='FlanT5-XL', model_type_llm='moonshot-v1-8k', device='cuda', device_id=0, prompt_dir=None):
    """
    Generates pseudo labels and description text for a given set of image data.

    Args:
        img_data (torch.utils.data.Dataset): The image data.
        clip_model (torch.nn.Module): The CLIP model.
        clip_processor (transformers.CLIPProcessor): The CLIP processor.
        label2name (dict): A dictionary mapping label indices to label names.
        model_size_vqa (str, optional): The size of the VQA model. Defaults to 'FlanT5-XL'.
        model_type_llm (str, optional): The type of the LLM model. Defaults to 'moonshot-v1-8k'.
        device (str, optional): The device to run the model on. Defaults to 'cuda'.
        device_id (int, optional): The ID of the device. Defaults to 0.
        prompt_dir (str, optional): The directory containing prompt files. Defaults to None.

    Returns:
        Tuple[Dict[str, str], Dict[str, str], List[bool]]: A tuple containing the pseudo labels, description text, and validity flags.
            - img_pseudo_label (Dict[str, str]): A dictionary mapping unique indices to pseudo labels.
            - img_describe_text (Dict[str, str]): A dictionary mapping unique indices to description text.
            - is_valid (List[bool]): A list indicating the validity of each image.

    """
    img_describe_result, img_guess_json_result, is_valid = get_query_result(
        img_data=img_data, 
        model_size_vqa=model_size_vqa,
        model_type_llm=model_type_llm,
        device=device, 
        device_id=device_id, 
        prompt_dir=prompt_dir, 
        mode='get_pseudo_labels'
    )

    img_pseudo_label = dict()
    img_describe_text = dict()
    with torch.no_grad():
        for i, (img, label, uq_idxs, mask_lab) in enumerate(tqdm(img_data, desc='pseudo_labels')):
            if not is_valid[i]:
                continue
            uq_idxs = uq_idxs.item()
            img_guess_json = json.loads(img_guess_json_result[str(uq_idxs)])
            describe_text = img_guess_json['information summary']
            img_describe_text[str(uq_idxs)] = describe_text
            if mask_lab:
                img_pseudo_label[str(uq_idxs)] = label2name[label.item()]
                continue
            label_list = img_guess_json['three possible names']
            inputs = clip_processor(text=label_list, images=img, return_tensors="pt", padding=True, truncation=True, do_rescale=False).to(f'{device}:{device_id}')
            outputs = clip_model(**inputs)
            logits_per_image = outputs.logits_per_image
            num = logits_per_image.argmax()
            img_pseudo_label[str(uq_idxs)] = label_list[num]
    return img_pseudo_label, img_describe_text, is_valid

def get_two_class_distinguish(img_data, two_class_dict, model_size_vqa='FlanT5-XL', model_type_llm='moonshot-v1-8k', device='cuda', device_id=0, prompt_dir=None):
    """
    Get the distinguish result for two classes based on the given image data.

    Args:
        img_data (numpy.ndarray): The image data.
        two_class_dict (dict): A dictionary containing the two classes to distinguish.
        model_size_vqa (str, optional): The size of the VQA model. Defaults to 'FlanT5-XL'.
        model_type_llm (str, optional): The type of the LLM model. Defaults to 'moonshot-v1-8k'.
        device (str, optional): The device to run the models on. Defaults to 'cuda'.
        device_id (int, optional): The ID of the device. Defaults to 0.
        prompt_dir (str, optional): The directory containing the prompt files. Defaults to None.

    Returns:
        dict: A dictionary containing the distinguish result for the two classes.
    """

    distinguish_two_class_result = get_query_result(
        img_data=img_data, 
        model_size_vqa=model_size_vqa,
        model_type_llm=model_type_llm,
        device=device, 
        device_id=device_id, 
        prompt_dir=prompt_dir, 
        mode='get_distinguish_two_class', 
        two_class_dict=two_class_dict
    )

    return distinguish_two_class_result
    

if __name__ == "__main__":
    from PIL import Image
    from transformers import CLIPProcessor, CLIPModel
    from config import exp_root, dino_pretrain_path, clip_pretrain_path
    img_path_1 = "/home/czq/2024_12_11_clip/labelled/red_fox/n02119022_28.JPEG"
    img_path_2 = "/home/czq/2024_12_11_clip/labelled/ferret/n02443484_14.JPEG"
    img_path_3 = "/home/czq/2024_12_11_clip/labelled/ferret/n02443484_70.JPEG"
    img_path_4 = "/home/czq/2024_12_11_clip/test/bobtail/ILSVRC2012_val_00002247.JPEG"
    img_path_5 = "/home/czq/2024_12_11_clip/test/bobtail/ILSVRC2012_val_00003413.JPEG"
    img_1 = Image.open(img_path_1)
    img_2 = Image.open(img_path_2)
    img_3 = Image.open(img_path_3)
    img_4 = Image.open(img_path_4)
    img_5 = Image.open(img_path_5)

    img_pseudolabel_list = [(img_1, 1, 0, True), (img_2, 2, 1, True), (img_3, 2, 2, True), (img_4, 3, 3, False)]
    img_distinguish_list = [(img_5, 3, 4, False)]
    two_class_dict = {"4": ["ferret", "red_fox"], "5": ["bobtail", "ferret"]}
    label2name = {0: "red_fox", 1: "ferret", 2: "ferret", 3: "bobtail"}

    clip_model = CLIPModel.from_pretrained(clip_pretrain_path)
    clip_processor = CLIPProcessor.from_pretrained(clip_pretrain_path)

    img_pseudo_label_res, img_describe_text_res = get_pseudo_labels(
        img_data=img_pseudolabel_list, 
        clip_model=clip_model, 
        clip_processor=clip_processor, 
        label2name=label2name, 
        model_type_llm='qwen-turbo',
        prompt_dir='./outputs/LLM4GCD/prompt/test'
    )
    img_distinguish_res = get_two_class_distinguish(
        img_data=img_distinguish_list, 
        two_class_dict=two_class_dict, 
        model_type_llm='qwen-turbo',
        prompt_dir='./outputs/LLM4GCD/prompt/test'
    )

    # get_query_result(img_data=img_list, prompt_dir='./outputs/LLM4GCD/prompt/test')

    # parser = argparse.ArgumentParser(description='Discovery', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # parser.add_argument('--mode',
    #                     type=str,
    #                     default='describe',
    #                     choices=['identify', 'howto', 'describe', 'guess', 'postprocess'],
    #                     help='operating mode for each stage')
    # parser.add_argument('--config_file_env',
    #                     type=str,
    #                     default='./configs/env_machine.yml',
    #                     help='location of host environment related config file')
    # parser.add_argument('--config_file_expt',
    #                     type=str,
    #                     default='./configs/expts/bird200_all.yml',
    #                     help='location of host experiment related config file')



