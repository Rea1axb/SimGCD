import torch
import os
import argparse
import json.decoder
from tqdm import tqdm
from termcolor import colored
from collections import Counter

# from utils.fileios import dump_json, load_json, dump_txt
from util.prompt_utils import seed_everything, setup_config, get_distinguish_prompt, get_attr_prompt, get_guess_prompt, dump_json, load_json

# from data import DATA_STATS, PROMPTERS, DATA_DISCOVERY
# from data.prompt_identify import prompts_howto
from agents.vqa_bot import VQABot
from agents.llm_bot import LLMBot
import re

# Debugging knob
DEBUG = False




def extract_superidentify(cfg, individual_results):
    words = []
    for v in individual_results.values():
        this_word = v.split(' ')[-1]
        words.append(this_word.lower())
    word_counts = Counter(words)

    if cfg['dataset_name'] == 'pet':
        return [super_name for super_name, _ in word_counts.most_common(2)]
    else:
        return [super_name for super_name, _ in word_counts.most_common(1)]


def extract_python_list(text):
    pattern = r"\[(.*?)\]"
    matches = re.findall(pattern, text)
    return matches


def trim_result2json(raw_reply: str):
    """
    the raw_answer is a dirty output from LLM following our template.
    this function helps to extract the target JSON content contained in the
    output.
    """
    if raw_reply.find("Output JSON:") >= 0:
        answer = raw_reply.split("Output JSON:")[1].strip()
    else:
        answer = raw_reply.strip()

    if answer.startswith('{'):
        answer = answer.removeprefix('{')
    if answer.endswith('}'): 
        answer = answer.removesuffix('}')

    if "json" in answer:
        answer = answer.strip().removeprefix("'''json").removesuffix("'''").strip()
        answer = answer.strip().removeprefix("```json").removesuffix("```").strip()

    if not answer.startswith('{'): answer = '{' + answer

    if not answer.endswith('}'): answer = answer + '}'

    # json_answer = json.loads(answer)
    return answer


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
    pattern = r'\[([^\]]*)\]'
    super_class_distinguish_attr_path = os.path.join(prompt_dir, 'super_class_distinguish_attr.json')
    super_class_distinguish_attr = load_json(super_class_distinguish_attr_path)
    # super_class_res = dict()
    for super_class, prompt in prompt_dict.items():
        if super_class in super_class_distinguish_attr:
            continue
        reply = bot.infer(prompt, temperature=0.1)
        used_tokens = bot.get_used_tokens()
        matches = re.findall(pattern, reply)
        result = matches[0].strip().replace('\n', '').replace('"', "'").replace("', '", "','")
        super_class_distinguish_attr[super_class] = result.split(',')
    dump_json(super_class_distinguish_attr_path, super_class_distinguish_attr)
    return super_class_distinguish_attr


def main_identify(bot, data_disco, prompt_dir):
    img_super_classes_path = os.path.join(prompt_dir, 'img_super_classes_result.json')
    img_super_classes = load_json(img_super_classes_path)
    # img_super_classes = {}             # img: [attr1, attr2, ..., attrN]
    super_classes_set = set()

    for img, label, uq_idxs, mask_lab in tqdm(data_disco, desc='identify'):
        if str(uq_idxs) in img_super_classes:
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

def main_describe(bot, data_disco, img_superclass_result, superclass_attr_dict, prompt_dir):
    img_describe_path = os.path.join(prompt_dir, 'img_describe.json')
    img_describe = load_json(img_describe_path)
    # img_describe_result = dict()
    for img, label, uq_idxs, mask_lab in tqdm(data_disco, desc='describe'):
        if str(uq_idxs) in img_describe:
            continue
        super_class = img_superclass_result[str(uq_idxs)]
        des_res = describe_attr(bot, img, superclass_attr_dict[super_class], super_class)
        img_describe[str(uq_idxs)] = des_res

        # DEBUG mode
        if DEBUG:
            break
    dump_json(img_describe_path, img_describe)
    return img_describe


def main_guess(bot, img_superclass_result, img_describe, prompt_dir):
    guess_raw_path = os.path.join(prompt_dir, 'guess_raw.json')
    guess_json_path = os.path.join(prompt_dir, 'guess_json.json')
    guess_raw = load_json(guess_raw_path)
    guess_json = load_json(guess_json_path)
    # replies_raw = dict()
    # replies_json_to_save = dict()

    for uq_idxs, describe in tqdm(img_describe.items(), desc='guess'):
        if str(uq_idxs) in guess_raw or str(uq_idxs) in guess_json:
            continue

        guess_prompt = get_guess_prompt(img_superclass_result[uq_idxs], describe)
        raw_reply = bot.infer(guess_prompt, temperature=0.9)
        used_tokens = bot.get_used_tokens()
        jsoned_reply = trim_result2json(raw_reply=raw_reply)

        guess_raw[str(uq_idxs)] = raw_reply
        guess_json[str(uq_idxs)] = jsoned_reply
    dump_json(guess_raw_path, guess_raw)
    dump_json(guess_json_path, guess_json)
    return guess_raw, guess_json


def main_result(data_disco, img_describe, img_guess_json):
    img_describe_result = dict()
    img_guess_json_result = dict()
    for img, label, uq_idxs, mask_lab in tqdm(data_disco, desc='main_result'):
        img_describe_result[str(uq_idxs)] = img_describe[str(uq_idxs)]
        img_guess_json_result[str(uq_idxs)] = img_guess_json[str(uq_idxs)]
    return img_describe_result, img_guess_json_result


def post_process(jsoned_replies):
    reply_list = []
    num_of_failures = 0
    # duplicated dict
    for k, v in jsoned_replies.items():
        # print(k)
        # print(v)
        # print()
        # print()
        rex = r"""(?<=[}\]"'])\s*,\s*(?!\s*[{["'])""" # 去除多余逗号
        v_json = json.loads(re.sub(rex, "", v, 0))
        reply_list.append(v_json)

        # v_json = json.loads(v)ypp
        # reply_list.append(v_json)

    guessed_names = []
    for item in reply_list:
        item_keys = list(item.keys())
        for name_list in item_keys:
            guessed_names.extend(name_list.split(','))

    guessed_names = extract_names(guessed_names, clean=False)


    # print(30 * '=')
    # print(f"\t\t Finished Post-processing")
    # print(30 * '=')

    # print(f"\t\t ---> total discovered names = {len(guessed_names)}")
    # print(guessed_names)
    # print()
    # print(f"\t\t ---> total discovered names = {len(guessed_names)}")
    # print(f"\t\t ---> number of failure entries = {num_of_failures}")

    # print('END' + 30 * '=')
    # print()
    return guessed_names


def get_query_result(img_data, model_size_vqa='FlanT5-XL', model_type_llm='moonshot-v1-8k', device='cuda', device_id=0, prompt_dir=None):
    if not os.path.exists(prompt_dir):
        os.makedirs(prompt_dir)

    # step 1
    vqa_bot = VQABot(model_tag=model_size_vqa, device=device, device_id=device_id, bit8=False)
    img_superclass_result, superclass_set = main_identify(vqa_bot, img_data, prompt_dir)

    # step 2
    llm_bot = LLMBot(model=model_type_llm, temperature=0.1)
    distinguish_prompt_dict = get_distinguish_prompt(superclass_set)
    superclass_distinguish_attr = how_to_distinguish(llm_bot, distinguish_prompt_dict, prompt_dir)

    # step 3
    img_describe = main_describe(vqa_bot, img_data, img_superclass_result, superclass_distinguish_attr, prompt_dir)

    # step 4
    guess_raw, guess_json = main_guess(llm_bot, img_superclass_result, img_describe, prompt_dir)

    # get img_data result
    img_describe_result, img_guess_json_result = main_result(img_data, img_describe, guess_json)

    # get class name
    guess_result_path = os.path.join(prompt_dir, 'guess_result.json')

    dump_json(guess_result_path, img_guess_json_result)        
    jsoned_replies = load_json(guess_result_path)

    guessed_names = post_process(jsoned_replies)
    # TODO only use img_guess_json_result
    return img_describe_result, img_guess_json_result, guessed_names

def get_pseudo_labels(img_data, clip_model, clip_processor, label2name, img_guess_json_result, device='cuda', device_id=0):
    img_pseudo_label = dict()
    img_describe_text = dict()
    with torch.no_grad():
        for img, label, uq_idxs, mask_lab in tqdm(img_data, desc='pseudo_labels'):
            describe_text = list(img_guess_json_result[str(uq_idxs)].values())[0]
            img_describe_text[str(uq_idxs)] = describe_text
            if mask_lab:
                img_pseudo_label[str(uq_idxs)] = label2name[label]
                continue
            label_list = list(img_guess_json_result[str(uq_idxs)].keys())[0].split(',')
            inputs = clip_processor(text=label_list, images=img, return_tensors="pt", padding=True)
            outputs = clip_model(**inputs)
            logits_per_image = outputs.logits_per_image
            _, num = logits_per_image.max()
            img_pseudo_label[str(uq_idxs)] = label_list[num]
    return img_pseudo_label, img_describe_text


def get_prototypes(img_data, clip_model, clip_processor, img_pseudo_label, img_describe_text, device_id=0):
    pseudo_label_features = dict()
    prototypes = dict()
    with torch.no_grad():
        for img, label, uq_idxs, mask_lab in tqdm(img_data, desc='prototypes'):
            pseudo_label = img_pseudo_label[str(uq_idxs)]
            describe_text = img_describe_text[str(uq_idxs)]
            device = f"cuda:{device_id}"
            # 对图片进行编码
            image_features = clip_model.get_image_features(
                clip_processor(images=img, return_tensors="pt").to(device)["pixel_values"]
            ).detach().cpu()

            # 对文本描述进行编码
            text_features = clip_model.get_text_features(
                clip_processor(text=describe_text, return_tensors="pt").to(device)["input_ids"]
            ).detach().cpu()

            # 累加特征
            if pseudo_label not in pseudo_label_features:
                pseudo_label_features[pseudo_label] = {"image": [], "text": []}

            pseudo_label_features[pseudo_label]["image"].append(image_features)
            pseudo_label_features[pseudo_label]["text"].append(text_features)
        
        for pseudo_label, features in pseudo_label_features.items():
            image_prototype = torch.mean(torch.stack(features["image"]), dim=0)
            text_prototype = torch.mean(torch.stack(features["text"]), dim=0)
            prototypes[pseudo_label] = (image_prototype, text_prototype)
        
    return prototypes

if __name__ == "__main__":
    from PIL import Image
    img_path_1 = "/home/czq/2024_12_11_clip/labelled/red_fox/n02119022_28.JPEG"
    img_path_2 = "/home/czq/2024_12_11_clip/labelled/ferret/n02443484_14.JPEG"
    img_path_3 = "/home/czq/2024_12_11_clip/labelled/ferret/n02443484_70.JPEG"
    img_1 = Image.open(img_path_1)
    img_2 = Image.open(img_path_2)
    img_list = [(img_1, 1, 0, True), (img_2, 2, 1, True)]
    get_query_result(img_data=img_list, prompt_dir='./outputs/LLM4GCD/prompt/test')

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



