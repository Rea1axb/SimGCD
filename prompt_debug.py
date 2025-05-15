import torch
from torchvision import transforms

def debug_prompt():
    from PIL import Image
    from transformers import CLIPProcessor, CLIPModel
    from config import exp_root, dino_pretrain_path, clip_pretrain_path
    from get_prompt import get_pseudo_labels, get_two_class_distinguish
    from util.clip_utils import clip_finetune, get_new_prototypes

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
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整大小
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.ToTensor(),  # 转换为张量
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
    ])

    sample_imgs = [img_1, img_2, img_3, img_4, img_5]
    sample_class_labels = torch.tensor([1, 2, 2, 3, 3])
    sample_uq_idxs = torch.tensor([0, 1, 2, 3, 4])
    sample_mask_lab = torch.tensor([True, True, True, False, False])
    img_pseudolabel_list = list(zip(sample_imgs, sample_class_labels, sample_uq_idxs, sample_mask_lab))
    # img_distinguish_list = [(img_5, 3, 4, False)]

    # img_pseudolabel_list = [(img_1, 1, 0, True), (img_2, 2, 1, True), (img_3, 2, 2, True), (img_4, 3, 3, False)]
    # img_distinguish_list = [(img_5, 3, 4, False)]
    two_class_dict = {"4": ["ferret", "red_fox"], "5": ["bobtail", "ferret"]}
    label2name = {0: "red_fox", 1: "ferret", 2: "ferret", 3: "bobtail"}

    device = "cuda:0"
    clip_model = CLIPModel.from_pretrained(clip_pretrain_path).to(device)
    
    clip_processor = CLIPProcessor.from_pretrained(clip_pretrain_path)

    img_pseudo_label_res, img_describe_text_res, is_valid = get_pseudo_labels(
        img_data=img_pseudolabel_list, 
        clip_model=clip_model, 
        clip_processor=clip_processor, 
        label2name=label2name, 
        model_type_llm='qwen-turbo',
        prompt_dir='./outputs/LLM4GCD/prompt/test'
    )
    # img_distinguish_res = get_two_class_distinguish(
    #     img_data=img_distinguish_list, 
    #     two_class_dict=two_class_dict, 
    #     model_type_llm='qwen-turbo',
    #     prompt_dir='./outputs/LLM4GCD/prompt/test'
    # )
    for param in clip_model.parameters():
        param.requires_grad = False
    
    # last_block = clip_model.vision_model.encoder.layers[-1]
    # vision_proj = clip_model.visual_projection

    vision_last_block = clip_model.vision_model.encoder.layers[-1]
    text_last_block = clip_model.text_model.encoder.layers[-1]
    vision_proj = clip_model.visual_projection
    text_proj = clip_model.text_projection

    for param in vision_last_block.parameters():
        param.requires_grad = True

    for param in vision_proj.parameters():
        param.requires_grad = True

    for param in text_last_block.parameters():
        param.requires_grad = True

    for param in text_proj.parameters():
        param.requires_grad = True
    
    clip_finetune(clip_model, clip_processor, sample_imgs, sample_class_labels, sample_uq_idxs, sample_mask_lab, img_describe_text_res, transform=transform, train_epoch=3, device_id=0)
    prototypes = get_new_prototypes(clip_model, clip_processor, img_pseudolabel_list, img_pseudo_label_res, img_describe_text_res, device_id=0)

def debug_sample():
    from util.general_utils import three_stage_sampling
    all_distances = torch.randn(32, 10).abs()
    sample_dict = three_stage_sampling(all_distances, 5, 3, 2)

if __name__ == "__main__":
    debug_prompt()
    # debug_sample()    