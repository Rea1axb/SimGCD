import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

def get_params_groups(model):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]

class CLIPDataset(Dataset):
    def __init__(self, images, labels, uq_idxs, mask_lab, text_dict, transform=None):
        self.images = images
        self.labels = labels
        self.uq_idxs = uq_idxs
        self.mask_lab = mask_lab
        self.text_dict = text_dict
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        uq_idx = self.uq_idxs[idx].item()
        mask = self.mask_lab[idx]
        # text = self.text_dict[str(uq_idx)]
        text = " ".join(self.text_dict[str(uq_idx)])
        return image, text, label, mask
    

def clip_finetune(clip_model, clip_processor, sample_imgs, sample_class_labels, sample_uq_idxs, sample_mask_lab, img_describe_text_res, transform=None, train_epoch=3, device_id=0):
    # if transform is None:
    #     transform = transforms.Compose([
    #         transforms.ToTensor(),  # 转换为张量
    #     ])
    clip_dataset = CLIPDataset(sample_imgs, sample_class_labels, sample_uq_idxs, sample_mask_lab, img_describe_text_res, transform)
    data_loader = DataLoader(clip_dataset, batch_size=32, shuffle=True)


    # for param in clip_model.parameters():
    #     param.requires_grad = False
    
    # vision_last_block = clip_model.vision_model.encoder.layers[-1]
    # text_last_block = clip_model.text_model.encoder.layers[-1]
    # vision_proj = clip_model.visual_projection
    # text_proj = clip_model.text_projection

    # for param in vision_last_block.parameters():
    #     param.requires_grad = True

    # for param in vision_proj.parameters():
    #     param.requires_grad = True

    # for param in text_last_block.parameters():
    #     param.requires_grad = True

    # for param in text_proj.parameters():
    #     param.requires_grad = True

    params_groups = get_params_groups(clip_model)


    optimizer = optim.AdamW(params_groups, lr=5e-6)
    criterion = nn.CrossEntropyLoss()

    clip_model.train()

    device = f"cuda:{device_id}"

    for epoch in range(train_epoch):  
        for batch_idx, batch in enumerate(data_loader):
            images, texts, labels, masks = batch
            images = images
            labels = labels

            # 处理文本
            inputs = clip_processor(text=texts, images=images, return_tensors="pt", truncation=True, padding=True, do_rescale=False).to(device)

            # 前向传播
            outputs = clip_model(**inputs)
            logits_per_image = outputs.logits_per_image  # 图像和文本之间的相似度
            logits_per_text = outputs.logits_per_text

            # 计算损失
            target = torch.arange(len(images)).to(device)  # 假设对比学习
            loss_img = criterion(logits_per_image, target)
            loss_text = criterion(logits_per_text, target)
            loss = (loss_img + loss_text) / 2

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Fine-tune CLIP Epoch {epoch + 1}, Loss: {loss.item():.4f}")


def get_new_prototypes(clip_model, clip_processor, sample_bank, img_pseudo_label_bank, img_describe_text_bank, device_id=0):
    """
    Generate new prototypes based on the given inputs.

    Args:
        clip_model (torch.nn.Module): The CLIP model used for encoding images and text.
        clip_processor (CLIPProcessor): The CLIP processor used for preprocessing images and text.
        sample_bank (list): A sample bank.
        img_pseudo_label_bank (dict): A dictionary mapping unique indices to pseudo labels.
        img_describe_text_bank (dict): A dictionary mapping unique indices to text descriptions.
        device_id (int, optional): The ID of the device to use for computation. Defaults to 0.

    Returns:
        dict: A dictionary mapping pseudo labels to their corresponding image and text prototypes.
    """


    device = f"cuda:{device_id}"
    pseudo_label_features = dict()
    prototypes = dict()
    with torch.no_grad():
        for img, label, uq_idxs, mask_lab in tqdm(sample_bank, desc='prototypes'):
            uq_idxs = uq_idxs.item()
            pseudo_label = img_pseudo_label_bank[str(uq_idxs)]
            describe_text = " ".join(img_describe_text_bank[str(uq_idxs)])
            # 对图片进行编码
            image_features = clip_model.get_image_features(
                clip_processor(images=img, return_tensors="pt", do_rescale=False).to(device)["pixel_values"]
            ).detach().cpu()

            # 对文本描述进行编码
            text_features = clip_model.get_text_features(
                clip_processor(text=describe_text, return_tensors="pt", truncation=True, padding=True).to(device)["input_ids"]
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

# if __name__ == "__main__":
#     from PIL import Image
#     from transformers import CLIPProcessor, CLIPModel
#     import sys
#     sys.path.append("..")
#     from config import exp_root, dino_pretrain_path, clip_pretrain_path
#     from get_prompt import get_pseudo_labels, get_two_class_distinguish

#     img_path_1 = "/home/czq/2024_12_11_clip/labelled/red_fox/n02119022_28.JPEG"
#     img_path_2 = "/home/czq/2024_12_11_clip/labelled/ferret/n02443484_14.JPEG"
#     img_path_3 = "/home/czq/2024_12_11_clip/labelled/ferret/n02443484_70.JPEG"
#     img_path_4 = "/home/czq/2024_12_11_clip/test/bobtail/ILSVRC2012_val_00002247.JPEG"
#     img_path_5 = "/home/czq/2024_12_11_clip/test/bobtail/ILSVRC2012_val_00003413.JPEG"
#     img_1 = Image.open(img_path_1)
#     img_2 = Image.open(img_path_2)
#     img_3 = Image.open(img_path_3)
#     img_4 = Image.open(img_path_4)
#     img_5 = Image.open(img_path_5)

#     sample_imgs = [img_1, img_2, img_3, img_4, img_5]
#     sample_class_labels = torch.tensor([1, 2, 2, 3, 3])
#     sample_uq_idxs = torch.tensor([0, 1, 2, 3, 4])
#     sample_mask_lab = torch.tensor([True, True, True, False, False])
#     img_pseudolabel_list = list(zip(sample_imgs, sample_class_labels, sample_uq_idxs, sample_mask_lab))
#     img_distinguish_list = [(img_5, 3, 4, False)]

#     # img_pseudolabel_list = [(img_1, 1, 0, True), (img_2, 2, 1, True), (img_3, 2, 2, True), (img_4, 3, 3, False)]
#     # img_distinguish_list = [(img_5, 3, 4, False)]
#     two_class_dict = {"4": ["ferret", "red_fox"], "5": ["bobtail", "ferret"]}
#     label2name = {0: "red_fox", 1: "ferret", 2: "ferret", 3: "bobtail"}

#     device = "cuda:0"

#     clip_model = CLIPModel.from_pretrained(clip_pretrain_path)
#     clip_model.to(device)
#     clip_processor = CLIPProcessor.from_pretrained(clip_pretrain_path)

#     img_pseudo_label_res, img_describe_text_res = get_pseudo_labels(
#         img_data=img_pseudolabel_list, 
#         clip_model=clip_model, 
#         clip_processor=clip_processor, 
#         label2name=label2name, 
#         model_type_llm='qwen-turbo',
#         prompt_dir='./outputs/LLM4GCD/prompt/test'
#     )
#     img_distinguish_res = get_two_class_distinguish(
#         img_data=img_distinguish_list, 
#         two_class_dict=two_class_dict, 
#         model_type_llm='qwen-turbo',
#         prompt_dir='./outputs/LLM4GCD/prompt/test'
#     )
#     clip_finetune(clip_model, clip_processor, sample_imgs, sample_class_labels, sample_uq_idxs, sample_mask_lab, img_describe_text_res, train_epoch=3, device_id=0)
#     prototypes = get_new_prototypes(clip_model, clip_processor, img_pseudolabel_list, img_pseudo_label_res, img_describe_text_res, device_id=0)


