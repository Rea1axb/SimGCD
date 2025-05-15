import argparse

import math
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.augmentations import get_transform, denormalize
from data.get_datasets import get_datasets, get_class_splits, get_name2label_label2name

from util.general_utils import AverageMeter, init_experiment, get_mean_lr, three_stage_sampling, label_based_sampling
from util.cluster_and_log_utils import log_accs_from_preds
from util.ema_utils import EMA
from util.clip_utils import clip_finetune, get_new_prototypes
from config import exp_root, dino_pretrain_path, clip_pretrain_path
from get_prompt import get_pseudo_labels, get_two_class_distinguish
from model import DINOHead, LLMHead, info_nce_logits, SupConLoss, DistillLoss, ContrastiveLearningViewGenerator, get_params_groups

# from vit_model import vision_transformer as vits
from transformers import CLIPModel, CLIPProcessor

def update_mask_lab_and_class_labels(mask_lab_batch, class_labels_batch, uq_idxs_batch, pseudo_label_dict):
    """
    Updates the mask_lab_batch and class_labels_batch based on the pseudo_label_dict.

    Args:
        mask_lab_batch (list): A list of boolean values indicating whether each sample should be masked or not.
        class_labels_batch (list): A list of class labels for each sample.
        uq_idxs_batch (list): A list of unique indices for each sample.
        pseudo_label_dict (dict): A dictionary containing pseudo labels for certain unique indices.

    Returns:
        tuple: A tuple containing the updated mask_lab_batch and class_labels_batch.
    """
    for i, uq_idx in enumerate(uq_idxs_batch):
        if str(uq_idx) in pseudo_label_dict:
            mask_lab_batch[i] = True
            class_labels_batch[i] = pseudo_label_dict[str(uq_idx)]
    return mask_lab_batch, class_labels_batch

def train(student, clip_processor, train_loader, test_loader, unlabelled_train_loader, args):
    params_groups = get_params_groups(student)
    optimizer = SGD(params_groups, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    fp16_scaler = None
    if args.fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.lr * 1e-3,
        )


    cluster_criterion = DistillLoss(
                        args.warmup_teacher_temp_epochs,
                        args.epochs,
                        args.n_views,
                        args.warmup_teacher_temp,
                        args.teacher_temp,
                    )

    # # inductive
    # best_test_acc_lab = 0
    # # transductive
    # best_train_acc_lab = 0
    # best_train_acc_ubl = 0 
    # best_train_acc_all = 0
    img_pseudo_label_bank = {}
    img_describe_text_bank = {}
    sample_bank = []
    all_similarity = []
    all_imgs = []
    all_class_labels = []
    all_mask_lab = []
    all_uq_idxs = []

    for epoch in range(args.epochs):
        loss_record = AverageMeter()
        cls_loss_record = AverageMeter()
        cluster_loss_record = AverageMeter()
        sup_con_loss_record = AverageMeter()
        contrastive_loss_record = AverageMeter()

        train_acc_labelled = AverageMeter()

        student.train()
        for batch_idx, batch in enumerate(train_loader):
            images, class_labels, uq_idxs, mask_lab = batch
            mask_lab = mask_lab[:, 0]

            mask_lab, class_labels = update_mask_lab_and_class_labels(mask_lab, class_labels, uq_idxs, pseudo_label_dict=img_pseudo_label_bank)

            class_labels, mask_lab = class_labels.cuda(non_blocking=True), mask_lab.cuda(non_blocking=True).bool()
            images = torch.cat(images, dim=0).cuda(non_blocking=True)

            with torch.cuda.amp.autocast(fp16_scaler is not None):
                student_proj, student_out = student.projector(student.clip_model.get_image_features(images))
                teacher_out = student_out.detach()

                # clustering, sup
                sup_logits = torch.cat([f[mask_lab] for f in (student_out / 0.1).chunk(2)], dim=0)
                sup_labels = torch.cat([class_labels[mask_lab] for _ in range(2)], dim=0)
                cls_loss = nn.CrossEntropyLoss()(sup_logits, sup_labels)

                # clustering, unsup
                cluster_loss = cluster_criterion(student_out, teacher_out, epoch)
                avg_probs = (student_out / 0.1).softmax(dim=1).mean(dim=0)
                me_max_loss = - torch.sum(torch.log(avg_probs**(-avg_probs))) + math.log(float(len(avg_probs)))
                cluster_loss += args.memax_weight * me_max_loss

                # represent learning, unsup
                contrastive_logits, contrastive_labels = info_nce_logits(features=student_proj)
                contrastive_loss = torch.nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)

                # representation learning, sup                
                student_proj = torch.cat([f[mask_lab].unsqueeze(1) for f in student_proj.chunk(2)], dim=1)
                student_proj = torch.nn.functional.normalize(student_proj, dim=-1)
                # fine-label
                sup_con_labels = class_labels[mask_lab]
                sup_con_loss = SupConLoss()(student_proj, labels=sup_con_labels)

                pstr = ''
                pstr += f'cls_loss: {cls_loss.item():.4f} '
                pstr += f'cluster_loss: {cluster_loss.item():.4f} '
                pstr += f'sup_con_loss: {sup_con_loss.item():.4f} '
                pstr += f'contrastive_loss: {contrastive_loss.item():.4f} '


                loss = 0.
                loss = (1 - args.sup_weight) * cluster_loss + args.sup_weight * cls_loss + \
                        (1 - args.sup_weight) * contrastive_loss + args.sup_weight * sup_con_loss
                
            # Train acc
            _, sup_pred = sup_logits.max(1)
            sup_acc = (sup_pred == sup_labels).float().mean().item()
            train_acc_labelled.update(sup_acc, sup_pred.size(0))

            loss_record.update(loss.item(), class_labels.size(0))
            cls_loss_record.update(cls_loss.item(), class_labels.size(0))
            cluster_loss_record.update(cluster_loss.item(), class_labels.size(0))
            sup_con_loss_record.update(sup_con_loss.item(), class_labels.size(0))
            contrastive_loss_record.update(contrastive_loss.item(), class_labels.size(0))

            optimizer.zero_grad()
            if fp16_scaler is None:
                loss.backward()
                optimizer.step()
            else:
                fp16_scaler.scale(loss).backward()
                fp16_scaler.step(optimizer)
                fp16_scaler.update()

            if batch_idx % args.print_freq == 0:
                args.logger.info('Epoch: [{}][{}/{}]\t loss {:.5f}\t {}'
                            .format(epoch, batch_idx, len(train_loader), loss.item(), pstr))
            
            if (epoch + 1) % args.query_freq == 0:
                all_similarity.append(teacher_out.chunk(2)[0].detach().cpu())
                all_imgs.append(images.chunk(2)[0].detach().cpu())
                all_class_labels.append(class_labels)
                all_mask_lab.append(mask_lab)
                all_uq_idxs.append(uq_idxs)

        args.logger.info('Train Epoch: {} Avg Loss: {:.4f} '.format(epoch, loss_record.avg))

        if (epoch + 1) % args.eval_freq == 0:
            args.logger.info('Testing on unlabelled examples in the training data...')
            all_acc, old_acc, new_acc = test(student, unlabelled_train_loader, epoch=epoch, save_name='Train ACC Unlabelled', args=args)
            args.logger.info('Testing on disjoint test set...')
            all_acc_test, old_acc_test, new_acc_test = test(student, test_loader, epoch=epoch, save_name='Test ACC', args=args)


            args.logger.info('Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc, new_acc))
            args.logger.info('Test Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc_test, old_acc_test, new_acc_test))



        # Step schedule
        exp_lr_scheduler.step()

        args.writer.add_scalar('Loss', loss_record.avg, epoch)
        args.writer.add_scalar('cls loss', cls_loss_record.avg, epoch)
        args.writer.add_scalar('cluster loss', cluster_loss_record.avg, epoch)
        args.writer.add_scalar('sup con loss', sup_con_loss_record.avg, epoch)
        args.writer.add_scalar('contrastive loss', contrastive_loss_record.avg, epoch)
        args.writer.add_scalar('Train Acc Labelled Data', train_acc_labelled.avg, epoch)
        args.writer.add_scalar('LR', get_mean_lr(optimizer), epoch) 

        if (epoch + 1) % args.save_freq == 0:
            save_dict = {
                'model': student.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1,
            }
            torch.save(save_dict, args.model_path)
            args.logger.info("model saved to {}.".format(args.model_path))
        

        if (epoch + 1) % args.query_freq == 0:
            with torch.no_grad():
                all_similarity = torch.cat(all_similarity, dim=0)
                all_imgs = torch.cat(all_imgs, dim=0)
                all_class_labels = torch.cat(all_class_labels, dim=0)
                all_mask_lab = torch.cat(all_mask_lab, dim=0)
                all_uq_idxs = torch.cat(all_uq_idxs, dim=0)

                unlabel_similarity = all_similarity[~all_mask_lab]
                unlabel_imgs = all_imgs[~all_mask_lab]
                unlabel_imgs = denormalize(unlabel_imgs)
                unlabel_class_labels = all_class_labels[~all_mask_lab]
                unlabel_mask_lab = all_mask_lab[~all_mask_lab]
                unlabel_uq_idxs = all_uq_idxs[~all_mask_lab]

                label_similarity = all_similarity[all_mask_lab]
                label_imgs = all_imgs[all_mask_lab]
                label_imgs = denormalize(label_imgs)
                label_class_labels = all_class_labels[all_mask_lab]
                label_mask_lab = all_mask_lab[all_mask_lab]
                label_uq_idxs = all_uq_idxs[all_mask_lab]

                # unlabel
                sample_dict, nearest_two_class = three_stage_sampling(unlabel_similarity, args.n_samples_1, args.n_samples_2, args.n_samples_3)
                # label
                label_indices = label_based_sampling(label_class_labels, args.n_samples_label)


                pseudo_sample_imgs = torch.cat([unlabel_imgs[sample_dict['stage_1']], unlabel_imgs[sample_dict['stage_3']], label_imgs[label_indices]], dim=0)
                pseudo_sample_class_labels = torch.cat([unlabel_class_labels[sample_dict['stage_3']], unlabel_class_labels[sample_dict['stage_3']], label_class_labels[label_indices]], dim=0)
                pseudo_sample_uq_idxs = torch.cat([unlabel_uq_idxs[sample_dict['stage_1']], unlabel_uq_idxs[sample_dict['stage_3']], label_uq_idxs[label_indices]], dim=0)
                pseudo_sample_mask_lab = torch.cat([unlabel_mask_lab[sample_dict['stage_1']], unlabel_mask_lab[sample_dict['stage_3']], label_mask_lab[label_indices]], dim=0)

                distinguish_sample_imgs = unlabel_imgs[sample_dict['stage_2']]
                distinguish_sample_class_labels = unlabel_class_labels[sample_dict['stage_2']]
                distinguish_sample_uq_idxs = unlabel_uq_idxs[sample_dict['stage_2']]
                distinguish_sample_mask_lab = unlabel_mask_lab[sample_dict['stage_2']]
                img_pseudolabel_list = list(zip(pseudo_sample_imgs, pseudo_sample_class_labels, pseudo_sample_uq_idxs, pseudo_sample_mask_lab))
                img_distinguish_list = list(zip(distinguish_sample_imgs, distinguish_sample_class_labels, distinguish_sample_uq_idxs, distinguish_sample_mask_lab))

                
                #TODO:vqa load once
                img_pseudo_label_res, img_describe_text_res, is_valid = get_pseudo_labels(
                    img_data=img_pseudolabel_list, 
                    clip_model=student.clip_model, 
                    clip_processor=clip_processor, 
                    label2name=student.projector.label2name, 
                    model_type_llm='qwen-turbo',
                    prompt_dir=args.prompt_dir
                )
                img_pseudo_label_bank.update(img_pseudo_label_res)
                img_describe_text_bank.update(img_describe_text_res)

                valid_img_pseudolabel_list = list(zip(pseudo_sample_imgs[is_valid], pseudo_sample_class_labels[is_valid], pseudo_sample_uq_idxs[is_valid], pseudo_sample_mask_lab[is_valid]))
                sample_bank.extend(valid_img_pseudolabel_list)

                #TODO: distinguish to use
                # nearest_two_class_dict = {str(uq_idx.item()):(student.projector.label2name[nearest_two_class[idx][0]], student.projector.label2name[nearest_two_class[idx][1]]) for idx, uq_idx in enumerate(distinguish_sample_uq_idxs)}
                # img_distinguish_res = get_two_class_distinguish(
                #     img_data=img_distinguish_list,
                #     two_class_dict=nearest_two_class_dict,
                #     model_type_llm='qwen-turbo',
                #     prompt_dir=args.prompt_dir

                # )

            clip_finetune(
                clip_model=student.clip_model, 
                clip_processor=clip_processor, 
                sample_imgs=pseudo_sample_imgs[is_valid], 
                sample_class_labels=pseudo_sample_class_labels[is_valid], 
                sample_uq_idxs=pseudo_sample_uq_idxs[is_valid], 
                sample_mask_lab=pseudo_sample_mask_lab[is_valid], 
                img_describe_text_res=img_describe_text_res, 
                train_epoch=args.clip_train_epochs, 
                device_id=0
            )

            with torch.no_grad():
                new_prototypes = get_new_prototypes(
                    clip_model=student.clip_model,
                    clip_processor=clip_processor,
                    sample_bank=sample_bank,
                    img_pseudo_label_bank=img_pseudo_label_bank,
                    img_describe_text_bank=img_describe_text_bank,
                ) # dict
                
                student.projector.update_prototypes(new_prototypes)

            all_similarity = []
            all_imgs = []
            all_class_labels = []
            all_mask_lab = []
            all_uq_idxs = []


def test(model, test_loader, epoch, save_name, args):

    model.eval()

    preds, targets = [], []
    mask = np.array([])
    for batch_idx, data in enumerate(tqdm(test_loader)):
        (images, label, _) = data
        images = images.cuda(non_blocking=True)
        with torch.no_grad():
            _, logits = model.projector(model.clip_model.get_image_features(images))
            preds.append(logits.argmax(1).cpu().numpy())
            targets.append(label.cpu().numpy())
            mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes)) else False for x in label]))

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    args=args)

    return all_acc, old_acc, new_acc


def get_init_prototypes(model):
    pass

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='cluster', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v1', 'v2', 'v2b'])

    parser.add_argument('--warmup_model_dir', type=str, default=None)
    parser.add_argument('--dataset_name', type=str, default='scars', help='options: cifar10, cifar100, imagenet_100, cub, scars, fgvc_aricraft, herbarium_19')
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--use_ssb_splits', action='store_true', default=True)

    parser.add_argument('--grad_from_block', type=int, default=11)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--exp_root', type=str, default=exp_root)
    parser.add_argument('--transform', type=str, default='imagenet')
    parser.add_argument('--sup_weight', type=float, default=0.35)
    parser.add_argument('--n_views', default=2, type=int)
    
    parser.add_argument('--memax_weight', type=float, default=2)
    parser.add_argument('--warmup_teacher_temp', default=0.07, type=float, help='Initial value for the teacher temperature.')
    parser.add_argument('--teacher_temp', default=0.04, type=float, help='Final value (after linear warmup)of the teacher temperature.')
    parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int, help='Number of warmup epochs for the teacher temperature.')

    parser.add_argument('--fp16', action='store_true', default=False)
    parser.add_argument('--print_freq', default=10, type=int)
    parser.add_argument('--exp_name', default=None, type=str)
    parser.add_argument('--setting', type=str, default='default', help='dataset setting')
    parser.add_argument('--eval_freq', type=int, default=10, help='eval frequency when training')
    parser.add_argument('--save_freq', type=int, default=10, help='save frequency when training')

    parser.add_argument('--use_coarse_label', action='store_true', default=False)
    parser.add_argument('--sup_coarse_con_weight', type=float, default=0.5)

    parser.add_argument('--use_ema', action='store_true', default=False)
    parser.add_argument('--momentum_ema', type=float, default=0.999)
    parser.add_argument('--interval_ema', type=int, default=1, help='ema update interval')

    parser.add_argument('--n_samples_1', type=int, default=5, help='clip finetune epoch')
    parser.add_argument('--n_samples_2', type=int, default=5, help='clip finetune epoch')
    parser.add_argument('--n_samples_3', type=int, default=5, help='clip finetune epoch')
    parser.add_argument('--n_samples_label', type=int, default=5, help='clip finetune epoch')
    parser.add_argument('--clip_train_epochs', type=int, default=1, help='clip finetune epoch')
    parser.add_argument('--query_freq', type=int, default=10, help='query frequency')
    parser.add_argument('--prompt_dir', type=str, default=None)

    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()
    device = torch.device('cuda:0')
    args = get_class_splits(args)

    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    init_experiment(args, runner_name=['LLM4GCD'])
    args.logger.info(f'Using evaluation function {args.eval_funcs[0]} to print results')
    
    torch.backends.cudnn.benchmark = True

    # ----------------------
    # BASE MODEL
    # ----------------------
    args.interpolation = 3
    args.crop_pct = 0.875

    # backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
    clip_model = CLIPModel.from_pretrained(clip_pretrain_path)
    clip_processor = CLIPProcessor.from_pretrained(clip_pretrain_path)
    # backbone = vits.__dict__['vit_base']()
    # pretrain_path = dino_pretrain_path
    # state_dict = torch.load(pretrain_path, map_location='cpu')
    # backbone.load_state_dict(state_dict)

    
    
    # NOTE: Hardcoded image size as we do not finetune the entire ViT model
    args.image_size = 224
    args.feat_dim = 512
    args.num_mlp_layers = 3
    # args.mlp_out_dim = args.num_labeled_classes + args.num_unlabeled_classes
    args.mlp_out_dim = args.num_labeled_classes

    # ----------------------
    # HOW MUCH OF BASE MODEL TO FINETUNE
    # ----------------------

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

    # for param in last_block.parameters():
    #     param.requires_grad = True

    # for param in vision_proj.parameters():
    #     param.requires_grad = True
    
    args.logger.info('model build')

    # --------------------
    # CONTRASTIVE TRANSFORM
    # --------------------
    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=args.n_views)
    # --------------------
    # DATASETS
    # --------------------
    train_dataset, test_dataset, unlabelled_train_examples_test, datasets = get_datasets(args.dataset_name,
                                                                                         train_transform,
                                                                                         test_transform,
                                                                                         args)

    # --------------------
    # SAMPLER
    # Sampler which balances labelled and unlabelled examples in each batch
    # --------------------
    label_len = len(train_dataset.labelled_dataset)
    unlabelled_len = len(train_dataset.unlabelled_dataset)
    sample_weights = [1 if i < label_len else label_len / unlabelled_len for i in range(len(train_dataset))]
    sample_weights = torch.DoubleTensor(sample_weights)
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(train_dataset))

    # --------------------
    # DATALOADERS
    # --------------------
    train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False,
                              sampler=sampler, drop_last=True, pin_memory=True)
    test_loader_unlabelled = DataLoader(unlabelled_train_examples_test, num_workers=args.num_workers,
                                        batch_size=256, shuffle=False, pin_memory=False)
    test_loader_labelled = DataLoader(test_dataset, num_workers=args.num_workers,
                                      batch_size=256, shuffle=False, pin_memory=False)

    # ----------------------
    # PROJECTION HEAD
    # ----------------------
    # projector = DINOHead(in_dim=args.feat_dim, out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers)
    init_name2label, init_label2name = get_name2label_label2name(args.dataset_name, test_dataset, args.train_classes)
    projector = LLMHead(
        in_dim=args.feat_dim, 
        initial_out_dim=args.mlp_out_dim, 
        init_name2label=init_name2label,
        init_label2name=init_label2name,
        nlayers=args.num_mlp_layers
    )
    model = nn.ModuleDict({
        'clip_model': clip_model,
        'projector': projector,
    }).to(device)
    # model = nn.Sequential(clip_model, projector).to(device)

    if args.warmup_model_dir is not None:
        args.logger.info(f'Loading weights from {args.warmup_model_dir}')
        model.load_state_dict(torch.load(args.warmup_model_dir, map_location='cpu')['model'])
    # ----------------------
    # TRAIN
    # ----------------------
    # test(model, test_loader_labelled, epoch=None, save_name='Test ACC', args=args)
    # if args.use_ema:
    #     backbone_t = vits.__dict__['vit_base']()
    #     projector_t = DINOHead(in_dim=args.feat_dim, out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers)
    #     model_t = nn.Sequential(backbone_t, projector_t).to(device)
    #     train_ema(model, model_t, train_loader, test_loader_labelled, test_loader_unlabelled, args)
    # else:
    train(model, clip_processor, train_loader, test_loader_labelled, test_loader_unlabelled, args)
