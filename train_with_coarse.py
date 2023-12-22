import argparse
import os

from collections import OrderedDict
import math
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits

from util.general_utils import AverageMeter, init_experiment, get_mean_lr
from util.cluster_and_log_utils import log_accs_from_preds, log_coarse_accs_from_preds, cluster_acc
from util.ema_utils import EMA
from config import exp_root, dino_pretrain_path, resnet_pretrain_path
from model import DINOHead, CoarseHead, TwoHead, info_nce_logits, coarse_info_nce_logits, SupConLoss, CoarseSupConLoss, DistillLoss, TCALoss, ContrastiveLearningViewGenerator, get_params_groups

from vit_model import vision_transformer as vits
from resnet_model import resnet 

def train(student, train_loader, test_loader, unlabelled_train_loader, args):
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
    start_epoch = 0

    if args.warmup_model_dir is not None:
        args.logger.info(f'Loading weights from {args.warmup_model_dir}')
        model.load_state_dict(torch.load(args.warmup_model_dir, map_location='cpu')['model'])
        optimizer.load_state_dict(torch.load(args.warmup_model_dir, map_location='cpu')['optimizer'])
        exp_lr_scheduler.load_state_dict(torch.load(args.warmup_model_dir, map_location='cpu')['scheduler'])
        start_epoch = torch.load(args.warmup_model_dir, map_location='cpu')['epoch']


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
    # coarse_weight_schedule = np.concatenate((
    #     np.linspace(args.warmup_coarse_weight,
    #                 args.coarse_weight, args.warmup_coarse_weight_epochs),
    #     np.ones(args.epochs - args.warmup_coarse_weight_epochs) * args.coarse_weight
    # ))
    coarse_weight_schedule = np.concatenate((
        np.ones(args.warmup_coarse_weight_start_epoch) * args.warmup_coarse_weight,
        np.linspace(args.warmup_coarse_weight,
                    args.coarse_weight, args.warmup_coarse_weight_end_epoch - args.warmup_coarse_weight_start_epoch),
        np.ones(args.cooloff_coarse_weight_start_epoch - args.warmup_coarse_weight_end_epoch) * args.coarse_weight,
        np.linspace(args.coarse_weight,
                    args.cooloff_coarse_weight, args.cooloff_coarse_weight_end_epoch - args.cooloff_coarse_weight_start_epoch),
        np.ones(args.epochs - args.cooloff_coarse_weight_end_epoch) * args.cooloff_coarse_weight
    ))

    for epoch in range(start_epoch, args.epochs):
        loss_record = AverageMeter()
        cls_loss_record = AverageMeter()
        cluster_loss_record = AverageMeter()
        sup_con_loss_record = AverageMeter()
        contrastive_loss_record = AverageMeter()

        coarse_cls_loss_record = AverageMeter()
        coarse_cluster_loss_record = AverageMeter()
        coarse_sup_con_loss_record = AverageMeter()
        coarse_contrastive_loss_record = AverageMeter()

        train_acc_labelled = AverageMeter()
        if args.use_coarse_label:
            train_acc_coarse_labelled = AverageMeter()

        student.train()
        for batch_idx, batch in enumerate(train_loader):
            if args.use_coarse_label:
                images, class_labels, coarse_labels, uq_idxs, mask_lab = batch
                coarse_labels = coarse_labels.cuda(non_blocking=True)
            else:
                images, class_labels, uq_idxs, mask_lab = batch
            mask_lab = mask_lab[:, 0]

            class_labels, mask_lab = class_labels.cuda(non_blocking=True), mask_lab.cuda(non_blocking=True).bool()
            images = torch.cat(images, dim=0).cuda(non_blocking=True)

            with torch.cuda.amp.autocast(fp16_scaler is not None):
                student_rep, student_pred, student_proj, student_out, student_coarse_out = student(images)
                teacher_rep = student_rep.detach()
                teacher_out = student_out.detach()
                teacher_coarse_out = student_coarse_out.detach()

                coarse_prototypes = student.projector.get_coarse_prototypes()
                fine_prototypes = student.projector.get_fine_prototypes()

                # clustering, sup
                sup_logits = torch.cat([f[mask_lab] for f in (student_out / 0.1).chunk(2)], dim=0)
                sup_labels = torch.cat([class_labels[mask_lab] for _ in range(2)], dim=0)
                cls_loss = nn.CrossEntropyLoss()(sup_logits, sup_labels)

                if args.use_coarse_label:
                    coarse_sup_labels = torch.cat([class_labels[mask_lab] for _ in range(2)], dim=0)

                # clustering, unsup
                cluster_loss = cluster_criterion(student_out, teacher_out, epoch)
                avg_probs = (student_out / 0.1).softmax(dim=1).mean(dim=0)
                me_max_loss = - torch.sum(torch.log(avg_probs**(-avg_probs))) + math.log(float(len(avg_probs)))
                cluster_loss += args.memax_weight * me_max_loss

                # coarse clustering, sup, tca
                coarse_sup_logits = torch.cat([f[mask_lab] for f in student_coarse_out.chunk(2)], dim=0)
                coarse_cls_loss = TCALoss()(coarse_logits=coarse_sup_logits, fine_labels=sup_labels, coarse_prototypes=coarse_prototypes, fine_prototypes=fine_prototypes)

                # coarse clustering, unsup
                coarse_cluster_loss = cluster_criterion(student_coarse_out, teacher_coarse_out, epoch)
                coarse_avg_probs = (student_coarse_out / 0.1).softmax(dim=1).mean(dim=0)
                coarse_me_max_loss = - torch.sum(torch.log(coarse_avg_probs**(-coarse_avg_probs))) + math.log(float(len(coarse_avg_probs)))
                coarse_cluster_loss += args.memax_weight * coarse_me_max_loss

                # represent learning, unsup
                contrastive_logits, contrastive_labels = info_nce_logits(features=student_proj)
                contrastive_loss = torch.nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)

                # coarse represent learning, unsup
                coarse_contrastive_logits, coarse_contrastive_labels = coarse_info_nce_logits(features=student_proj, prototypes=coarse_prototypes, coarse_logits=student_coarse_out)
                coarse_contrastive_loss = torch.nn.CrossEntropyLoss()(coarse_contrastive_logits, coarse_contrastive_labels)

                # representation learning, sup                
                student_proj = torch.cat([f[mask_lab].unsqueeze(1) for f in student_proj.chunk(2)], dim=1)
                student_proj = torch.nn.functional.normalize(student_proj, dim=-1)
                sup_con_labels = class_labels[mask_lab]
                sup_con_loss = SupConLoss()(student_proj, labels=sup_con_labels)

                # coarse representation learning, sup 
                student_pred = torch.cat([f[mask_lab].unsqueeze(1) for f in student_pred.chunk(2)], dim=1)
                teacher_rep = torch.cat([f[mask_lab].unsqueeze(1) for f in teacher_rep.chunk(2)], dim=1)
                sup_con_labels = class_labels[mask_lab]
                coarse_sup_con_loss = CoarseSupConLoss()(teacher_rep, student_pred, labels=sup_con_labels)

                pstr = ''
                pstr += f'cls_loss: {cls_loss.item():.4f} '
                pstr += f'cluster_loss: {cluster_loss.item():.4f} '
                pstr += f'sup_con_loss: {sup_con_loss.item():.4f} '
                pstr += f'contrastive_loss: {contrastive_loss.item():.4f} '
                pstr += f'coarse_cls_loss: {coarse_cls_loss.item():.4f} '
                pstr += f'coarse_cluster_loss: {coarse_cluster_loss.item():.4f} '
                pstr += f'coarse_sup_con_loss: {coarse_sup_con_loss.item():.4f} '
                pstr += f'coarse_contrastive_loss: {coarse_contrastive_loss.item():.4f} '

                fine_loss = 0.
                fine_loss = args.sup_weight * (cls_loss + sup_con_loss) + (1 - args.sup_weight) * (cluster_loss + contrastive_loss)
                coarse_loss = 0.
                # NOTE: all loss
                # coarse_loss = args.sup_weight * (coarse_cls_loss + coarse_sup_con_loss) + (1 - args.sup_weight) * (coarse_cluster_loss + coarse_contrastive_loss)
                # NOTE: only unsup coarse loss
                # coarse_loss = coarse_cluster_loss + coarse_contrastive_loss
                # NOTE: only coarse cluster loss
                # coarse_loss = coarse_cluster_loss
                # NOTE: coarse_cls_loss + coarse_cluster_loss
                # coarse_loss = args.sup_weight * coarse_cls_loss + (1 - args.sup_weight) * coarse_cluster_loss
                # NOTE: no coarse_cls_loss
                coarse_loss = args.sup_weight * coarse_sup_con_loss + (1 - args.sup_weight) * (coarse_cluster_loss + coarse_contrastive_loss)

                loss = 0.
                loss = args.fine_weight * fine_loss + coarse_weight_schedule[epoch] * coarse_loss
                
            # Train acc
            _, sup_pred = sup_logits.max(1)
            sup_acc = (sup_pred == sup_labels).float().mean().item()
            train_acc_labelled.update(sup_acc, sup_pred.size(0))

            if args.use_coarse_label:
                _, sup_coarse_pred = coarse_sup_logits.max(1)
                sup_coarse_acc = cluster_acc(y_true=coarse_sup_labels.cpu().numpy(), y_pred=sup_coarse_pred.cpu().numpy())
                train_acc_coarse_labelled.update(sup_coarse_acc, sup_coarse_pred.size(0))

            # NOTE: The count of cls_loss and coarse_cls_loss is not accuracy
            loss_record.update(loss.item(), class_labels.size(0))
            cls_loss_record.update(cls_loss.item(), class_labels.size(0))
            cluster_loss_record.update(cluster_loss.item(), class_labels.size(0))
            sup_con_loss_record.update(sup_con_loss.item(), class_labels.size(0))
            contrastive_loss_record.update(contrastive_loss.item(), class_labels.size(0))
            
            coarse_cls_loss_record.update(coarse_cls_loss.item(), class_labels.size(0))
            coarse_cluster_loss_record.update(coarse_cluster_loss.item(), class_labels.size(0))
            coarse_sup_con_loss_record.update(coarse_sup_con_loss.item(), class_labels.size(0))
            coarse_contrastive_loss_record.update(coarse_contrastive_loss.item(), class_labels.size(0))

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

        args.logger.info('Train Epoch: {} Avg Loss: {:.4f} '.format(epoch, loss_record.avg))

        if (epoch + 1) % args.eval_freq == 0:
            args.logger.info('Testing on unlabelled examples in the training data...')
            all_acc, old_acc, new_acc, coarse_acc = test(student, unlabelled_train_loader, epoch=epoch, save_name='Train ACC Unlabelled', args=args)
            args.logger.info('Testing on disjoint test set...')
            all_acc_test, old_acc_test, new_acc_test, coarse_acc_test = test(student, test_loader, epoch=epoch, save_name='Test ACC', args=args)
            args.logger.info('Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc, new_acc))
            args.logger.info('Test Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc_test, old_acc_test, new_acc_test))
            if args.use_coarse_label:
                args.logger.info('Train Coarse Accuracies: {:.4f}'.format(coarse_acc))
                args.logger.info('Test Coarse Accuracies: {:.4f}'.format(coarse_acc_test))

        # Step schedule
        exp_lr_scheduler.step()

        args.writer.add_scalar('Loss', loss_record.avg, epoch)
        args.writer.add_scalar('cls loss', cls_loss_record.avg, epoch)
        args.writer.add_scalar('cluster loss', cluster_loss_record.avg, epoch)
        args.writer.add_scalar('sup con loss', sup_con_loss_record.avg, epoch)
        args.writer.add_scalar('contrastive loss', contrastive_loss_record.avg, epoch)
        args.writer.add_scalar('coarse cls loss', coarse_cls_loss_record.avg, epoch)
        args.writer.add_scalar('coarse cluster loss', coarse_cluster_loss_record.avg, epoch)
        args.writer.add_scalar('coarse sup con loss', coarse_sup_con_loss_record.avg, epoch)
        args.writer.add_scalar('coarse contrastive loss', coarse_contrastive_loss_record.avg, epoch)
        args.writer.add_scalar('Train Acc Labelled Data', train_acc_labelled.avg, epoch)
        if args.use_coarse_label:
            args.writer.add_scalar('Train Acc Coarse Labelled Data', train_acc_coarse_labelled.avg, epoch)
        args.writer.add_scalar('LR', get_mean_lr(optimizer), epoch) 

        if (epoch + 1) % args.save_freq == 0:
            save_dict = {
                'model': student.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': exp_lr_scheduler.state_dict(),
                'epoch': epoch + 1,
            }
            save_path = os.path.join(args.model_dir, f'model_{epoch + 1}.pt')
            torch.save(save_dict, save_path)
            args.logger.info("model saved to {}.".format(save_path))
            # torch.save(save_dict, args.model_path)
            # args.logger.info("model saved to {}.".format(args.model_path))

        # if old_acc_test > best_test_acc_lab:
        #     
        #     args.logger.info(f'Best ACC on old Classes on disjoint test set: {old_acc_test:.4f}...')
        #     args.logger.info('Best Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc, new_acc))
        #     
        #     torch.save(save_dict, args.model_path[:-3] + f'_best.pt')
        #     args.logger.info("model saved to {}.".format(args.model_path[:-3] + f'_best.pt'))
        #     
        #     # inductive
        #     best_test_acc_lab = old_acc_test
        #     # transductive            
        #     best_train_acc_lab = old_acc
        #     best_train_acc_ubl = new_acc
        #     best_train_acc_all = all_acc
        # 
        # args.logger.info(f'Exp Name: {args.exp_name}')
        # args.logger.info(f'Metrics with best model on test set: All: {best_train_acc_all:.4f} Old: {best_train_acc_lab:.4f} New: {best_train_acc_ubl:.4f}')


def test(model, test_loader, epoch, save_name, args):

    model.eval()

    preds, targets = [], []
    coarse_acc = 0.
    if args.use_coarse_label:
        coarse_preds, coarse_targets = [], []
    mask = np.array([])
    for batch_idx, data in enumerate(tqdm(test_loader)):
        if args.use_coarse_label:
            (images, label, coarse_label, _) = data
        else:
            (images, label, _) = data
        images = images.cuda(non_blocking=True)
        with torch.no_grad():
            _, _, _, logits, coarse_logits = model(images)
            # _, logits, coarse_logits = model(images)
            preds.append(logits.argmax(1).cpu().numpy())
            targets.append(label.cpu().numpy())
            mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes)) else False for x in label]))
            if args.use_coarse_label:
                coarse_preds.append(coarse_logits.argmax(1).cpu().numpy())
                coarse_targets.append(coarse_label.cpu().numpy())

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    args=args)

    if args.use_coarse_label:
        coarse_preds = np.concatenate(coarse_preds)
        coarse_targets = np.concatenate(coarse_targets)
        coarse_acc = log_coarse_accs_from_preds(y_true=coarse_targets, y_pred=coarse_preds,
                                            T=epoch, save_name=save_name, args=args)

    return all_acc, old_acc, new_acc, coarse_acc

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='cluster', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_arch', type=str, default='vit')
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
    parser.add_argument('--save_freq', type=int, default=50, help='save frequency of model when training')

    parser.add_argument('--use_coarse_label', action='store_true', default=False)
    parser.add_argument('--sup_coarse_con_weight', type=float, default=0.5)

    parser.add_argument('--use_ema', action='store_true', default=False)
    parser.add_argument('--momentum_ema', type=float, default=0.999)
    parser.add_argument('--interval_ema', type=int, default=1, help='ema update interval')

    parser.add_argument('--fine_weight', type=float, default=1.0, help='Weight of fine-grained loss')
    parser.add_argument('--warmup_coarse_weight', type=float, default=2.0, help='Initial value for coarse_weight')
    parser.add_argument('--warmup_coarse_weight_start_epoch', type=int, default=20, help='start epoch of linear warmup')
    parser.add_argument('--warmup_coarse_weight_end_epoch', type=int, default=30, help='end epoch of linear warmup')
    parser.add_argument('--coarse_weight', type=float, default=0.5, help='value (after linear warmup, before linear cooloff) of coarse_weight')
    parser.add_argument('--cooloff_coarse_weight_start_epoch', type=int, default=20, help='start epoch of linear cooloff')
    parser.add_argument('--cooloff_coarse_weight_end_epoch', type=int, default=30, help='end epoch of linear cooloff')
    parser.add_argument('--cooloff_coarse_weight', type=float, default=2.0, help='Initial value for coarse_weight')
    

    parser.add_argument('--do_test', action='store_true', default=False)

    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()
    device = torch.device('cuda:0')
    args = get_class_splits(args)

    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    init_experiment(args, runner_name=['simgcd'])
    args.logger.info(f'Using evaluation function {args.eval_funcs[0]} to print results')
    
    torch.backends.cudnn.benchmark = True

    # ----------------------
    # BASE MODEL
    # ----------------------
    args.interpolation = 3
    args.crop_pct = 0.875
    args.mlp_out_dim = args.num_labeled_classes + args.num_unlabeled_classes
    args.coarse_out_dim = 20

    if args.model_arch == 'vit':
        args.feat_dim = 768
        args.image_size = 224
        args.num_mlp_layers = 3
        backbone = vits.__dict__['vit_base']()
        pretrain_path = dino_pretrain_path
        state_dict = torch.load(pretrain_path, map_location='cpu')
        backbone.load_state_dict(state_dict)
        # ----------------------
        # HOW MUCH OF BASE MODEL TO FINETUNE
        # ----------------------
        for m in backbone.parameters():
            m.requires_grad = False

        # Only finetune layers from block 'args.grad_from_block' onwards
        for name, m in backbone.named_parameters():
            if 'block' in name:
                block_num = int(name.split('.')[1])
                if block_num >= args.grad_from_block:
                    m.requires_grad = True
    elif args.model_arch == 'resnet18':
        args.feat_dim = 256
        args.image_size = 128
        args.num_mlp_layers = 1
        backbone = resnet.__dict__['resnet18'](num_classes=args.feat_dim)
        pretrain_path = resnet_pretrain_path
        state_dict = torch.load(pretrain_path, map_location='cpu')
        backbone.load_state_dict(state_dict, strict=False)
        # for name, param in backbone.named_parameters():
        #     if 'projector' not in name and 'layer4' not in name:
        #         param.requires_grad = False
    else:
        raise NotImplementedError

    
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
    # projector = CoarseHead(dino_in_dim=args.feat_dim, dino_out_dim=args.mlp_out_dim, dino_nlayers=args.num_mlp_layers)
    projector = TwoHead(in_dim=args.feat_dim, out_dim_fine=args.mlp_out_dim, out_dim_coarse=args.coarse_out_dim, mlp_nlayers=args.num_mlp_layers)
    # model = nn.Sequential(backbone, projector).to(device)
    model = nn.Sequential(OrderedDict([
        ('backbone', backbone),
        ('projector', projector)
    ])).to(device)

    # ----------------------
    # TRAIN
    # ----------------------
    # test(model, test_loader_labelled, epoch=None, save_name='Test ACC', args=args)
    train(model, train_loader, test_loader_labelled, test_loader_unlabelled, args)
    if args.do_test:
        if args.warmup_model_dir is None:
            raise ValueError('args.warmup_model_dir is None')
        test(model, test_loader_unlabelled, epoch=None, save_name='Train ACC Unlabelled', args=args)
        test(model, test_loader_labelled, epoch=None, save_name='Test ACC', args=args)
