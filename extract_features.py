from collections import OrderedDict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# import timm
from torchvision import transforms
import torchvision

import argparse
import os
from tqdm import tqdm

from data.stanford_cars import CarsDataset
from data.cifar import CustomCIFAR10, CustomCIFAR100, cifar_10_root, cifar_100_root, get_cifar_100_small_datasets
from data.herbarium_19 import HerbariumDataset19, herbarium_dataroot
from data.augmentations import get_transform
from data.imagenet import get_imagenet_100_datasets
from data.data_utils import MergedDataset
from data.cub import CustomCub2011, cub_root
from data.fgvc_aircraft import FGVCAircraft, aircraft_root

from vit_model import vision_transformer as vits
from model import TwoHead, DoubleCoarseHead

# from project_utils.general_utils import strip_state_dict, str2bool
from copy import deepcopy

from config import feature_extract_dir, dino_pretrain_path

def extract_features_dino(model, loader, save_dir, extract_block=False, extract_block_num=11):

    model.to(device)
    model.eval()

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader)):
            images, labels, idxs = batch[:3]
            images = images.to(device)
            if extract_block:
                features = model.backbone.get_intermediate_layers(images, n=12 - extract_block_num)
                for layer, feat in enumerate(features, extract_block_num):
                    for f, t, uq in zip(feat, labels, idxs):
                        t = t.item()
                        uq = uq.item()
                        save_path = os.path.join(save_dir, f'layer_{layer}', f'{t}', f'{uq}.npy')
                        torch.save(f.detach().cpu().numpy(), save_path)
            else:
                features, _ = model.backbone(images)         # CLS_Token for ViT, Average pooled vector for R50
                # Save features
                for f, t, uq in zip(features, labels, idxs):

                    t = t.item()
                    uq = uq.item()

                    save_path = os.path.join(save_dir, f'{t}', f'{uq}.npy')
                    torch.save(f.detach().cpu().numpy(), save_path)


def extract_features_timm(model, loader, save_dir, extract_block=False):

    model.to(device)
    model.eval()

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader)):

            images, labels, idxs = batch[:3]
            images = images.to(device)

            features = model.forward_features(images)         # CLS_Token for ViT, Average pooled vector for R50

            # Save features
            for f, t, uq in zip(features, labels, idxs):

                t = t.item()
                uq = uq.item()

                save_path = os.path.join(save_dir, f'{t}', f'{uq}.npy')
                torch.save(f.detach().cpu().numpy(), save_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            description='cluster',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--root_dir', type=str, default=feature_extract_dir)
    parser.add_argument('--warmup_model_dir', type=str,
                        default=None)
    # parser.add_argument('--use_best_model', type='store_true', default=False)
    parser.add_argument('--model_name', type=str, default='vit_dino', help='Format is {model_name}_{pretrain}')
    parser.add_argument('--transform', type=str, default='imagenet')
    parser.add_argument('--dataset', type=str, default='aircraft', help='options: cifar10, cifar100, scars')
    parser.add_argument('--setting', type=str, default='default', help='dataset setting')
    parser.add_argument('--extract_block', action='store_true', default=False, help='extract feature from all blocks')
    parser.add_argument('--extract_block_num', default=11, type=int, help='number of block which start to extract feature, number start from 0')
    parser.add_argument('--exp_name', default=None, type=str)

    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()
    device = torch.device('cuda:0')

    args.save_dir = os.path.join(args.root_dir, f'{args.exp_name}')
    print(args)

    args.interpolation = 3
    args.crop_pct = 0.875
    # _, val_transform = get_transform('imagenet', image_size=224, args=args)
    _, val_transform = get_transform(args.transform, image_size=224, args=args)


    print('Loading data...')
    # ----------------------
    # DATASET
    # ----------------------
    if args.dataset == 'cifar10':

        train_dataset = CustomCIFAR10(root=cifar_10_root, train=True, transform=val_transform)
        test_dataset = CustomCIFAR10(root=cifar_10_root, train=False, transform=val_transform)
        targets = list(set(train_dataset.targets))

    elif args.dataset == 'cifar100':

        train_dataset = CustomCIFAR100(root=cifar_100_root, train=True, transform=val_transform)
        test_dataset = CustomCIFAR100(root=cifar_100_root, train=False, transform=val_transform)
        targets = list(set(train_dataset.targets))
        args.coarse_out_dim = 20
        args.mlp_out_dim = 100

    elif args.dataset == 'cifar100small':

        datasets = get_cifar_100_small_datasets(train_transform=val_transform, test_transform=val_transform,
                                               train_classes=range(80),
                                               prop_train_labels=0.5)
        datasets['train_labelled'].target_transform = None
        datasets['train_unlabelled'].target_transform = None
        train_dataset = MergedDataset(labelled_dataset=deepcopy(datasets['train_labelled']),
                                      unlabelled_dataset=deepcopy(datasets['train_unlabelled']),
                                      )

        test_dataset = datasets['test']
        targets = list(set(test_dataset.targets))
        args.coarse_out_dim = 20
        args.mlp_out_dim = 100

    elif args.dataset == 'scars':

        train_dataset = CarsDataset(train=True, transform=val_transform)
        test_dataset = CarsDataset(train=False, transform=val_transform)
        targets = list(set(train_dataset.target))
        targets = [i - 1 for i in targets]          # SCars are labelled 1 - 197. Change to 0 - 196

    elif args.dataset == 'herbarium_19':

        train_dataset = HerbariumDataset19(root=os.path.join(herbarium_dataroot, 'small-train'),
                                           transform=val_transform)

        test_dataset = HerbariumDataset19(root=os.path.join(herbarium_dataroot, 'small-validation'),
                                           transform=val_transform)

        targets = list(set(train_dataset.targets))

    elif args.dataset == 'imagenet_100':

        datasets = get_imagenet_100_datasets(train_transform=val_transform, test_transform=val_transform,
                                             train_classes=range(50),
                                             prop_train_labels=0.5)

        datasets['train_labelled'].target_transform = None
        datasets['train_unlabelled'].target_transform = None

        train_dataset = MergedDataset(labelled_dataset=deepcopy(datasets['train_labelled']),
                                      unlabelled_dataset=deepcopy(datasets['train_unlabelled']),
                                      )

        test_dataset = datasets['test']
        targets = list(set(test_dataset.targets))

    elif args.dataset == 'cub':

        train_dataset = CustomCub2011(root=cub_root, transform=val_transform, train=True)
        test_dataset = CustomCub2011(root=cub_root, transform=val_transform, train=False)
        targets = list(set(train_dataset.data.target.values))
        targets = [i - 1 for i in targets]          # SCars are labelled 1 - 200. Change to 0 - 199

    elif args.dataset == 'aircraft':

        train_dataset = FGVCAircraft(root=aircraft_root, transform=val_transform, split='trainval')
        test_dataset = FGVCAircraft(root=aircraft_root, transform=val_transform, split='test')
        targets = list(set([s[1] for s in train_dataset.samples]))

    else:

        raise NotImplementedError

    print('Loading model...')
    # ----------------------
    # MODEL
    # ----------------------
    if args.model_name == 'twohead':
        extract_features_func = extract_features_dino
        args.feat_dim = 768
        args.image_size = 224
        args.num_mlp_layers = 3
        backbone = vits.__dict__['vit_base']()
        projector = TwoHead(in_dim=args.feat_dim, out_dim_fine=args.mlp_out_dim, out_dim_coarse=args.coarse_out_dim, mlp_nlayers=args.num_mlp_layers)
        model = nn.Sequential(OrderedDict([
            ('backbone', backbone),
            ('projector', projector)
        ]))
    elif args.model_name == 'doublecoarse':
        extract_features_func = extract_features_dino
        args.feat_dim = 768
        args.image_size = 224
        args.num_mlp_layers = 3
        backbone = vits.__dict__['vit_base']()
        projector = DoubleCoarseHead(in_dim=args.feat_dim, out_dim_fine=args.mlp_out_dim, out_dim_coarse=args.coarse_out_dim, mlp_nlayers=args.num_mlp_layers)
        model = nn.Sequential(OrderedDict([
            ('backbone', backbone),
            ('projector', projector)
        ]))
    else:
        raise NotImplementedError

    if args.warmup_model_dir is not None:

        print(f'Using weights from {args.warmup_model_dir} ...')
        model.load_state_dict(torch.load(args.warmup_model_dir, map_location='cpu')['model'])
        if args.extract_block:
            args.save_dir += '_block'

        print(f'Saving to {args.save_dir}')

    # ----------------------
    # DATALOADER
    # ----------------------
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print('Creating base directories...')
    # ----------------------
    # INIT SAVE DIRS
    # Create a directory for each class
    # ----------------------
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if args.extract_block:
        for layer in range(args.extract_block_num, 12):
            for fold in ('train', 'test'):
                fold_dir = os.path.join(args.save_dir, fold)
                if not os.path.exists(fold_dir):
                    os.mkdir(fold_dir)
                layer_dir = os.path.join(fold_dir, f'layer_{layer}')
                if not os.path.exists(layer_dir):
                    os.mkdir(layer_dir)

                for t in targets:
                    target_dir = os.path.join(layer_dir, f'{t}')
                    if not os.path.exists(target_dir):
                        os.mkdir(target_dir)
    else:
        for fold in ('train', 'test'):

            fold_dir = os.path.join(args.save_dir, fold)
            if not os.path.exists(fold_dir):
                os.mkdir(fold_dir)

            for t in targets:
                target_dir = os.path.join(fold_dir, f'{t}')
                if not os.path.exists(target_dir):
                    os.mkdir(target_dir)

    # ----------------------
    # EXTRACT FEATURES
    # ----------------------
    # Extract train features
    train_save_dir = os.path.join(args.save_dir, 'train')
    print('Extracting features from train split...')
    extract_features_func(model=model, loader=train_loader, save_dir=train_save_dir, extract_block=args.extract_block, extract_block_num=args.extract_block_num)

    # Extract test features
    test_save_dir = os.path.join(args.save_dir, 'test')
    print('Extracting features from test split...')
    extract_features_func(model=model, loader=test_loader, save_dir=test_save_dir, extract_block=args.extract_block, extract_block_num=args.extract_block_num)

    print('Done!')