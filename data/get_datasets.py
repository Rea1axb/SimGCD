from data.data_utils import MergedDataset, get_cifar100_coarse_labels_dict

from data.cifar import get_cifar_10_datasets, get_cifar_100_datasets, get_cifar_100_small_datasets
from data.herbarium_19 import get_herbarium_datasets
from data.stanford_cars import get_scars_datasets
from data.imagenet import get_imagenet_100_datasets, get_imagenet_1k_datasets, get_imagenet_datasets, get_imagenet_200_datasets, get_imagenet_10_datasets
from data.cub import get_cub_datasets
from data.fgvc_aircraft import get_aircraft_datasets

from copy import deepcopy
import pickle
import os

from config import osr_split_dir


get_dataset_funcs = {
    'cifar10': get_cifar_10_datasets,
    'cifar100': get_cifar_100_datasets,
    'cifar100small': get_cifar_100_small_datasets,
    'imagenet': get_imagenet_datasets,
    'imagenet_200': get_imagenet_200_datasets,
    'imagenet_10': get_imagenet_10_datasets,
    'imagenet_100': get_imagenet_100_datasets,
    'imagenet_1k': get_imagenet_1k_datasets,
    'herbarium_19': get_herbarium_datasets,
    'cub': get_cub_datasets,
    'aircraft': get_aircraft_datasets,
    'scars': get_scars_datasets
}


def get_datasets(dataset_name, train_transform, test_transform, args):

    """
    :return: train_dataset: MergedDataset which concatenates labelled and unlabelled
             test_dataset,
             unlabelled_train_examples_test,
             datasets
    """

    #
    if dataset_name not in get_dataset_funcs.keys():
        raise ValueError

    # Get datasets
    get_dataset_f = get_dataset_funcs[dataset_name]
    datasets = get_dataset_f(train_transform=train_transform, test_transform=test_transform,
                            train_classes=args.train_classes,
                            prop_train_labels=args.prop_train_labels,
                            split_train_val=True,
                            use_coarse_label=args.use_coarse_label, args=args)
    # Set target transforms:
    target_transform_dict = {}
    for i, cls in enumerate(list(args.train_classes) + list(args.unlabeled_classes)):
        target_transform_dict[cls] = i
    target_transform = lambda x: target_transform_dict[x]

    for dataset_name, dataset in datasets.items():
        if dataset is not None:
            dataset.target_transform = target_transform

    # Train split (labelled and unlabelled classes) for training
    train_dataset = MergedDataset(labelled_dataset=deepcopy(datasets['train_labelled']),
                                  unlabelled_dataset=deepcopy(datasets['train_unlabelled']),
                                  use_coarse_label=args.use_coarse_label)

    test_dataset = datasets['test']
    unlabelled_train_examples_test = deepcopy(datasets['train_unlabelled'])
    unlabelled_train_examples_test.transform = test_transform

    return train_dataset, test_dataset, unlabelled_train_examples_test, datasets


def get_class_splits(args):

    # For FGVC datasets, optionally return bespoke splits
    if args.dataset_name in ('scars', 'cub', 'aircraft'):
        if hasattr(args, 'use_ssb_splits'):
            use_ssb_splits = args.use_ssb_splits
        else:
            use_ssb_splits = False

    # -------------
    # GET CLASS SPLITS
    # -------------
    if args.dataset_name == 'cifar10':

        args.image_size = 32
        if args.setting == 'default':
            args.train_classes = range(5)
            args.unlabeled_classes = range(5, 10)
        elif args.setting == 'animal_6_transportation_0_0.5':
            args.train_classes = [2, 3, 4, 5, 6, 7]
            args.unlabeled_classes = [0, 1, 8, 9]
        elif args.setting == 'animal_0_transportation_4_0.5':
            args.train_classes = [0, 1, 8, 9]
            args.unlabeled_classes = [2, 3, 4, 5, 6, 7]
        elif args.setting == 'animal_0_transportation_2_0.5':
            args.train_classes = [0, 1]
            args.unlabeled_classes = [2, 3, 4, 5, 6, 7, 8, 9]
        elif args.setting == 'animal_3_transportation_2_0.5':
            # same as default
            args.train_classes = [0, 1, 2, 3, 4]
            args.unlabeled_classes = [5, 6, 7, 8, 9]
        elif args.setting == 'animal_1_transportation_1_0.5':
            args.train_classes = [0, 2]
            args.unlabeled_classes = [1, 3, 4, 5, 6, 7, 8, 9]
        elif args.setting == 'animal_2_transportation_0_0.5':
            args.train_classes = [2, 3]
            args.unlabeled_classes = [0, 1, 4, 5, 6, 7, 8, 9]
        else:
            raise NotImplementedError

    elif args.dataset_name == 'cifar100' or args.dataset_name == 'cifar100small':

        args.image_size = 32
        cifar100_coarse_labels_dict = get_cifar100_coarse_labels_dict()
        if args.setting == 'default':
            args.train_classes = range(80)
            args.unlabeled_classes = range(80, 100)
        elif args.setting == 'completely_new':
            args.train_classes = list()
            args.unlabeled_classes = list()
            for i in range(16):
                args.train_classes.extend(cifar100_coarse_labels_dict[i])
            for i in range(16, 20):
                args.unlabeled_classes.extend(cifar100_coarse_labels_dict[i])
        elif args.setting == 'completely_old':
            args.train_classes = list()
            args.unlabeled_classes = list()
            for i in range(20):
                for j in range(4):
                    args.train_classes.append(cifar100_coarse_labels_dict[i][j])
                args.unlabeled_classes.append(cifar100_coarse_labels_dict[i][4])
        elif args.setting == '50OLD50NEW':
            args.train_classes = range(50)
            args.unlabeled_classes = range(50, 100)
        else:
            raise NotImplementedError

    elif args.dataset_name == 'herbarium_19':

        args.image_size = 224
        herb_path_splits = os.path.join(osr_split_dir, 'herbarium_19_class_splits.pkl')

        with open(herb_path_splits, 'rb') as handle:
            class_splits = pickle.load(handle)

        args.train_classes = class_splits['Old']
        args.unlabeled_classes = class_splits['New']

    elif args.dataset_name == 'imagenet':

        args.image_size = 224
        if args.setting == 'default':
            args.train_classes = range(50)
            args.unlabeled_classes = range(50, 100)
        elif args.setting == '30new':
            args.train_classes = range(30)
            args.unlabeled_classes = range(30, 100)
        elif args.setting == '10new':
            args.train_classes = range(10)
            args.unlabeled_classes = range(10, 100)
        elif args.setting == '70new':
            args.train_classes = range(70)
            args.unlabeled_classes = range(70, 100)
        elif args.setting == '90new':
            args.train_classes = range(90)
            args.unlabeled_classes = range(90, 100)
        else:
            raise NotImplementedError

    elif args.dataset_name == 'imagenet_200':

        args.image_size = 224
        args.train_classes = range(100)
        args.unlabeled_classes = range(100, 200)

    elif args.dataset_name == 'imagenet_10':

        args.image_size = 224
        if args.setting == 'default':
            args.train_classes = [0, 1, 4, 5, 7]
            args.unlabeled_classes = [2, 3, 6, 8, 9]

    elif args.dataset_name == 'imagenet_100':

        args.image_size = 224
        if args.setting == 'default':
            args.train_classes = range(50)
            args.unlabeled_classes = range(50, 100)
        elif args.setting == '30new':
            args.train_classes = range(30)
            args.unlabeled_classes = range(30, 100)
        elif args.setting == '10new':
            args.train_classes = range(10)
            args.unlabeled_classes = range(10, 100)
        elif args.setting == '70new':
            args.train_classes = range(70)
            args.unlabeled_classes = range(70, 100)
        elif args.setting == '90new':
            args.train_classes = range(90)
            args.unlabeled_classes = range(90, 100)
        else:
            raise NotImplementedError

    elif args.dataset_name == 'imagenet_1k':

        args.image_size = 224
        args.train_classes = range(500)
        args.unlabeled_classes = range(500, 1000)
    
    elif args.dataset_name == 'scars':

        args.image_size = 224

        if use_ssb_splits:

            split_path = os.path.join(osr_split_dir, 'scars_osr_splits.pkl')
            with open(split_path, 'rb') as handle:
                class_info = pickle.load(handle)

            args.train_classes = class_info['known_classes']
            open_set_classes = class_info['unknown_classes']
            args.unlabeled_classes = open_set_classes['Hard'] + open_set_classes['Medium'] + open_set_classes['Easy']

        else:

            args.train_classes = range(98)
            args.unlabeled_classes = range(98, 196)

    elif args.dataset_name == 'aircraft':

        args.image_size = 224
        if use_ssb_splits:

            split_path = os.path.join(osr_split_dir, 'aircraft_osr_splits.pkl')
            with open(split_path, 'rb') as handle:
                class_info = pickle.load(handle)

            args.train_classes = class_info['known_classes']
            open_set_classes = class_info['unknown_classes']
            args.unlabeled_classes = open_set_classes['Hard'] + open_set_classes['Medium'] + open_set_classes['Easy']

        else:

            args.train_classes = range(50)
            args.unlabeled_classes = range(50, 100)

    elif args.dataset_name == 'cub':

        args.image_size = 224

        if use_ssb_splits:

            split_path = os.path.join(osr_split_dir, 'cub_osr_splits.pkl')
            with open(split_path, 'rb') as handle:
                class_info = pickle.load(handle)

            args.train_classes = class_info['known_classes']
            open_set_classes = class_info['unknown_classes']
            args.unlabeled_classes = open_set_classes['Hard'] + open_set_classes['Medium'] + open_set_classes['Easy']

        else:

            args.train_classes = range(100)
            args.unlabeled_classes = range(100, 200)

    else:

        raise NotImplementedError

    return args
