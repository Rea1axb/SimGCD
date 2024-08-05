# -----------------
# DATASET ROOTS
# -----------------
cifar_10_root = '../../../data/CIFAR10'
cifar_100_root = '../../../data/CIFAR100'
cub_root = '../../../data/CUB'
aircraft_root = '../../../data/fgvc-aircraft-2013b'
car_root = '../../../data/stanford_cars'
herbarium_dataroot = '../../../data/herbarium_19'
imagenet_root = '../../../data/imagenet100_small'
imagenet_200_root = '../../../data/imagenet200_small'

# OSR Split dir
osr_split_dir = 'data/ssb_splits'

# -----------------
# OTHER PATHS
# -----------------
exp_root = 'outputs' # All logs and checkpoints will be saved here
feature_extract_dir = './extracted_features'
dino_pretrain_path = './pretrained_models/dino_vitbase16_pretrain.pth'
resnet_pretrain_path = './pretrained_models/simclr_cifar_100.pth.tar'
# resnet_pretrain_path = './pretrained_models/simclr_imagenet_100.pth.tar'