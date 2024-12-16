import numpy as np
from torch.utils.data import Dataset

def subsample_instances(dataset, prop_indices_to_subsample=0.8):

    np.random.seed(0)
    subsample_indices = np.random.choice(range(len(dataset)), replace=False,
                                         size=(int(prop_indices_to_subsample * len(dataset)),))

    return subsample_indices

def get_cifar100_coarse_labels(fine_labels):
    coarse_labels = np.array([ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,
                                    3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                                    6, 11,  5, 10,  7,  6, 13, 15,  3, 15, 
                                    0, 11,  1, 10, 12, 14, 16,  9, 11,  5,
                                    5, 19,  8,  8, 15, 13, 14, 17, 18, 10,
                                    16, 4, 17,  4,  2,  0, 17,  4, 18, 17,
                                    10, 3,  2, 12, 12, 16, 12,  1,  9, 19, 
                                    2, 10,  0,  1, 16, 12,  9, 13, 15, 13,
                                    16, 19,  2,  4,  6, 19,  5,  5,  8, 19,
                                    18,  1,  2, 15,  6,  0, 17,  8, 14, 13])
    return coarse_labels[fine_labels]

def get_cifar100_coarse_labels_dict():
    coarse_labels_list = [ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,
                                    3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                                    6, 11,  5, 10,  7,  6, 13, 15,  3, 15, 
                                    0, 11,  1, 10, 12, 14, 16,  9, 11,  5,
                                    5, 19,  8,  8, 15, 13, 14, 17, 18, 10,
                                    16, 4, 17,  4,  2,  0, 17,  4, 18, 17,
                                    10, 3,  2, 12, 12, 16, 12,  1,  9, 19, 
                                    2, 10,  0,  1, 16, 12,  9, 13, 15, 13,
                                    16, 19,  2,  4,  6, 19,  5,  5,  8, 19,
                                    18,  1,  2, 15,  6,  0, 17,  8, 14, 13]
    coarse_labels_dict = {i: [] for i in range(20)}
    for fine_label, coarse_label in enumerate(coarse_labels_list):
        coarse_labels_dict[coarse_label].append(fine_label)
        
    for k, v in coarse_labels_dict.items():
        assert len(v) == 5, 'coarse_labels_dict error!'
    
    return coarse_labels_dict

def get_imagenet_coarse_labels(fine_labels):
    get_coarse_label = np.load('../../../data/imagenet100_small/get_coarse_labels.npy')
    return get_coarse_label[fine_labels]

def get_imagenet200_coarse_labels(fine_labels):
    get_coarse_label = np.load('../../../data/imagenet200_small/get_coarse_labels.npy')
    return get_coarse_label[fine_labels]

def get_imagenet10_coarse_labels(fine_labels):
    get_coarse_label = np.load('../../../data/imagenet10_small/get_coarse_labels.npy')
    return get_coarse_label[fine_labels]

class MergedDataset(Dataset):

    """
    Takes two datasets (labelled_dataset, unlabelled_dataset) and merges them
    Allows you to iterate over them in parallel
    """

    def __init__(self, labelled_dataset, unlabelled_dataset, use_coarse_label=False):

        self.labelled_dataset = labelled_dataset
        self.unlabelled_dataset = unlabelled_dataset
        self.use_coarse_label = use_coarse_label
        self.target_transform = None

    def __getitem__(self, item):

        if item < len(self.labelled_dataset):
            data = self.labelled_dataset[item]
            # img, label, uq_idx = self.labelled_dataset[item]
            labeled_or_not = 1

        else:
            data = self.unlabelled_dataset[item - len(self.labelled_dataset)]
            # img, label, uq_idx = self.unlabelled_dataset[item - len(self.labelled_dataset)]
            labeled_or_not = 0

        if self.use_coarse_label:
            img, label, coarse_label, uq_idx = data
            return img, label, coarse_label, uq_idx, np.array([labeled_or_not])
        else:
            img, label, uq_idx = data
            return img, label, uq_idx, np.array([labeled_or_not])

        return img, label, uq_idx, np.array([labeled_or_not])

    def __len__(self):
        return len(self.unlabelled_dataset) + len(self.labelled_dataset)
