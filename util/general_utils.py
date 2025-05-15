import os
import torch
import inspect

from datetime import datetime
from loguru import logger

from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import torch

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):

        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):

        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def init_experiment(args, runner_name=None, exp_id=None):
    # Get filepath of calling script
    if runner_name is None:
        runner_name = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))).split(".")[-2:]

    root_dir = os.path.join(args.exp_root, *runner_name)

    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    # Either generate a unique experiment ID, or use one which is passed
    if exp_id is None:

        if args.exp_name is None:
            raise ValueError("Need to specify the experiment name")
        # Unique identifier for experiment
        now = '{}_({:02d}.{:02d}.{}_|_'.format(args.exp_name, datetime.now().day, datetime.now().month, datetime.now().year) + \
              datetime.now().strftime("%S.%f")[:-3] + ')'

        log_dir = os.path.join(root_dir, 'log', now)
        while os.path.exists(log_dir):
            now = '({:02d}.{:02d}.{}_|_'.format(datetime.now().day, datetime.now().month, datetime.now().year) + \
                  datetime.now().strftime("%S.%f")[:-3] + ')'

            log_dir = os.path.join(root_dir, 'log', now)

    else:

        log_dir = os.path.join(root_dir, 'log', f'{exp_id}')

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
        
    logger.add(os.path.join(log_dir, 'log.txt'))
    args.logger = logger
    args.log_dir = log_dir

    # Instantiate directory to save models to
    model_root_dir = os.path.join(args.log_dir, 'checkpoints')
    if not os.path.exists(model_root_dir):
        os.mkdir(model_root_dir)

    args.model_dir = model_root_dir
    args.model_path = os.path.join(args.model_dir, 'model.pt')

    print(f'Experiment saved to: {args.log_dir}')

    args.writer = SummaryWriter(log_dir=args.log_dir)

    hparam_dict = {}

    for k, v in vars(args).items():
        if isinstance(v, (int, float, str, bool, torch.Tensor)):
            hparam_dict[k] = v

    args.writer.add_hparams(hparam_dict=hparam_dict, metric_dict={})
    
    print(runner_name)
    print(args)

    return args

def label_based_sampling(labels, sample_num):
    """
    Performs sampling on labeled data, ensuring each class has `sample_num` samples.
    If a class has fewer than `sample_num` samples, all samples from that class are selected.

    Args:
        labels (torch.Tensor): 1D tensor of shape (N,) containing class labels for each sample.
        sample_num (int): Number of samples to select per class.

    Returns:
        list: List of sampled indices.
    """
    #TODO: debug
    label_to_indices = dict()
    for idx, label in enumerate(labels):
        if label.item() not in label_to_indices:
            label_to_indices[label.item()] = []
        label_to_indices[label.item()].append(idx)

    # 按类别进行采样
    sampled_indices = []
    for label, indices in label_to_indices.items():
        if len(indices) <= sample_num:
            sampled_indices.extend(indices)  # 全部采样
        else:
            sampled_indices.extend([indices[i] for i in torch.randperm(len(indices))[:sample_num].tolist()])  # 随机采样

    return sampled_indices

def three_stage_sampling(similarity, num_samples_1, num_samples_2, num_samples_3, eps=1e-6):
    """
    Perform three-stage sampling based on similarity scores.

    Args:
        similarity (torch.Tensor): Tensor of similarity scores.
        num_samples_1 (int): Number of samples to be selected in the first stage.
        num_samples_2 (int): Number of samples to be selected in the second stage.
        num_samples_3 (int): Number of samples to be selected in the third stage.
        eps (float, optional): Small value added to avoid division by zero. Defaults to 1e-6.

    Returns:
        tuple: A tuple containing two elements:
            - sampled_indices (dict): A dictionary containing the sampled indices for each stage.
            - nearest_two_prototypes (list): A list of indices of the nearest two prototypes for each sample in the second stage.

    """
    sampled_indices = {}
    sorted_similarity, sorted_idxs = torch.sort(similarity, dim=1)  # Sort similarity
    sorted_similarity = (sorted_similarity + 1.0) / 2.0  # Normalize similarity to [0, 1]

    # First stage sampling
    weights_1 = sorted_similarity[:, 0]  # Similarity to the nearest prototype
    weights_1 /= torch.sum(weights_1)  # Normalize weights
    sampled_indices['stage_1'] = torch.multinomial(weights_1, num_samples_1, replacement=False).tolist()

    # Second stage sampling
    diff_distances = torch.abs(sorted_similarity[:, 0] - sorted_similarity[:, 1])  # Difference between similarity to the nearest and second nearest prototypes
    weights_2 = 1 / (diff_distances + eps)  # Inverse of difference similarity
    weights_2 /= torch.sum(weights_2)  # Normalize weights
    sampled_indices['stage_2'] = torch.multinomial(weights_2, num_samples_2, replacement=False).tolist()

    nearest_two_prototypes = []  # List of indices of the nearest two prototypes
    for idx in sampled_indices['stage_2']:
        nearest_two_prototypes.append(sorted_idxs[idx, :2].tolist())  # Get indices of the nearest two prototypes

    # Third stage sampling
    weights_3 = 1 / (sorted_similarity[:, 0] + eps)  # Inverse of similarity
    weights_3 /= torch.sum(weights_3)  # Normalize weights
    sampled_indices['stage_3'] = torch.multinomial(weights_3, num_samples_3, replacement=False).tolist()
    
    return sampled_indices, nearest_two_prototypes
    

def get_mean_lr(optimizer):
    return torch.mean(torch.Tensor([param_group['lr'] for param_group in optimizer.param_groups])).item()

class DistributedWeightedSampler(torch.utils.data.distributed.DistributedSampler):

    def __init__(self, dataset, weights, num_samples, num_replicas=None, rank=None,
                 replacement=True, generator=None):
        super(DistributedWeightedSampler, self).__init__(dataset, num_replicas, rank)
        if not isinstance(num_samples, int) or isinstance(num_samples, bool) or \
                num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(num_samples))
        if not isinstance(replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(replacement))
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_samples = num_samples
        self.replacement = replacement
        self.generator = generator
        self.weights = self.weights[self.rank::self.num_replicas]
        self.num_samples = self.num_samples // self.num_replicas

    def __iter__(self):
        rand_tensor = torch.multinomial(self.weights, self.num_samples, self.replacement, generator=self.generator)
        rand_tensor =  self.rank + rand_tensor * self.num_replicas
        yield from iter(rand_tensor.tolist())

    def __len__(self):
        return self.num_samples
