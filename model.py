import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, 
                 nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        elif nlayers != 0:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(in_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x_proj = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        # x = x.detach()
        logits = self.last_layer(x)
        return x_proj, logits

class MultiheadAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, attn_drop=0., proj_drop=0.):
        super(MultiheadAttention, self).__init__()
        self.hid_dim = hid_dim
        self.n_heads = n_heads

        assert hid_dim % n_heads == 0

        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)
        self.attn_drop = nn.Dropout(attn_drop)

        self.fc = nn.Linear(hid_dim, hid_dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.scale = (hid_dim // n_heads) ** -0.5


    def forward(self, query, key, value, mask=None):
        # query: (batch_size, q_cnt, hid_dim)
        # key: (batch_size, k_cnt, hid_dim)
        # value: (batch_size, v_cnt, hid_dim), k_cnt == v_cnt
        bsz = query.shape[0]
        Q = self.w_q(query) # (batch_size, q_cnt, hid_dim)
        K = self.w_k(key) # (batch_size, k_cnt, hid_dim)
        V = self.w_v(value) # (batch_size, v_cnt, hid_dim)
        # Q = query
        # K = key
        # V = value

        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3) # (batch_size, n_head, q_cnt, head_dim)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3) # (batch_size, n_head, k_cnt, head_dim)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3) # (batch_size, n_head, v_cnt, head_dim)

        attention = torch.matmul(Q, K.permute(0, 1, 3, 2)) * self.scale # (batch_size, n_head, q_cnt, k_cnt)

        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e10)

        attention = self.attn_drop(torch.softmax(attention, dim=-1))

        x = torch.matmul(attention, V) # (batch_size, n_head, q_cnt, head_dim)

        x = x.permute(0, 2, 1, 3).contiguous() # (batch_size, q_cnt, n_head, head_dim)

        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads)) # (batch_size, q_cnt, hid_dim)
        x = self.fc(x)
        x = self.proj_drop(x)
        return x, attention


class CoarseFromFineHead(nn.Module):
    def __init__(self, in_dim, out_dim_fine, out_dim_coarse, use_bn=False, norm_last_layer=True,
                 mlp_nlayers=3, hidden_dim=2048, bottleneck_dim=256, attn_head=1):
        super(CoarseFromFineHead, self).__init__()
        nlayers = max(mlp_nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        elif nlayers != 0:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.predictor = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, in_dim)
        )
        # self.attn = MultiheadAttention(hid_dim=in_dim, n_heads=attn_head)
        self.coarse_last_layer = nn.Linear(out_dim_fine, out_dim_coarse, bias=False)
        # self.coarse_query_layer = nn.Linear(in_dim, out_dim_coarse, bias=False)
        self.apply(self._init_weights)
        self.fine_last_layer = nn.utils.weight_norm(nn.Linear(in_dim, out_dim_fine, bias=False))
        self.fine_last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.fine_last_layer.weight_g.requires_grad = False
        
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x_proj = self.mlp(x)
        x_pred = self.predictor(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x_pred = nn.functional.normalize(x_pred, dim=-1, p=2)
        # x = x.detach()
        fine_logits = self.fine_last_layer(x)
        # coarse_prototypes = self.get_coarse_prototypes_with_attention(proj=False)
        coarse_prototypes = self.get_coarse_prototypes(proj=False)
        coarse_logits = torch.matmul(x, coarse_prototypes.T)
        return x, x_pred, x_proj, fine_logits, coarse_logits
    
    # def get_coarse_prototypes_with_attention(self, proj=True, return_att_weight=False):
    #     coarse_weight = self.coarse_query_layer.weight
    #     coarse_weight = coarse_weight.unsqueeze(0)
    #     fine_weight = self.fine_last_layer.weight
    #     fine_weight = fine_weight.unsqueeze(0)
    #     coarse_prototypes, att_weight = self.attn(coarse_weight, fine_weight, fine_weight)
    #     coarse_prototypes = coarse_prototypes.squeeze(0)
    #     att_weight = att_weight.squeeze(0)
    #     if proj:
    #         coarse_prototypes = self.mlp(coarse_prototypes)
    #     if return_att_weight:
    #         return coarse_prototypes, att_weight
    #     return coarse_prototypes
    
    # NOTE: in pytorch 1.12, The weight is recomputed once at module forward, so use the following functions after module forward
    def get_coarse_prototypes(self, proj=True):
        fine_weight = self.fine_last_layer.weight
        coarse_weight = self.coarse_last_layer(fine_weight.T).T
        if proj:
            coarse_weight = self.mlp(coarse_weight)
        coarse_weight = nn.functional.normalize(coarse_weight, dim=-1, p=2)
        return coarse_weight
    
    def get_fine_prototypes(self, proj=True):
        fine_weight = self.fine_last_layer.weight
        if proj:
            fine_weight = self.mlp(fine_weight)
        fine_weight = nn.functional.normalize(fine_weight, dim=-1, p=2)
        return fine_weight


class TwoHead(nn.Module):
    def __init__(self, in_dim, out_dim_fine, out_dim_coarse, use_bn=False, norm_last_layer=True,
                 mlp_nlayers=3, hidden_dim=2048, bottleneck_dim=256, attn_head=6):
        super(TwoHead, self).__init__()
        nlayers = max(mlp_nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        elif nlayers != 0:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.predictor = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, in_dim)
        )
        self.attn = MultiheadAttention(hid_dim=in_dim, n_heads=attn_head)
        self.apply(self._init_weights)
        self.fine_last_layer = nn.utils.weight_norm(nn.Linear(in_dim, out_dim_fine, bias=False))
        self.coarse_last_layer = nn.utils.weight_norm(nn.Linear(in_dim, out_dim_coarse, bias=False))
        self.fine_last_layer.weight_g.data.fill_(1)
        self.coarse_last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.fine_last_layer.weight_g.requires_grad = False
            self.coarse_last_layer.weight_g.requires_grad = False
        
        
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x_proj = self.mlp(x)
        x_pred = self.predictor(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x_pred = nn.functional.normalize(x_pred, dim=-1, p=2)
        # x = x.detach()
        fine_logits = self.fine_last_layer(x)
        coarse_logits = self.coarse_last_layer(x)
        return x, x_pred, x_proj, fine_logits, coarse_logits
    
    def get_coarse_prototypes_with_attention(self, proj=True, return_att_weight=False):
        coarse_weight = self.coarse_last_layer.weight
        coarse_weight = coarse_weight.unsqueeze(0)
        fine_weight = self.fine_last_layer.weight
        fine_weight = fine_weight.unsqueeze(0)
        coarse_prototypes, att_weight = self.attn(coarse_weight, fine_weight, fine_weight)
        coarse_prototypes = coarse_prototypes.squeeze(0)
        att_weight = att_weight.squeeze(0)
        if proj:
            coarse_prototypes = self.mlp(coarse_prototypes)
        if return_att_weight:
            return coarse_prototypes, att_weight
        return coarse_prototypes
    
    # NOTE: in pytorch 1.12, The weight is recomputed once at module forward, so use the following two functions after module forward
    def get_coarse_prototypes(self, proj=True):
        coarse_weight = self.coarse_last_layer.weight
        if proj:
            coarse_weight = self.mlp(coarse_weight)
        return coarse_weight

    def get_fine_prototypes(self, proj=True):
        fine_weight = self.fine_last_layer.weight
        if proj:
            fine_weight = self.mlp(fine_weight)
        return fine_weight


class CoarseHead(nn.Module):
    def __init__(self, dino_in_dim, dino_out_dim, dino_nlayers, proto_num=20, latent_dim=768, tau=0.1, init=False):
        super(CoarseHead, self).__init__()

        self.proto_num = proto_num
        self.latent_dim = latent_dim
        self.tau = tau
        self.dino_projector = DINOHead(in_dim=dino_in_dim, out_dim=dino_out_dim, nlayers=dino_nlayers)

        self.prototypes = nn.Parameter(torch.randn(self.proto_num, self.latent_dim), requires_grad=True) 

        if init:
            self.prototypes.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
    
    def forward(self, x):
        c = self.prototypes
        c = F.normalize(c, dim=1)
        z = F.normalize(x, dim=1)
        coarse_logits = torch.mm(z, c.t())

        proj, logits = self.dino_projector(x)
        return proj, logits, coarse_logits



class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        if not isinstance(self.base_transform, list):
            return [self.base_transform(x) for i in range(self.n_views)]
        else:
            return [self.base_transform[i](x) for i in range(self.n_views)]

class CoarseSupConLoss(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    From: https://github.com/HobbitLong/SupContrast"""
    def __init__(self, temperature=0.07, 
                 base_temperature=0.07):
        super(CoarseSupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features_pred, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        if features_pred.shape != features.shape:
            raise ValueError('the shapes of `features_pred` and `features` need to be same')

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)
        if len(features_pred.shape) > 3:
            features_pred = features_pred.view(features_pred.shape[0], features_pred.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        anchor_count = contrast_count
        anchor_feature = torch.cat(torch.unbind(features_pred, dim=1), dim=0)
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute mean over positive
        mean_prob_pos = (mask * logits).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class SupConLoss(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    From: https://github.com/HobbitLong/SupContrast"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss



def info_nce_logits(features, n_views=2, temperature=1.0, device='cuda'):

    b_ = 0.5 * int(features.size(0))

    labels = torch.cat([torch.arange(b_) for i in range(n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    logits = logits / temperature
    return logits, labels

def coarse_info_nce_logits(features, prototypes, coarse_logits, temperature=1.0, device='cuda'):
    features = F.normalize(features, dim=1)
    prototypes = F.normalize(prototypes, dim=1)
    similarity_matrix = torch.matmul(features, prototypes.T)
    _, labels = torch.max(coarse_logits, dim=1)
    labels = labels.to(device)
    logits = similarity_matrix / temperature
    return logits, labels

def get_coarse_sup_logits_mean_labels(teacher_coarse_logits, fine_labels, fine_out_dim, device='cuda'):
    fine_labels = torch.eye(fine_out_dim)[fine_labels].to(device)
    fine_matrix = torch.matmul(fine_labels, fine_labels.T) # fine_matrix[i, j] == 1 means instance `i` and instance `j` have same label
    coarse_logits_labels = torch.matmul(fine_matrix, teacher_coarse_logits) / torch.sum(fine_matrix, dim=1, keepdim=True)
    return coarse_logits_labels

def get_coarse_sup_logits_random_labels(teacher_coarse_logits, fine_labels, fine_out_dim, device='cuda'):
    fine_labels = torch.eye(fine_out_dim)[fine_labels].to(device)
    fine_matrix = torch.matmul(fine_labels, fine_labels.T)
    map_matrix = torch.zeros_like(fine_matrix)
    indices_with_ones = (fine_matrix == 1).nonzero()
    for i in range(fine_matrix.size(0)):
        row_indices = indices_with_ones[indices_with_ones[:, 0] == i, 1]
        selected_index = torch.randint(0, row_indices.size(0), (1,))
        map_matrix[i, row_indices[selected_index]] = 1
    coarse_logits_labels = torch.matmul(map_matrix, teacher_coarse_logits)
    return coarse_logits_labels

def get_coarse_sup_logits_mq_labels(fine_labels, mq_labels, device='cuda'):
    coarse_logits_labels = mq_labels[fine_labels].to(device)
    return coarse_logits_labels
    
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


class DistillLoss(nn.Module):
    def __init__(self, warmup_teacher_temp_epochs, nepochs, 
                 ncrops=2, warmup_teacher_temp=0.07, teacher_temp=0.04,
                 student_temp=0.1):
        super().__init__()
        self.student_temp = student_temp
        self.ncrops = ncrops
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax(teacher_output / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        return total_loss

class TCALoss(nn.Module):
    def __init__(self, temperature=1.0) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, coarse_logits, fine_labels, coarse_prototypes, fine_prototypes):
        similarity_cf = torch.matmul(coarse_prototypes, fine_prototypes.T)
        fine_logits = torch.matmul(coarse_logits / self.temperature, similarity_cf)
        loss = nn.CrossEntropyLoss()(fine_logits, fine_labels)
        return loss
        
class PrototypesLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, prototypes, device='cuda'):
        similarity_matrix = torch.matmul(prototypes, prototypes.T)
        similarity_matrix = similarity_matrix + 1.0
        # similarity_max, _ = torch.max(similarity_matrix, dim=-1, keepdim=True)
        # similarity_matrix = similarity_matrix - similarity_max.detach()
        mask = torch.eye(prototypes.shape[0], dtype=torch.bool)
        mask = (~mask).float().to(device)
        loss = torch.sum(similarity_matrix * mask) / mask.sum()
        return loss


if __name__ == "__main__":
    in_dim = 768
    out_dim_fine = 100
    out_dim_coarse = 20

    model = TwoHead(in_dim=in_dim, out_dim_fine=out_dim_fine, out_dim_coarse=out_dim_coarse, use_bn=False)
    for i in range(3):
        input = torch.randn((32, 768))
        # 生成随机整数标签
        random_labels = torch.randint(0, 100, size=(32,))
        # 使用 torch.eye() 和索引操作转换为 one-hot 矩阵
        labels = torch.eye(100)[random_labels]
        x, x_pred, x_proj, fine_logits, coarse_logits = model(input)
        coarse_prototypes = model.get_coarse_prototypes(proj=False)
        fine_prototypes = model.get_fine_prototypes(proj=False)
        criterion = TCALoss()
        loss = criterion(coarse_logits, labels, coarse_prototypes, fine_prototypes)
        opt = torch.optim.Adam(model.parameters(), lr=0.1)
        opt.zero_grad()
        loss.backward()
        opt.step()
        coarse_prototypes = model.get_coarse_prototypes(proj=False)
        fine_prototypes = model.get_fine_prototypes(proj=False)
        print(loss)
