import torch

class MemoryQueue:
    """ MoCo-style memory bank
    """
    def __init__(self, max_size=1024, fine_label_num=100, coarse_label_num=20, name='labeled'):
        self.max_size = max_size
        self.fine_label_num = fine_label_num
        self.coarse_label_num = coarse_label_num
        self.name = name
        self.pointer = 0
        self.fine_labels = torch.zeros(self.max_size, dtype=torch.long)-1 ### -1 denotes invalid
        self.coarse_labels = torch.zeros(self.max_size, dtype=torch.long)-1
        self.coarse_logits = torch.zeros(self.max_size, self.coarse_label_num)
        return
    
    def add(self, fine_logits, coarse_logits):
        fine_logits = fine_logits.detach().cpu()
        coarse_logits = coarse_logits.detach().cpu()
        _, fine_label = fine_logits.max(1)
        _, coarse_label = coarse_logits.max(1)

        N = fine_label.size(0)
        
        idx = (torch.arange(N) + self.pointer).fmod(self.max_size).long()
        self.fine_labels[idx] = fine_label
        self.coarse_labels[idx] = coarse_label
        self.coarse_logits[idx] = coarse_logits
        self.pointer = idx[-1] + 1
        return
    
    @torch.no_grad()
    def hard_query(self, device='cuda'):
        """
        # Return
            fine2coarse_hard_label, shape: (fine_label_num, )
        """
        if len(self) == 0:
            return None, None
        idx = (self.fine_labels != -1)
        # Combine label and coarse_label into a single index, then use torch.bincount to count occurrences
        combined_index = self.fine_labels[idx] * self.coarse_label_num + self.coarse_labels[idx]
        count_vector = torch.bincount(combined_index, minlength=self.fine_label_num * self.coarse_label_num)

        # Reshape count_vector to a 2D tensor
        count_matrix = count_vector.reshape(self.fine_label_num, self.coarse_label_num)
        _, fine2coarse_hard_label = count_matrix.max(1)
        fine2coarse_hard_label = fine2coarse_hard_label.to(device)

        return fine2coarse_hard_label
    
    @torch.no_grad()
    def soft_query(self, device='cuda'):
        """
        # Return
            fine2coarse_soft_label, shape: (fine_label_num, coarse_label_num)
        """
        if len(self) == 0:
            return None, None
        idx = (self.fine_labels != -1)
        fine_labels_one_hot = torch.eye(self.fine_label_num)[self.fine_labels[idx]]
        instance_num = torch.sum(fine_labels_one_hot, dim=0, keepdim=True)
        instance_num[instance_num == 0] = 1
        fine2coarse_soft_label = torch.matmul(fine_labels_one_hot.T, self.coarse_logits[idx]) / instance_num.T
        fine2coarse_soft_label = fine2coarse_soft_label.to(device)
        return fine2coarse_soft_label
    
    def __len__(self):
        return (self.fine_labels != -1).sum().item()
    

if __name__=='__main__':
    membank = MemoryQueue(max_size=150, fine_label_num=10, coarse_label_num=3)
    fine_logits = torch.randn((100, 10))
    coarse_logits = torch.randn((100, 3))
    membank.add(fine_logits, coarse_logits)
    fine_logits = torch.randn((100, 10))
    coarse_logits = torch.randn((100, 3))
    membank.add(fine_logits, coarse_logits)
    soft_out = membank.soft_query(device='cpu')
    hard_out = membank.hard_query(device='cpu')
    