import torch
import torch.nn.functional as F

def supervised_contrastive_loss(features, labels, temperature=0.05):
    """
    Compute the supervised contrastive loss as per Khosla et al. (2020).
    """
    device = features.device
    labels = labels.unsqueeze(1)
    mask = torch.eq(labels, labels.T).float().to(device)

    dot_product = torch.matmul(features, features.T) / temperature
    exp_dot_product = torch.exp(dot_product)

    mask_self = torch.eye(labels.shape[0], device=device).bool()
    exp_dot_product = exp_dot_product.masked_fill(mask_self, 0)

    denom = exp_dot_product.sum(dim=1, keepdim=True) + 1e-8
    log_prob = dot_product - torch.log(denom)
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
    loss = -mean_log_prob_pos.mean()

    return loss