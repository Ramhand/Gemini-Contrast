import torch
import torch.nn.functional as F

def contrastive_loss(z_i, z_j, temperature=0.5):
    """
    Compute the contrastive loss for SimCLR.
    """
    batch_size = z_i.size(0)
    z = torch.cat([z_i, z_j], dim=0)

    sim = torch.matmul(z, z.T)
    sim = sim / temperature

    labels = torch.arange(batch_size).to(z_i.device)
    labels = torch.cat([labels, labels], dim=0)

    mask = torch.eye(2 * batch_size, device=z_i.device).bool()
    sim = sim.masked_fill(mask, -1e9)

    loss = F.cross_entropy(sim, labels)
    return loss
