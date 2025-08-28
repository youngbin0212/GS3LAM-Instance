import os

import numpy as np
import random
import torch
import torch.nn.functional as F


def seed_everything(seed=42):
    """
        Set the `seed` value for torch and numpy seeds. Also turns on
        deterministic execution for cudnn.
        
        Parameters:
        - seed:     A hashable seed value
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed set to: {seed} (type: {type(seed)})")


# New code for contrastive loss
def contrastive_loss(f1, f2, label, margin=1.0):
    """
    f1, f2: (B, D) shape embeddings
    label: (B,) 1 for same instance, 0 for different
    """
    dist = torch.norm(f1 - f2, dim=1)
    pos_loss = label * dist.pow(2)
    neg_loss = (1 - label) * F.relu(margin - dist).pow(2)
    return (pos_loss + neg_loss).mean()
