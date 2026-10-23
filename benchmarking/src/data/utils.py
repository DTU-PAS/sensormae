import random

import numpy as np
import torch


def detection_x_collate_fn(batch):
    batch = [b for b in batch if b is not None]
    inputs_list, targets = zip(*batch)
    modalities = inputs_list[0].keys()
    return {mod: torch.stack([x[mod] for x in inputs_list]) for mod in modalities}, targets


def _worker_init_fn(worker_id):
    """Re-seed numpy / random and the albumentations pipeline of every worker."""
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)
    info = torch.utils.data.get_worker_info()
    if info is not None:
        ds = info.dataset
        if hasattr(ds, "geometric_transform") and ds.geometric_transform is not None:
            gt = ds.geometric_transform
            if hasattr(gt, "transform") and hasattr(gt.transform, "set_random_seed"):
                gt.transform.set_random_seed(seed)
