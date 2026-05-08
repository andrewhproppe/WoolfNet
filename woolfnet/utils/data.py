import numpy as np
from jax.tree_util import tree_map
from torch.utils.data import default_collate


def numpy_collate(batch):
    """
    Collate function specifies how to combine a list of data samples into a batch.
    default_collate creates pytorch tensors, then tree_map converts them into numpy
    arrays.
    """
    return tree_map(np.asarray, default_collate(batch))
