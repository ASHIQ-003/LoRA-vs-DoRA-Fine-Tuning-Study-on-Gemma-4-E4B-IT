import torch
import random
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

def set_seed(seed: int = 42):
    """
    Sets the seed for reproducibility across Python, NumPy, and PyTorch.
    """
    logger.info(f"Setting global seed to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
