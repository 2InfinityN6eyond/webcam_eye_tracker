import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader

from basic_net import EyeTrackerNet as BasicNet


class ModelFactory :    
    def __init__(self, model_name) :
        pass