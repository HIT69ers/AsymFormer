import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.myformer import get_segformer_backbone
from src.models import *


class Builder(nn.Module):
    def __init__(self, encoder, rgb_branch, d_branch, decoder):
        self.encoder = encoder

        if self.encoder == "biformer":
            assert rgb_branch, d_branch in ['b0', 'b1', 'b2', 'b3']