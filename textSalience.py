import torch
import torch.nn as nn
import torch.nn.init
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.weight_norm import weight_norm
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm
from interaction import cross_modal_interaction
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict
import math
import os

# return [BN, salience_embed_size]
class EncoderText_salience(nn.Module):
    def __init__(self, embed_size, Channel, kernel_num, kernel_size, salience_embed_size, p,):
        super(EncoderText_salience, self).__init__()
        # caption Text CNN para
        self.embed_size = embed_size
        self.Channel = Channel
        self.p = p
        # Numbers of conv kernel----int
        self.kernel_num = kernel_num
        # Size of conv kernel----list
        self.kernel_size = kernel_size
        # Saliency embedding dim
        self.salience_embed_size = salience_embed_size

        self.conv2d = nn.ModuleList([nn.Conv2d(self.Channel, self.kernel_num, (K, self.embed_size)) for K in self.kernel_size])
        self.dropout = nn.Dropout(self.p)
        self.fc = nn.Linear(len(self.kernel_size) * self.kernel_num, self.salience_embed_size)

    def forward(self, x):

        # CNN_operator
        # -----------------------------------------------------------------------#
        cap_emb = x.unsqueeze(1)  # (BN, 1, W, D)
        cap_emb = [F.relu(conv(cap_emb)).squeeze(3) for conv in self.conv2d]
        cap_emb = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in cap_emb]
        cap_emb = torch.cat(cap_emb, 1)
        cap_emb = self.dropout(cap_emb)
        cap_emb = self.fc(cap_emb)
        # -----------------------------------------------------------------------#
        #(BN, saliency_embed_size)

        return cap_emb


x = torch.rand(10, 11, 20)
model = EncoderText_salience(20, 1, 3, [3, 5, 9], 30, 0.01)
results = model(x)
print(x.shape)