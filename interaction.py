import torch
import numpy as np
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
import argparse
import math

class cross_modal_interaction(nn.Module):
    def __init__(self, opt):

        super(cross_modal_interaction, self).__init__()
        self.embed_size = opt.embed_size

        # K
        self.image_K = nn.Linear(opt.embed_size, opt.embed_size)
        self.text_K = nn.Linear(opt.embed_size, opt.embed_size)

        # Q
        self.image_Q = nn.Linear(opt.embed_size, opt.embed_size)
        self.text_Q = nn.Linear(opt.embed_size, opt.embed_size)

        # V
        self.image_V = nn.Linear(opt.embed_size, opt.embed_size)
        self.text_V = nn.Linear(opt.embed_size, opt.embed_size)

        #
        self.image = nn.Linear((opt.embed_size + opt.embed_size), opt.embed_size)
        self.text = nn.Linear((opt.embed_size + opt.embed_size), opt.embed_size)

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        # image init
        image_r = np.sqrt(6.) / np.sqrt(self.image_K.in_features + self.image_K.out_features)
        self.image_K.weight.data.uniform_(-image_r, image_r)
        self.image_Q.weight.data.uniform_(-image_r, image_r)
        self.image_V.weight.data.uniform_(-image_r, image_r)
        self.image_K.bias.data.fill_(0)
        self.image_Q.bias.data.fill_(0)
        self.image_V.bias.data.fill_(0)
        image_r1 = np.sqrt(6.) / np.sqrt(self.image.in_features + self.image.out_features)
        self.image.weight.data.uniform_(-image_r1, image_r1)
        self.image.bias.data.fill_(0)

        # text init
        text_r = np.sqrt(6.) / np.sqrt(self.text_K.in_features + self.text_K.out_features)
        self.text_K.weight.data.uniform_(-text_r, text_r)
        self.text_Q.weight.data.uniform_(-text_r, text_r)
        self.text_V.weight.data.uniform_(-text_r, text_r)
        self.text_K.bias.data.fill_(0)
        self.text_Q.bias.data.fill_(0)
        self.text_V.bias.data.fill_(0)

        text_r1 = np.sqrt(6.) / np.sqrt(self.text.in_features + self.text.out_features)
        self.text.weight.data.uniform_(-text_r1, text_r1)
        self.text.bias.data.fill_(0)


    def fea_update(self, images, texts):

        # The matrix relates to images
        image_K = self.image_K(images)
        image_Q = self.image_Q(images)
        image_V = self.image_V(images)

        # The matrix relates to texts
        text_K = self.text_K(texts)
        text_Q = self.text_Q(texts)
        text_V = self.text_V(texts)

        # text to image
        att_T2V = torch.bmm(image_Q, text_K.transpose(1, 2))
        att_T2V = F.softmax(torch.div(att_T2V, math.sqrt(self.embed_size)), dim=-1)
        image_feature = torch.bmm(att_T2V, text_V)

        # image to text
        att_V2T = torch.bmm(text_Q, image_K.transpose(1, 2))
        att_V2T = F.softmax(torch.div(att_V2T, math.sqrt(self.embed_size)), dim=-1)
        text_feature = torch.bmm(att_V2T, image_V)

        # update feature
        image_feature = torch.cat((images, image_feature), dim=-1)
        image_feature = self.image(image_feature)

        text_feature = torch.cat((texts, text_feature), dim=-1)
        text_feature = self.text(text_feature)


        return image_feature, text_feature

    def forward(self, images, caps):

        ima_fea, caps_fea = self.fea_update(images, caps)

        return ima_fea, caps_fea

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(cross_modal_interaction, self).load_state_dict(new_state)
