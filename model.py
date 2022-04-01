# -----------------------------------------------------------
# Multi-Perspective Network implementation based on
# "Multi-Perspective Motivated Neural Network for Image-Text Matching", and part of the code refer to SCAN
# Xueyang Qin, Lishuang Li, Guangyao Pang
# Writen by Xueyang Qin, 2020
# ---------------------------------------------------------------
"""MPMNN model"""

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
os.environ['CUDA_VISIBLE_DEVICES'] = '5,4'

def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def EncoderImage(data_name, img_dim, embed_size, precomp_enc_type='basic', 
                 no_imgnorm=False):
    """A wrapper to image encoders. Chooses between an different encoders
    that uses precomputed image features.
    """
    if precomp_enc_type == 'basic':
        img_enc = EncoderImagePrecomp(
            img_dim, embed_size, no_imgnorm)
    elif precomp_enc_type == 'weight_norm':
        img_enc = EncoderImageWeightNormPrecomp(
            img_dim, embed_size, no_imgnorm)
    else:
        raise ValueError("Unknown precomp_enc_type: {}".format(precomp_enc_type))

    return img_enc


class EncoderImagePrecomp(nn.Module):

    def __init__(self, img_dim, embed_size, no_imgnorm=False):
        super(EncoderImagePrecomp, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.fc = nn.Linear(img_dim, embed_size)

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized

        features = self.fc(images)

        # normalize in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features, dim=-1)

        return features

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImagePrecomp, self).load_state_dict(new_state)


class EncoderImageWeightNormPrecomp(nn.Module):

    def __init__(self, img_dim, embed_size, no_imgnorm=False):
        super(EncoderImageWeightNormPrecomp, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.fc = weight_norm(nn.Linear(img_dim, embed_size), dim=None)

    def forward(self, images):

        features = self.fc(images)
        # normalize in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features, dim=-1)

        return features

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImageWeightNormPrecomp, self).load_state_dict(new_state)


# RNN Based Language Model
class EncoderText(nn.Module):

    def __init__(self, vocab_size, word_dim, embed_size, num_layers,
                 use_bi_gru=True, no_txtnorm=False):
        super(EncoderText, self).__init__()
        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm

        # word embedding
        #self.e = np.loadtxt('f30k300.txt')
        self.embed = nn.Embedding(vocab_size, word_dim)
        #self.embed.weight.data.copy_(torch.from_numpy(self.e))

        # caption embedding
        self.use_bi_gru = use_bi_gru
        self.rnn = nn.GRU(word_dim, embed_size, num_layers, batch_first=True, bidirectional=use_bi_gru)

        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x, lengths):

        # Embed word ids to vectors
        x = self.embed(x)
        packed = pack_padded_sequence(x, lengths, batch_first=True)

        # Forward propagate RNN
        out, _ = self.rnn(packed)

        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)
        cap_emb, cap_len = padded

        if self.use_bi_gru:
            cap_emb = (cap_emb[:,:,:cap_emb.size(2)/2] + cap_emb[:,:,cap_emb.size(2)/2:])/2

        # normalization in the joint embedding space
        if not self.no_txtnorm:
            cap_emb = l2norm(cap_emb, dim=-1)
        return cap_emb, cap_len


def func_attention(query, context, opt, smooth, eps=1e-8):

    batch_size_q, queryL = query.size(0), query.size(1)
    batch_size, sourceL = context.size(0), context.size(1)

    # Get attention
    # --> (batch, d, queryL)
    queryT = torch.transpose(query, 1, 2)

    # (batch, sourceL, d)(batch, d, queryL)
    # --> (batch, sourceL, queryL)
    attn = torch.bmm(context, queryT)
    if opt.raw_feature_norm == "softmax":
        # --> (batch*sourceL, queryL)
        attn = attn.view(batch_size*sourceL, queryL)
        attn = nn.Softmax()(attn)
        # --> (batch, sourceL, queryL)
        attn = attn.view(batch_size, sourceL, queryL)
    elif opt.raw_feature_norm == "l2norm":
        attn = l2norm(attn, 2)
    elif opt.raw_feature_norm == "clipped_l2norm":
        attn = nn.LeakyReLU(0.1)(attn)
        attn = l2norm(attn, 2)
    elif opt.raw_feature_norm == "l1norm":
        attn = l1norm_d(attn, 2)
    elif opt.raw_feature_norm == "clipped_l1norm":
        attn = nn.LeakyReLU(0.1)(attn)
        attn = l1norm_d(attn, 2)
    elif opt.raw_feature_norm == "clipped":
        attn = nn.LeakyReLU(0.1)(attn)
    elif opt.raw_feature_norm == "no_norm":
        pass
    else:
        raise ValueError("unknown first norm type:", opt.raw_feature_norm)
    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous()
    # --> (batch*queryL, sourceL)
    attn = attn.view(batch_size*queryL, sourceL)
    attn = nn.Softmax()(attn*smooth)
    # --> (batch, queryL, sourceL)
    attn = attn.view(batch_size, queryL, sourceL)
    # --> (batch, sourceL, queryL)
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # --> (batch, d, sourceL)
    contextT = torch.transpose(context, 1, 2)
    # (batch x d x sourceL)(batch x sourceL x queryL)
    # --> (batch, d, queryL)
    weightedContext = torch.bmm(contextT, attnT)
    # --> (batch, queryL, d)
    weightedContext = torch.transpose(weightedContext, 1, 2)

    return weightedContext, attnT


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


def xattn_score_t2i(images, captions, cap_lens, opt):
    """
    Images: (n_image, n_regions, d) matrix of images
    Captions: (n_caption, max_n_word, d) matrix of captions
    CapLens: (n_caption) array of caption lengths
    """
    similarities = []
    n_image = images.size(0)
    n_caption = captions.size(0)
    for i in range(n_caption):
        # Get the i-th text description
        n_word = cap_lens[i]
        cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
        # --> (n_image, n_word, d)
        cap_i_expand = cap_i.repeat(n_image, 1, 1)
        """
            word(query): (n_image, n_word, d)
            image(context): (n_image, n_regions, d)
            weiContext: (n_image, n_word, d)
            attn: (n_image, n_region, n_word)
        """
        weiContext, attn = func_attention(cap_i_expand, images, opt, smooth=opt.lambda_softmax)
        cap_i_expand = cap_i_expand.contiguous()
        weiContext = weiContext.contiguous()
        # (n_image, n_word)
        row_sim = cosine_similarity(cap_i_expand, weiContext, dim=2)
        if opt.agg_func == 'LogSumExp':
            row_sim.mul_(opt.lambda_lse).exp_()
            row_sim = row_sim.sum(dim=1, keepdim=True)
            row_sim = torch.log(row_sim)/opt.lambda_lse
        elif opt.agg_func == 'Max':
            row_sim = row_sim.max(dim=1, keepdim=True)[0]
        elif opt.agg_func == 'Sum':
            row_sim = row_sim.sum(dim=1, keepdim=True)
        elif opt.agg_func == 'Mean':
            row_sim = row_sim.mean(dim=1, keepdim=True)
        else:
            raise ValueError("unknown aggfunc: {}".format(opt.agg_func))
        similarities.append(row_sim)

    # (n_image, n_caption)
    similarities = torch.cat(similarities, 1)
    
    return similarities


def xattn_score_i2t(images, captions, cap_lens, opt):
    """
    Images: (batch_size, n_regions, d) matrix of images
    Captions: (batch_size, max_n_words, d) matrix of captions
    CapLens: (batch_size) array of caption lengths
    """
    similarities = []
    n_image = images.size(0)
    n_caption = captions.size(0)
    n_region = images.size(1)
    for i in range(n_caption):
        # Get the i-th text description
        n_word = cap_lens[i]
        cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
        # (n_image, n_word, d)
        cap_i_expand = cap_i.repeat(n_image, 1, 1)
        """
            word(query): (n_image, n_word, d)
            image(context): (n_image, n_region, d)
            weiContext: (n_image, n_region, d)
            attn: (n_image, n_word, n_region)
        """
        weiContext, attn = func_attention(images, cap_i_expand, opt, smooth=opt.lambda_softmax)
        # (n_image, n_region)
        row_sim = cosine_similarity(images, weiContext, dim=2)
        if opt.agg_func == 'LogSumExp':
            row_sim.mul_(opt.lambda_lse).exp_()
            row_sim = row_sim.sum(dim=1, keepdim=True)
            row_sim = torch.log(row_sim)/opt.lambda_lse
        elif opt.agg_func == 'Max':
            row_sim = row_sim.max(dim=1, keepdim=True)[0]
        elif opt.agg_func == 'Sum':
            row_sim = row_sim.sum(dim=1, keepdim=True)
        elif opt.agg_func == 'Mean':
            row_sim = row_sim.mean(dim=1, keepdim=True)
        else:
            raise ValueError("unknown aggfunc: {}".format(opt.agg_func))
        similarities.append(row_sim)

    # (n_image, n_caption)
    similarities = torch.cat(similarities, 1)
    return similarities


class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """
    def __init__(self, opt, margin=0, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.opt = opt
        self.margin = margin
        self.max_violation = max_violation

    def forward(self, im, s, s_l):
        # compute image-sentence score matrix
        if self.opt.cross_attn == 't2i':
            scores = xattn_score_t2i(im, s, s_l, self.opt)
        elif self.opt.cross_attn == 'i2t':
            scores = xattn_score_i2t(im, s, s_l, self.opt)
        else:
            raise ValueError("unknown first norm type:", opt.raw_feature_norm)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]
        return cost_s.sum() + cost_im.sum()


def global_cos(im, cap, eps=1e-8):

    w12 = torch.mm(im, cap.t())
    w1 = torch.norm(im, 2, dim=-1).unsqueeze(1)
    w2 = torch.norm(cap, 2, dim=-1).unsqueeze(0)
    return w12 / (torch.mm(w1, w2)).clamp(min=eps)


def sim_i2t(images, captions, cap_lens, opt):

    return global_cos(images, captions)

def sim_t2i(images, captions, cap_lens, opt):

    return global_cos(captions, images)

class global_ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """
    def __init__(self, opt, margin=0, max_violation=False):
        super(global_ContrastiveLoss, self).__init__()
        self.opt = opt
        self.margin = margin
        self.max_violation = max_violation

    def forward(self, im, s, s_l):
        # compute image-sentence score matrix
        if self.opt.cross_attn == 't2i':
            scores = sim_t2i(im, s, s_l, self.opt)
        elif self.opt.cross_attn == 'i2t':
            scores = sim_i2t(im, s, s_l, self.opt)
        else:
            raise ValueError("unknown first norm type:", opt.raw_feature_norm)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]
        return cost_s.sum() + cost_im.sum()


class EncoderImage_global(nn.Module):

    def __init__(self,  embed_size, global_embed_size, no_imgnorm=False):
        super(EncoderImage_global, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.global_embed_size = global_embed_size
        self.fc = nn.Linear(embed_size, global_embed_size)

        self.fc1 = nn.Linear(36, 1)


    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized

        features = self.fc(images)

        features = torch.transpose(features, 1, 2)
        features = self.fc1(features)
        features = (torch.transpose(features, 1, 2)).squeeze(1)

        # normalize in the  global embedding space
        if not self.no_imgnorm:
            features = l2norm(features, dim=-1)

        return features

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImage_global, self).load_state_dict(new_state)


class Encodertext_global(nn.Module):

    def __init__(self,  embed_size, global_embed_size, no_txtnorm=False):
        super(Encodertext_global, self).__init__()
        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm
        self.global_embed_size = global_embed_size
        self.fc = nn.Linear(embed_size, global_embed_size)

        self.fc1 = nn.Linear(11, 1)


    def forward(self, captions, cap_lens):
        """Extract text feature vectors."""
        # assuming that the precomputed features are already l2-normalized
        features = self.fc(captions)

        features = torch.transpose(features, 1, 2)
        features = self.fc1(features)
        features = torch.transpose(features, 1, 2).squeeze(1)

        # normalize in the  global embedding space
        if not self.no_txtnorm:
            features = l2norm(features, dim=-1)

        return features

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(Encodertext_global, self).load_state_dict(new_state)




# encoding salience images feature
# return [BN, salience_embed_size]
class EncoderImage_salience(nn.Module):

    def __init__(self,  salience_img_dim, salience_embed_size, no_imgnorm=False):
        super(EncoderImage_salience, self).__init__()
        self.no_imgnorm = no_imgnorm
        self.fc = nn.Linear(salience_img_dim, salience_embed_size)
        #self.init_weights()

    def forward(self, images):
        """Extract image feature vectors."""

        features = self.fc(images)

        # normalize in the  global embedding space
        if not self.no_imgnorm:
            features = l2norm(features, dim=-1)

        # [batch_size, salience_embed_size]
        return features

    def init_weights(self):
        self.fc.weight.data.uniform_(-0.1, 0.1)


    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImage_salience, self).load_state_dict(new_state)




# Encoding and calc saliency fearure

def salience_cos(im, cap, eps=1e-8):

    w12 = torch.mm(im, cap.t())
    w1 = torch.norm(im, 2, dim=-1).unsqueeze(1)
    w2 = torch.norm(cap, 2, dim=-1).unsqueeze(0)
    return w12 / (torch.mm(w1, w2)).clamp(min=eps)



def ssim_i2t(images, captions, cap_lens, opt):

    return salience_cos(images, captions)


def ssim_t2i(images, captions, cap_lens, opt):

    return salience_cos(captions, images)


class salience_ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """
    def __init__(self, opt, margin=0, max_violation=False):
        super(salience_ContrastiveLoss, self).__init__()
        self.opt = opt
        self.margin = margin
        self.max_violation = max_violation

    def forward(self, im, s, s_l):
        # compute image-sentence score matrix
        if self.opt.cross_attn == 't2i':
            scores = ssim_t2i(im, s, s_l, self.opt)
        elif self.opt.cross_attn == 'i2t':
            scores = ssim_i2t(im, s, s_l, self.opt)
        else:
            raise ValueError("unknown first norm type:", opt.raw_feature_norm)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]
        return cost_s.sum() + cost_im.sum()




# return [BN, salience_embed_size]
class EncoderText_salience(nn.Module):
    def __init__(self, vocab_size, word_dim, embed_size, num_layers, Channel, kernel_num, kernel_size, salience_embed_size, p,
                 use_bi_gru=False, no_txtnorm=False):
        super(EncoderText_salience, self).__init__()
        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm
        # word embedding
        self.embed = nn.Embedding(vocab_size, word_dim)
        # caption embedding
        self.use_bi_gru = use_bi_gru
        self.rnn = nn.GRU(word_dim, embed_size, num_layers, batch_first=True, bidirectional=use_bi_gru)

        # caption Text CNN para

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


    def forward(self, x, lengths):

        """Handles variable size captions
        """
        # Embed word ids to vectors
        x = self.embed(x)
        packed = pack_padded_sequence(x, lengths, batch_first=True)

        # Forward propagate RNN
        out, _ = self.rnn(packed)

        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)
        cap_emb, cap_len = padded
        #print(cap_emb.shape)

        if self.use_bi_gru:
            cap_emb = (cap_emb[:, :, :int(cap_emb.size(2)/2)] +cap_emb[:, :, int(cap_emb.size(2)/2):])/2

        # normalization in the joint embedding space
        if not self.no_txtnorm:
            cap_emb = l2norm(cap_emb, dim=-1)

        # CNN_operator
        # -----------------------------------------------------------------------#
        cap_emb = cap_emb.unsqueeze(1)  # (BN, 1, W, D)
        cap_emb = [F.relu(conv(cap_emb)).squeeze(3) for conv in self.conv2d]
        cap_emb = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in cap_emb]
        cap_emb = torch.cat(cap_emb, 1)
        cap_emb = self.dropout(cap_emb)
        cap_emb = self.fc(cap_emb)
        # -----------------------------------------------------------------------#
        #(BN, saliency_embed_size)

        return cap_emb


class MPMNN(object):
    """
    Multi-Perspective Network (MPMNN) model
    """
    def __init__(self, opt):
        # Build Models
        self.opt = opt
        self.grad_clip = opt.grad_clip

        # local feature
        self.img_enc = EncoderImage(opt.data_name, opt.img_dim, opt.embed_size,
                                    precomp_enc_type=opt.precomp_enc_type,
                                    no_imgnorm=opt.no_imgnorm)
        self.txt_enc = EncoderText(opt.vocab_size, opt.word_dim,
                                   opt.embed_size, opt.num_layers, 
                                   use_bi_gru=opt.bi_gru,  
                                   no_txtnorm=opt.no_txtnorm)

        # global feature
        self.txt_enc_global = Encodertext_global(opt.embed_size, opt.global_embed_size,
                                                 no_txtnorm=opt.no_txtnorm)
        self.img_enc_global = EncoderImage_global(opt.embed_size, opt.global_embed_size,
                                                 no_imgnorm=opt.no_imgnorm)



        self.txt_enc_salience = EncoderText_salience(opt.vocab_size, opt.word_dim, opt.embed_size,
                                                     opt.num_layers, opt.Channel, opt.kernel_num,
                                                     opt.kernel_size, opt.salience_embed_size,
                                                     opt.p, use_bi_gru=opt.bi_gru,
                                                     no_txtnorm=opt.no_txtnorm)
        self.img_enc_salience = EncoderImage_salience(opt.salience_img_dim, opt.salience_embed_size,
                                                      no_imgnorm=opt.no_imgnorm)

        # update feature

        self.inter = cross_modal_interaction(opt)


        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()
            self.txt_enc_global.cuda()
            self.img_enc_global.cuda()
            self.txt_enc_salience.cuda()
            self.img_enc_salience.cuda()
            self.inter.cuda()
            cudnn.benchmark = True

        # Loss and Optimizer
        self.criterion = ContrastiveLoss(opt=opt,
                                         margin=opt.margin,
                                         max_violation=opt.max_violation)
        self.global_criterion = global_ContrastiveLoss(opt=opt,
                                         margin=opt.margin,
                                         max_violation=opt.max_violation)
        self.salience_criterion = salience_ContrastiveLoss(opt=opt,
                                                           margin=opt.margin,
                                                           max_violation=opt.max_violation)
        params = list(self.txt_enc.parameters())
        params += list(self.img_enc.fc.parameters())
        params += list(self.img_enc_global.fc.parameters())
        params += list(self.txt_enc_global.fc.parameters())
        params += list(self.img_enc_salience.fc.parameters())
        params += list(self.txt_enc_salience.fc.parameters())

        self.params = params

        self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)

        self.Eiters = 0

    def state_dict(self):
        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict(),
                      self.img_enc_global.state_dict(), self.txt_enc_global.state_dict(),
                      self.img_enc_salience.state_dict(), self.txt_enc_salience.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0])
        self.txt_enc.load_state_dict(state_dict[1])
        self.img_enc_global.load_state_dict(state_dict[2])
        self.txt_enc_global.load_state_dict(state_dict[3])
        self.img_enc_salience.load_state_dict(state_dict[4])
        self.txt_enc_salience.load_state_dict(state_dict[5])

    def train_start(self):
        """switch to train mode
        """
        self.img_enc.train()
        self.txt_enc.train()
        self.img_enc_global.train()
        self.txt_enc_global.train()
        self.img_enc_salience.train()
        self.txt_enc_salience.train()


    def val_start(self):
        """switch to evaluate mode
        """
        self.img_enc.eval()
        self.txt_enc.eval()
        self.img_enc_global.eval()
        self.txt_enc_global.eval()
        self.img_enc_salience.eval()
        self.txt_enc_salience.eval()


    def forward_emb(self, images, captions, lengths, volatile=False):
        """Compute the image and caption embeddings
        images: (batch_size, 36, 4096--->local+saliency feature)
        captions: (batch_size, 11)
        return: local, length, global, saliency     feature
        """
        # Set mini-batch dataset

        # global mimi-batch dataset
        global_images = images[:, :, 0:2048]
        global_captions = captions
        global_images = Variable(global_images, volatile=volatile)
        global_captions = Variable(global_captions, volatile=volatile)

        # salience mini-batch dataset
        salience_images = images[:, :, 0:2048]
        salience_images = torch.sum(salience_images, dim=1)
        salience_captions = captions
        salience_images = Variable(salience_images, volatile=volatile)
        salience_captions = Variable(salience_captions, volatile=volatile)

        # local mini-batch dataset
        local_images = images[:, :, 0:2048]
        local_images = Variable(local_images, volatile=volatile)
        local_captions = Variable(captions, volatile=volatile)


        if torch.cuda.is_available():
            local_images = local_images.cuda()
            local_captions = local_captions.cuda()
            global_images = global_images.cuda()
            global_captions = global_captions.cuda()
            salience_images = salience_images.cuda()
            salience_captions = salience_captions.cuda()

        # Forward images previous embed_size
        img_emb = self.img_enc(local_images)
        #img_glo = self.img_enc(global_images)

        # local embedding encoding
        # cap_emb (tensor), cap_lens (list)
        cap_emb, cap_lens = self.txt_enc(local_captions, lengths)
        #cap_glo, cap_lens = self.txt_enc(global_captions, lengths)

        # update feature
        img_emb, cap_emb = self.inter(img_emb, cap_emb)
        img_glo = img_emb
        cap_glo = cap_emb

        # global embedding encoding
        # encoding image img_glo[batch_size, regions, global_embed_size]
        img_glo = self.img_enc_global(img_glo)
        # encoding text cap_glo[batch_size, captiond_random_max_lengths, global_embed_size]
        cap_glo = self.txt_enc_global(cap_glo, cap_lens)

        # salience embedding encoding
        img_sal = self.img_enc_salience(salience_images)
        cap_sal = self.txt_enc_salience(salience_captions, lengths)

        return img_emb, cap_emb, cap_lens, img_glo, cap_glo, img_sal, cap_sal

    def forward_loss(self, img_emb, cap_emb, cap_len, **kwargs):
        """Compute the loss given pairs of image and caption embeddings
        """
        loss = self.criterion(img_emb, cap_emb, cap_len)
        self.logger.update('Le', loss.data[0], img_emb.size(0))
        return loss

    def forward_global_loss(self, img_glo, cap_glo, cap_len, **kwargs):
        global_loss = self.global_criterion(img_glo, cap_glo, cap_len)
        self.logger.update('global_Le', global_loss.data[0], img_glo.size(0))
        return global_loss

    def forward_salience_loss(self, img_sal, cap_sal, cap_len, **kwargs):
        salience_loss = self.salience_criterion(img_sal, cap_sal, cap_len)
        self.logger.update('salience_Le', salience_loss.data[0], img_sal.size(0))
        return salience_loss

    def train_emb(self, images, captions, lengths, ids=None, *args):
        """One training step given images and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # data split ---local feature , saliency feature
        # images_local = images[:, :, 0:2048]
        # images_saliency = images[:, :, 2048:]

        # compute the embeddings
        img_emb, cap_emb, cap_lens, img_glo, cap_glo, img_sal, cap_sal = self.forward_emb(images, captions, lengths)

        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss1 = self.forward_loss(img_emb, cap_emb, cap_lens)
        loss2 = self.forward_global_loss(img_glo, cap_glo, cap_lens)
        loss3 = self.forward_salience_loss(img_sal, cap_sal, cap_lens)

        # compute gradient and do SGD step
        (loss1+loss2+loss3).backward()
        #loss1.backward()
        if self.grad_clip > 0:
            clip_grad_norm(self.params, self.grad_clip)
        self.optimizer.step()
