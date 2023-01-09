from explainable_layers import XGraphConvolution, tensor_to_list, XReLU
import torch.nn as nn
import torch
import torch.nn.functional as F
import warnings
import numpy as np


class XGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, pad = 0, bias = None ):
        super(XGCN, self).__init__()
        self.bias = bias
        if bias is not None:
            raise NotImplementedError
        self.xgc1 = XGraphConvolution(nfeat, nhid, first_layer=True, bias=self.bias)
        self.relu = XReLU()
        self.xgc2 = XGraphConvolution(nhid, nclass, first_layer=False, bias=self.bias)
        self.dropout = dropout
        self.__explainable = False

    def forward(self, embedding, adjacency):
        embedding = self.xgc1(embedding, adjacency)
        embedding = self.relu(embedding)
        embedding = F.dropout(embedding, self.dropout, training=self.training)
        embedding = self.xgc2(embedding, adjacency)
        '''if self.__explainable:
            embedding = embedding.unsqueeze(0)  # batch
            embedding = embedding.unsqueeze(0)  # channel
        else:
            embedding = embedding.unsqueeze(1)  # add channel
        if not self.__explainable:
            embedding = F.log_softmax(embedding, dim=1)'''
        return embedding

    def relprop(self, R, lower_bound, higher_bound, sanity_checks=True):
        if sanity_checks:
            warnings.warn('Sanity checks enabled, this is computationally expensive.')
        else:
            warnings.warn('Sanity checks disabled.')
        assert self.__explainable, 'Relprop invoked but not in explainable mode.'
        if self.bias:
            raise NotImplementedError
        # perform LRP and cache relevance tensors after each layer
        Rs = dict()
        Rs['R_after_out_layer'] = tensor_to_list(R)
        R = self.xfc.relprop(R)
        Rs['R_after_last_fc'] = tensor_to_list(R)
        R = self.xmaxpool2d.relprop(R)
        Rs['R_after_max_pooling_layer'] = tensor_to_list(R)
        R, R_adj2 = self.xgc2.relprop(R)
        Rs['R_after_gcn_2_adjacency_fc'] = tensor_to_list(R_adj2)
        Rs['R_after_gcn_2_feature_fc'] = tensor_to_list(R)
        R, R_adj1 = self.xgc1.relprop(R, lower_bound=lower_bound, higher_bound=higher_bound)
        Rs['R_after_gcn_1_adjacency_fc'] = tensor_to_list(R_adj1)
        Rs['R_after_gcn_1_feature_fc'] = tensor_to_list(R)
        self.set_explainable(False)
        if sanity_checks:
            # check that conversation property holds
            R1 = torch.sum(R).item()
            R2 = torch.sum(torch.tensor(np.array(Rs['R_after_out_layer']))).item()
            diff = abs(R1 - R2)
            if diff >= 1e-8:
                warnings.warn('Conservation property violated with a difference of {}'.format(diff))
        return R, Rs

    def set_explainable(self, explainable):
        if explainable:
            # disable dropout in explainable mode
            self.eval()
        else:
            self.train()
        self.xgc1.set_explainable(explainable)
        self.xgc2.set_explainable(explainable)
        self.__explainable = explainable