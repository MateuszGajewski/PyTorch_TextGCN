import torch

from PyTorch_TextGCN.utils import parameter_parser
from lrv_model import XGCN
from trainer import PrepareData
class Explanation:
    def __init__(self, nfeat, nhid, nclass, dropout):
        self.model = XGCN(nfeat, nhid, nclass, dropout)

    def load_model(self, path):
        self.model.load_state_dict(torch.load("/home/arabica/Documents/pythonProject/GCN_PyTorch/PyTorch_TextGCN/model"))
        self.model.set_explainable(True)

    def forward(self, features, adj):
        out = self.model(features, adj)




if __name__ == '__main__':
    args = parameter_parser()
    args.dataset = 'allegro'
    data =  PrepareData(args)
    print(data.features)
    nfeat = 20747
    nhid = 200
    nclass = 5
    dropout = 0
    exp = Explanation(nfeat, nhid, nclass, dropout)
    exp.load_model("")
    exp.forward(data.features, data.adj)

