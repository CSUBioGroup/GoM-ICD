import torch
from torch import nn
from src.transformer import *

    
class AttentionLayer(nn.Module):

    def __init__(self, args, vocab):

        super(AttentionLayer, self).__init__()
        self.attention_mode = args.attention_mode
        self.size = args.hidden_size*2
        self.n_labels = vocab.label_num
        self.first_linears =nn.Linear(self.size,self.n_labels, bias=True)
        self.transformer_layer = TransformerLayers_PostLN(layersNum=1, feaSize=self.size, dk=args.dk,multiNum=args.multiNum,dropout=args.trans_drop)

    def forward(self, x , label_batch=None):
        att=None
        if self.attention_mode == "text_label":
            att = self.transformer_layer(qx=label_batch, kx=x, vx=x)
            att = att[0]
            weighted_output = self.first_linears.weight.mul(att).sum(dim=2).add(self.first_linears.bias)
        logits = weighted_output.reshape(weighted_output.shape[0], -1)

        return logits,att