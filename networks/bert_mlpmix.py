import sys
import torch
from transformers import BertModel, BertConfig
from torch import nn
from functools import partial
from einops.layers.torch import Rearrange, Reduce


class Net(torch.nn.Module):

    def __init__(self,taskcla,args):

        super(Net,self).__init__()

        self.taskcla=taskcla
        self.args=args

        config = BertConfig.from_pretrained(args.bert_model)
        self.bert = BertModel.from_pretrained(args.bert_model,config=config)

        #BERT fixed, i.e. BERT as feature extractor===========
        for param in self.bert.parameters():
            param.requires_grad = False

        self.MLP = MLPMixer(
            image_size=args.bert_hidden_size * args.max_seq_length,
            patch_size=args.bert_hidden_size,
            dim=args.bert_hidden_size,
            depth=6,
            dropout= 0.5,
            num_classes=2
        )

        self.last=torch.nn.ModuleList()
        for t,n in self.taskcla:
            self.last.append(torch.nn.Linear(args.bert_hidden_size,n))
        print('BERT (Fixed) + MLPMIXER')

        return

    def forward(self,input_ids, segment_ids, input_mask):
        sequence_output, pooled_output = \
            self.bert(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)

        h = sequence_output.unsqueeze(1)
        h = self.MLP(h)
        # h.squeeze(1)
        #loss ==============
        y=[]
        for t,i in self.taskcla:
            y.append(self.last[t](h))
        return y

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x

def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = nn.Linear):
    return nn.Sequential(
        dense(dim, dim * expansion_factor),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(dim * expansion_factor, dim),
        nn.Dropout(dropout)
    )

def MLPMixer(*, image_size, patch_size, dim, depth, num_classes, expansion_factor = 4, dropout = 0.):
    num_patches = (768 // 768) * (128 // 1)
    chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear

    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = 1, p2 = 768),
        nn.Linear(768, dim),
        *[nn.Sequential(
            PreNormResidual(dim, FeedForward(num_patches, expansion_factor, dropout, chan_first)),
            PreNormResidual(dim, FeedForward(dim, expansion_factor, dropout, chan_last))
        ) for _ in range(depth)],
        nn.LayerNorm(dim),
        Reduce('b n c -> b c', 'mean')
        # nn.Linear(dim, num_classes)
    )