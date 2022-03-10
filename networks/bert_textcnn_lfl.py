import sys
import torch
from transformers import BertModel, BertConfig
import utils
from torch import nn
import torch.nn.functional as F

class Net(torch.nn.Module):

    def __init__(self,taskcla,args):

        super(Net,self).__init__()

        self.taskcla=taskcla
        self.args=args

        pdrop1 = 0.2
        pdrop2 = 0.5
        self.nlayers = 2
        nhid = args.bert_hidden_size
        ncha= 1

        self.relu = torch.nn.ReLU()
        self.drop1 = torch.nn.Dropout(pdrop1)
        self.drop2 = torch.nn.Dropout(pdrop2)

        config = BertConfig.from_pretrained(args.bert_model)
        self.bert = BertModel.from_pretrained(args.bert_model,config=config)

        #BERT fixed, i.e. BERT as feature extractor===========
        for param in self.bert.parameters():
            param.requires_grad = False

        self.convs = nn.ModuleList(
            [nn.Conv2d(1, 2, (k, args.bert_hidden_size)) for k in [2,3,4]])
        self.fc1 = torch.nn.Linear(ncha * 2 * 3, nhid)
        self.fc2 = torch.nn.Linear(nhid, nhid)
        self.last=torch.nn.ModuleList()
        for t,n in self.taskcla:
            self.last.append(torch.nn.Linear(nhid,n))
        print('BERT (Fixed) + TEXTCNN + LFL')

        return

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self,input_ids, segment_ids, input_mask):

        sequence_output, pooled_output = \
            self.bert(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)

        h = sequence_output.unsqueeze(1)
        h = torch.cat([self.conv_and_pool(self.drop1(h), conv) for conv in self.convs], 1)

        h = self.drop2(self.relu(self.fc1(h)))
        h = self.drop2(self.relu(self.fc2(h)))

        y = []
        for t, i in self.taskcla:
            y.append(self.last[t](h))
        return y,h

