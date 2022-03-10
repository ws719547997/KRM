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
        self.gate = torch.nn.Sigmoid()

        config = BertConfig.from_pretrained(args.bert_model)
        self.bert = BertModel.from_pretrained(args.bert_model,config=config)

        #BERT fixed, i.e. BERT as feature extractor===========
        for param in self.bert.parameters():
            param.requires_grad = False

        self.convs = nn.ModuleList(
            [nn.Conv2d(1, 100, (k, args.bert_hidden_size)) for k in [3,4,5]])
        self.fc1 = torch.nn.Linear(ncha * 100 * 3, nhid)
        self.efc1 = torch.nn.Embedding(len(self.taskcla), nhid)
        if self.nlayers > 1:
            self.fc2 = torch.nn.Linear(nhid, nhid)
            self.efc2 = torch.nn.Embedding(len(self.taskcla), nhid)
            if self.nlayers > 2:
                self.fc3 = torch.nn.Linear(nhid, nhid)
                self.efc3 = torch.nn.Embedding(len(self.taskcla), nhid)

        self.last=torch.nn.ModuleList()
        for t,n in self.taskcla:
            self.last.append(torch.nn.Linear(args.bert_hidden_size,n))
        print('BERT (Fixed) + HAT')

        return

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self,input_ids, segment_ids, input_mask,t,s=1):
        masks = self.mask(t, s=s)
        if self.nlayers==1:
            gfc1=masks
        elif self.nlayers==2:
            gfc1,gfc2=masks
        elif self.nlayers==3:
            gfc1,gfc2,gfc3=masks

        sequence_output, pooled_output = \
            self.bert(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)

        h = sequence_output.unsqueeze(1)
        h = self.drop1(h)
        h = torch.cat([self.conv_and_pool(h, conv) for conv in self.convs], 1)

        # here begin HAT
        h = self.drop2(self.relu(self.fc1(h)))
        h = h * gfc1.expand_as(h)
        if self.nlayers > 1:
            h = self.drop2(self.relu(self.fc2(h)))
            h = h * gfc2.expand_as(h)
            if self.nlayers > 2:
                h = self.drop2(self.relu(self.fc3(h)))
                h = h * gfc3.expand_as(h)
        y = []
        for t, i in self.taskcla:
            y.append(self.last[t](h))
        return y, masks

    def mask(self,t,s=1):
        gfc1=self.gate(s*self.efc1(t))
        if self.nlayers==1: return gfc1
        gfc2=self.gate(s*self.efc2(t))
        if self.nlayers==2: return [gfc1,gfc2]
        gfc3=self.gate(s*self.efc3(t))
        return [gfc1,gfc2,gfc3]

    def get_view_for(self,n,masks):
        if self.nlayers==1:
            gfc1=masks
        elif self.nlayers==2:
            gfc1,gfc2=masks
        elif self.nlayers==3:
            gfc1,gfc2,gfc3=masks
        if n=='fc1.weight':
            return gfc1.data.view(-1,1).expand_as(self.fc1.weight)
        elif n=='fc1.bias':
            return gfc1.data.view(-1)
        elif n=='fc2.weight':
            post=gfc2.data.view(-1,1).expand_as(self.fc2.weight)
            pre=gfc1.data.view(1,-1).expand_as(self.fc2.weight)
            return torch.min(post,pre)
        elif n=='fc2.bias':
            return gfc2.data.view(-1)
        elif n=='fc3.weight':
            post=gfc3.data.view(-1,1).expand_as(self.fc3.weight)
            pre=gfc2.data.view(1,-1).expand_as(self.fc3.weight)
            return torch.min(post,pre)
        elif n=='fc3.bias':
            return gfc3.data.view(-1)
        return None
