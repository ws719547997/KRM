import sys
import torch
from transformers import BertModel, BertConfig
import utils
from torch import nn
import torch.nn.functional as F
import numpy as np

class Net(torch.nn.Module):

    def __init__(self,taskcla,args):

        super(Net,self).__init__()

        self.taskcla=taskcla
        self.args=args
        self.ntasks = len(self.taskcla)
        pdrop1 = 0.2
        pdrop2 = 0.5

        nhid = args.bert_hidden_size
        ncha= 1
        expand_factor = 0.258
        self.N = 3
        self.M = 16
        # """
        self.L = 2  # our architecture has 2 layers

        self.bestPath = -1 * np.ones((self.ntasks, self.L, self.N),
                                     dtype=np.int)  # we need to remember this between the tasks

        self.relu = torch.nn.ReLU()
        self.drop1 = torch.nn.Dropout(pdrop1)
        self.drop2 = torch.nn.Dropout(pdrop2)

        config = BertConfig.from_pretrained(args.bert_model)
        self.bert = BertModel.from_pretrained(args.bert_model,config=config)

        #BERT fixed, i.e. BERT as feature extractor===========
        for param in self.bert.parameters():
            param.requires_grad = False

        self.convs = nn.ModuleList(
            [nn.Conv2d(1, 100, (k, args.bert_hidden_size)) for k in [3,4,5]])

        self.fc1=torch.nn.ModuleList()
        self.sizefc1 = int(expand_factor*nhid)

        self.fc2=torch.nn.ModuleList()
        self.sizefc2 = int(expand_factor*nhid)

        self.last=torch.nn.ModuleList()

        for j in range(self.M):
            self.fc1.append(torch.nn.Linear(ncha * 2 * 3,self.sizefc1))
            self.fc2.append(torch.nn.Linear(self.sizefc1,self.sizefc2))

        for t,n in self.taskcla:
            self.last.append(torch.nn.Linear(self.sizefc2,n))

        print('BERT (Fixed) + PathNet kim')
        return

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self,input_ids, segment_ids, input_mask,t,P=None):
        sequence_output, pooled_output = \
            self.bert(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)

        h = sequence_output.unsqueeze(1)
        h = self.drop1(h)
        h_in = torch.cat([self.conv_and_pool(h, conv) for conv in self.convs], 1)

        # here begain PathNet
        if P is None:
            P = self.bestPath[t]

        h_pre = self.drop2(self.relu(self.fc1[P[0, 0]](h_in)))
        for j in range(1, self.N):
            h_pre = h_pre + self.drop2(self.relu(self.fc1[P[0, j]](h_in)))  # sum activations
        h = h_pre

        h_pre = self.drop2(self.relu(self.fc2[P[1, 0]](h)))
        for j in range(1, self.N):
            h_pre = h_pre + self.drop2(self.relu(self.fc2[P[1, j]](h)))  # sum activations
        h = h_pre

        y = []
        for t, i in self.taskcla:
            y.append(self.last[t](h))
        return y

    def unfreeze_path(self,t,Path):
        #freeze modules not in path P and the ones in bestPath paths for the previous tasks
        for i in range(self.M):
            self.unfreeze_module(self.fc1,i,Path[0,:],self.bestPath[0:t,0,:])
            self.unfreeze_module(self.fc2,i,Path[1,:],self.bestPath[0:t,1,:])
        return

    def unfreeze_module(self,layer,i,Path,bestPath):
        if (i in Path) and (i not in bestPath): #if the current module is in the path and not in the bestPath
            utils.set_req_grad(layer[i],True)
        else:
            utils.set_req_grad(layer[i],False)
        return