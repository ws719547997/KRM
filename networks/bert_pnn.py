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
        self.ntasks = len(self.taskcla)
        pdrop1 = 0.2
        pdrop2 = 0.5
        self.nlayers = 2
        nhid = args.bert_hidden_size
        ncha= 1
        expand_factor = 1.117

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
        self.fc1 = torch.nn.ModuleList()
        self.Vf1scale = torch.nn.ModuleList()
        self.Vf1 = torch.nn.ModuleList()
        self.Uf1 = torch.nn.ModuleList()
        self.sizefc1 = int(expand_factor * nhid / self.ntasks)

        self.fc2 = torch.nn.ModuleList()
        self.sizefc2 = int(expand_factor * nhid / self.ntasks)

        self.last = torch.nn.ModuleList()
        self.Vflscale = torch.nn.ModuleList()
        self.Vfl = torch.nn.ModuleList()
        self.Ufl = torch.nn.ModuleList()

        # declare task columns subnets
        for t, n in self.taskcla:
            self.fc1.append(torch.nn.Linear(ncha * 2 * 3,self.sizefc1))
            self.fc2.append(torch.nn.Linear(self.sizefc1, self.sizefc2))
            self.last.append(torch.nn.Linear(self.sizefc2, n))

            if t > 0:
                # lateral connections with previous columns
                self.Vf1scale.append(torch.nn.Embedding(1, t))
                self.Vf1.append(torch.nn.Linear(t * self.sizefc1, self.sizefc1))
                self.Uf1.append(torch.nn.Linear(self.sizefc1, self.sizefc1))

                self.Vflscale.append(torch.nn.Embedding(1, t))
                self.Vfl.append(torch.nn.Linear(t * self.sizefc2, self.sizefc2))
                self.Ufl.append(torch.nn.Linear(self.sizefc2, n))
        print('BERT (Fixed) + PNN')

        return

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self,input_ids, segment_ids, input_mask,t):
        sequence_output, pooled_output = \
            self.bert(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)

        h = sequence_output.unsqueeze(1)
        h = self.drop1(h)
        h_in = torch.cat([self.conv_and_pool(h, conv) for conv in self.convs], 1)

        # here begain PNN
        h = self.drop2(self.relu(self.fc1[t](h_in)))
        if t>0: #compute activations for previous columns/tasks
            h_prev3 = []
            for j in range(t):
                h_prev3.append(self.drop2(self.relu(self.fc1[j](h_in))))

        h_pre = self.fc2[t](h) #current column/task
        if t>0: #compute activations for previous columns/tasks & sum laterals
            hf_prev2 = [self.drop2(self.relu(self.fc2[j](h_prev3[j]))) for j in range(t)]

            # print('h_pre: ',h_pre.size())
            # print('h_prev3: ',h_prev3[0].size())

            h_pre = h_pre + self.Uf1[t-1](self.relu(self.Vf1[t-1](torch.cat([self.Vf1scale[t-1].weight[0][j] * h_prev3[j] for j in range(t)],1))))
        h=self.drop2(self.relu(h_pre))

        y = []
        for tid,i in self.taskcla:
            if t>0 and tid<t:
                h_pre = self.last[tid](hf_prev2[tid]) #current column/task
                if tid>0:
                    #sum laterals, no non-linearity for last layer
                    h_pre = h_pre + self.Ufl[tid-1](self.Vfl[tid-1](torch.cat([self.Vflscale[tid-1].weight[0][j] * hf_prev2[j] for j in range(tid)],1)))
                y.append(h_pre)
            else:
                y.append(self.last[tid](h))
        return y


    def unfreeze_column(self,t):
        utils.set_req_grad(self.fc1[t],True)
        utils.set_req_grad(self.fc2[t],True)
        utils.set_req_grad(self.last[t],True)
        if t>0:
            utils.set_req_grad(self.Vf1[t-1],True)
            utils.set_req_grad(self.Uf1[t-1],True)
            utils.set_req_grad(self.Vflscale[t-1],True)
            utils.set_req_grad(self.Vfl[t-1],True)
            utils.set_req_grad(self.Ufl[t-1],True)

        #freeze other columns
        for i in range(self.ntasks):
            if i!=t:
                utils.set_req_grad(self.fc1[i],False)
                utils.set_req_grad(self.fc2[i],False)
                utils.set_req_grad(self.last[i],False)
                if i>0:
                    utils.set_req_grad(self.Vf1scale[i-1],False)
                    utils.set_req_grad(self.Vf1[i-1],False)
                    utils.set_req_grad(self.Uf1[i-1],False)
                    utils.set_req_grad(self.Vflscale[i-1],False)
                    utils.set_req_grad(self.Vfl[i-1],False)
                    utils.set_req_grad(self.Ufl[i-1],False)
        return
