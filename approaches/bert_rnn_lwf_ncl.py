import sys,time
import numpy as np
import torch
from copy import deepcopy
# from copy import deepcopy

import utils
from tqdm import tqdm, trange
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

rnn_weights = [
    'mcl.lstm.rnn.weight_ih_l0',
    'mcl.lstm.rnn.weight_hh_l0',
    'mcl.lstm.rnn.bias_ih_l0',
    'mcl.lstm.rnn.bias_hh_l0',
    'mcl.gru.rnn.weight_ih_l0',
    'mcl.gru.rnn.weight_hh_l0',
    'mcl.gru.rnn.bias_ih_l0',
    'mcl.gru.rnn.bias_hh_l0']


class Appr(object):
    def __init__(self,model,nepochs=100,sbatch=64,lr=0.05,lr_min=1e-4,lr_factor=3,lr_patience=3,clipgrad=10000,args=None,logger=None):
    # def __init__(self,model,nepochs=100,sbatch=64,lr=0.001,lr_min=1e-5,lr_factor=2,lr_patience=3,clipgrad=10000,args=None,logger=None):
        self.model=model
        self.model_old = None
        self.fisher = None
        # self.initial_model=deepcopy(model)

        self.nepochs=nepochs
        self.sbatch=sbatch
        self.lr=lr
        self.lr_min=lr_min
        self.lr_factor=lr_factor
        self.lr_patience=lr_patience
        self.clipgrad=clipgrad

        self.criterion=torch.nn.CrossEntropyLoss()
        self.optimizer=self._get_optimizer()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_gpu = torch.cuda.device_count()

        self.lamb = 2  # Grid search = [0.1, 0.5, 1, 2, 4, 8, 10]; best was 2
        self.T = 1  # lambda in loss function

        print('CONTEXTUAL + LwF + RNN NCL')

        return

    def _get_optimizer(self,lr=None):
        if lr is None: lr=self.lr
        return torch.optim.SGD(self.model.parameters(),lr=lr)

    def train(self, t, train, valid, args):
        best_loss = np.inf
        best_model = utils.get_model(self.model)
        lr = self.lr
        patience = self.lr_patience
        self.optimizer = self._get_optimizer(lr)

        for e in range(self.nepochs):
            # Train
            clock0 = time.time()
            iter_bar = tqdm(train, desc='Train Iter (loss=X.XXX)')
            self.train_epoch(t, train, iter_bar)
            clock1 = time.time()
            train_loss, train_acc, _ = self.eval(t, train)
            clock2 = time.time()
            print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.1f}% |'.format(e + 1,1000 * self.sbatch * (clock1 - clock0) / len(train),
                                                                                                        1000 * self.sbatch * (clock2 - clock1) / len(train),
                                                                                                        train_loss,100 * train_acc),
                  end='')
            # Valid
            valid_loss, valid_acc, _ = self.eval(t, valid)
            print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss, 100 * valid_acc), end='')
            # Adapt lr
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_model = utils.get_model(self.model)
                patience = self.lr_patience
                print(' *', end='')
            else:
                patience -= 1
                if patience <= 0:
                    lr /= self.lr_factor
                    print(' lr={:.1e}'.format(lr), end='')
                    if lr < self.lr_min:
                        print()
                        break
                    patience = self.lr_patience
                    self.optimizer = self._get_optimizer(lr)
            print()
            # Restore best
            utils.set_model_(self.model, best_model)

            # Update old
            self.model_old = deepcopy(self.model)
            self.model_old.eval()
            utils.freeze_model(self.model_old)

        return

    def train_epoch(self,t,data,iter_bar):
        self.model.train()
        # Loop batches
        for step, batch in enumerate(iter_bar):
            batch = [
                bat.to(self.device) if bat is not None else None for bat in batch]
            input_ids, segment_ids, input_mask, targets= batch
            task=torch.autograd.Variable(torch.LongTensor([t]).cuda(),requires_grad=False)

            # Forward old model
            targets_old = None
            if t > 0:
                targets_old = self.model_old.forward(input_ids, segment_ids, input_mask)

            # Forward
            outputs=self.model.forward(input_ids, segment_ids, input_mask)
            loss=self.ce(t,targets_old,outputs,targets)
            iter_bar.set_description('Train Iter (loss=%5.3f)' % loss.item())

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.clipgrad)
            self.optimizer.step()

        return

    def eval(self,t,data):
        total_loss=0
        total_acc=0
        total_num=0
        predict_all = np.array([], dtype=int)
        labels_all = np.array([], dtype=int)
        self.model.eval()


        for step, batch in enumerate(data):
            batch = [
                bat.to(self.device) if bat is not None else None for bat in batch]
            input_ids, segment_ids, input_mask, targets= batch
            real_b=input_ids.size(0)
            task=torch.autograd.Variable(torch.LongTensor([t]).cuda(),requires_grad=False)

            targets_old = None
            if t > 0:
                targets_old = self.model_old.forward(input_ids, segment_ids, input_mask)

            outputs = self.model.forward(input_ids, segment_ids, input_mask)
            loss=self.ce(t,targets_old,outputs,targets)

            output = outputs[t]
            _,pred=output.max(1)

            # Log
            total_loss+=loss.data.cpu().numpy().item()*real_b
            total_num+=real_b

            targets = targets.data.cpu().numpy()
            pred = pred.data.cpu().numpy()

            labels_all = np.append(labels_all, targets)
            predict_all = np.append(predict_all, pred)

        f1 = f1_score(labels_all,predict_all,pos_label=0)
        acc = accuracy_score(labels_all,predict_all)
        return total_loss/total_num,acc,f1

    def ce(self,t,targets_old,outputs,targets):
        # TODO: warm-up of the new layer (paper reports that improves performance, but negligible)

        # Knowledge distillation loss for all previous tasks
        loss_dist=0
        for t_old in range(0,t):
            loss_dist+=utils.cross_entropy(outputs[t_old],targets_old[t_old],exp=1/self.T)

        # Cross entropy loss
        loss_ce=self.criterion(outputs[t],targets)

        # We could add the weight decay regularization mentioned in the paper. However, this might not be fair/comparable to other approaches

        return loss_ce+self.lamb*loss_dist