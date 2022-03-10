import sys, time
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
    def __init__(self, model,taskcla ,nepochs=100, sbatch=64, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=3, clipgrad=10000,
                 args=None, logger=None):
        # def __init__(self,model,nepochs=100,sbatch=64,lr=0.001,lr_min=1e-5,lr_factor=2,lr_patience=3,clipgrad=10000,args=None,logger=None):
        self.model = model
        self.model_old = None

        self.nepochs = nepochs
        self.sbatch = sbatch
        self.lr = lr
        self.lr_min = lr_min
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.clipgrad = clipgrad

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = self._get_optimizer()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.reg=0.0001

        print('CONTEXTUAL + IMM_MEAN + RNN NCL')

        return

    def _get_optimizer(self, lr=None):
        if lr is None: lr = self.lr
        return torch.optim.SGD(self.model.parameters(), lr=lr)

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
            print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.1f}% |'.format(e + 1,
                                                                                                        1000 * self.sbatch * (
                                                                                                                    clock1 - clock0) / len(
                                                                                                            train),
                                                                                                        1000 * self.sbatch * (
                                                                                                                    clock2 - clock1) / len(
                                                                                                            train),
                                                                                                        train_loss,
                                                                                                        100 * train_acc),
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

            utils.set_model_(self.model, best_model)
            if t > 0:
                model_state = utils.get_model(self.model)
                model_old_state = utils.get_model(self.model_old)
                for name, param in self.model.named_parameters():
                    # model_state[name]=(1-self.alpha)*model_old_state[name]+self.alpha*model_state[name]
                    model_state[name] = (model_state[name] + model_old_state[name] * t) / (t + 1)
                utils.set_model_(self.model, model_state)

            self.model_old = deepcopy(self.model)
            self.model_old.eval()
            utils.freeze_model(self.model_old)

        return

    def train_epoch(self, t, data, iter_bar):
        self.model.train()
        # Loop batches
        for step, batch in enumerate(iter_bar):
            batch = [
                bat.to(self.device) if bat is not None else None for bat in batch]
            input_ids, segment_ids, input_mask, targets = batch
            task = torch.autograd.Variable(torch.LongTensor([t]).cuda(), requires_grad=False)

            # Forward
            outputs= self.model.forward(input_ids, segment_ids, input_mask)
            output = outputs[t]
            loss = self.ce(output, targets, t)
            iter_bar.set_description('Train Iter (loss=%5.3f)' % loss.item())

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()

        return

    def eval(self, t, data):
        total_loss = 0
        total_acc = 0
        total_num = 0
        predict_all = np.array([], dtype=int)
        labels_all = np.array([], dtype=int)
        self.model.eval()

        for step, batch in enumerate(data):
            batch = [
                bat.to(self.device) if bat is not None else None for bat in batch]
            input_ids, segment_ids, input_mask, targets = batch
            real_b = input_ids.size(0)
            task = torch.autograd.Variable(torch.LongTensor([t]).cuda(), requires_grad=False)


            outputs = self.model.forward(input_ids, segment_ids, input_mask)
            output = outputs[t]
            loss = self.ce(output,targets,t)

            _, pred = output.max(1)

            # Log
            total_loss += loss.data.cpu().numpy().item() * real_b
            total_num += real_b

            targets = targets.data.cpu().numpy()
            pred = pred.data.cpu().numpy()

            labels_all = np.append(labels_all, targets)
            predict_all = np.append(predict_all, pred)

        f1 = f1_score(labels_all, predict_all, pos_label=0)
        acc = accuracy_score(labels_all, predict_all)
        return total_loss / total_num, acc, f1

    def ce(self, output, targets, t):

        loss_reg = 0
        if t > 0:
            for (name, param), (_, param_old) in zip(self.model.named_parameters(), self.model_old.named_parameters()):
                loss_reg += torch.sum((param_old - param).pow(2)) / 2

        # Cross entropy loss
        loss_ce = self.criterion(output, targets)

        return loss_ce + self.reg * loss_reg
