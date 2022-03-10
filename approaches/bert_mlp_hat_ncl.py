import sys, time
import numpy as np
import torch
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
    def __init__(self, model, nepochs=100, sbatch=64, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=3, clipgrad=10000,
                 args=None, logger=None):
        # def __init__(self,model,nepochs=100,sbatch=64,lr=0.001,lr_min=1e-5,lr_factor=2,lr_patience=3,clipgrad=10000,args=None,logger=None):
        self.model = model
        # self.initial_model=deepcopy(model)

        self.nepochs = nepochs
        self.sbatch = sbatch
        self.lr = lr
        self.lr_min = lr_min
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.clipgrad = clipgrad

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = self._get_optimizer()

        self.lamb = 0.75  # Grid search = [0.1, 0.25, 0.5, 0.75, 1, 1.5, 2.5, 4]; chosen was 0.75
        self.smax = 400  # Grid search = [25, 50, 100, 200, 400, 800]; chosen was 400

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.mask_pre = None
        self.mask_back = None
        self.epoch_np = 0
        self.network_capacity_usage = np.zeros((40,), dtype=np.float32)
        self.network_active = np.zeros((40,), dtype=np.float32)

        print('CONTEXTUAL + hat NCL')

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
            # Restore best
            utils.set_model_(self.model, best_model)

            task = torch.autograd.Variable(torch.LongTensor([t]).cuda(), volatile=False)
            mask = self.model.mask(task, s=self.smax)
            for i in range(len(mask)):
                mask[i] = torch.autograd.Variable(mask[i].data.clone(), requires_grad=False)
            if t == 0:
                self.mask_pre = mask
            else:
                for i in range(len(self.mask_pre)):
                    self.mask_pre[i] = torch.max(self.mask_pre[i], mask[i])

            # Weights mask

            self.mask_back = {}
            for n, _ in self.model.named_parameters():
                vals = self.model.get_view_for(n, self.mask_pre)
                if vals is not None:
                    self.mask_back[n] = 1 - vals

            self.epoch_np += 1
        return

    def train_epoch(self, t, data, iter_bar, thres_cosh=50, thres_emb=6):
        self.model.train()
        # Loop batches
        for step, batch in enumerate(iter_bar):
            batch = [
                bat.to(self.device) if bat is not None else None for bat in batch]
            input_ids, segment_ids, input_mask, targets = batch
            task = torch.autograd.Variable(torch.LongTensor([t]).cuda(), requires_grad=False)
            s = (self.smax - 1 / self.smax) * step / len(batch) + 1 / self.smax

            # Forward
            outputs, masks = self.model.forward(input_ids, segment_ids, input_mask, task, s=s)
            output = outputs[t]
            loss, _ = self.ce(output, targets, masks)
            iter_bar.set_description('Train Iter (loss=%5.3f)' % loss.item())

            # Backward
            self.optimizer.zero_grad()
            loss.backward()

            if t > 0:
                for n, p in self.model.named_parameters():
                    if n in self.mask_back and (p.grad is not None):
                        p.grad.data *= self.mask_back[n]

            # Compensate embedding gradients
            for n, p in self.model.named_parameters():
                if n.startswith('e') and (p.grad is not None):
                    num = torch.cosh(torch.clamp(s * p.data, -thres_cosh, thres_cosh)) + 1
                    den = torch.cosh(p.data) + 1
                    p.grad.data *= self.smax / s * num / den

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()

            for n, p in self.model.named_parameters():
                if n.startswith('e'):
                    p.data = torch.clamp(p.data, -thres_emb, thres_emb)

        return

    def eval(self, t, data):
        total_loss = 0
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

            outputs, masks = self.model.forward(input_ids, segment_ids, input_mask, task, s=self.smax)
            output = outputs[t]
            loss, reg = self.ce(output, targets, masks)

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

    def ce(self, outputs, targets, masks):
        reg = 0
        count = 0
        if self.mask_pre is not None:
            for m, mp in zip(masks, self.mask_pre):
                aux = 1 - mp
                reg += (m * aux).sum()
                count += aux.sum()
        else:
            for m in masks:
                reg += m.sum()
                count += np.prod(m.size()).item()
        reg /= count
        return self.criterion(outputs, targets) + self.lamb * reg, reg
