import sys,time
import numpy as np
import torch
# from copy import deepcopy

import utils
from copy import deepcopy
from tqdm import tqdm, trange
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

class Appr(object):
    def __init__(self,model,nepochs=100,sbatch=64,lr=0.05,lr_min=1e-4,lr_factor=3,lr_patience=3,clipgrad=10000,args=None,taskcla=None,logger=None):
    # def __init__(self,model,nepochs=100,sbatch=64,lr=0.001,lr_min=1e-5,lr_factor=2,lr_patience=3,clipgrad=10000,args=None,logger=None):
        self.model=model
        self.initial_model=deepcopy(model)

        self.N = self.model.N  # from paper, number of distinct modules permitted in a pathway
        self.M = self.model.M  # from paper, total num modules
        self.L = self.model.L

        self.generations = 20  # Grid search = [5,10,20,50,100,200]; best was 20
        self.P = 2

        self.nepochs = args.nepochs // self.generations
        self.sbatch=sbatch
        self.lr=lr
        self.lr_min=lr_min
        self.lr_factor=lr_factor
        self.lr_patience=lr_patience
        self.clipgrad=clipgrad

        self.criterion=torch.nn.CrossEntropyLoss()
        self.optimizer=self._get_optimizer()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print('CONTEXTUAL + MLP pathnet')

        return

    def _get_optimizer(self,lr=None):
        if lr is None: lr=self.lr
        return torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),lr=lr)

    def train(self, t, train, valid, args):

        if t>0: #reinit modules not in bestpath with random, according to the paper
            # layers = ['convs','fc1','fc2'] #为了试验对比 没在conv层使用path
            layers = ['fc1','fc2'] #mlp
            for (n,p),(m,q) in zip(self.model.named_parameters(),self.initial_model.named_parameters()):
                if 'bert' in n:
                    continue
                if n==m:
                    layer,module,par = n.split(".")
                    module = int(module)
                    if layer in layers:
                        if module not in self.model.bestPath[0:t,layers.index(layer)]:
                            p.data = deepcopy(q.data)

        #init path for this task
        Path = np.random.randint(0,self.M-1,size=(self.P,self.L,self.N))
        guesses = list(range(self.M))
        lr=[]
        patience=[]
        best_loss=[]
        for p in range(self.P):
            lr.append(self.lr)
            patience.append(self.lr_patience)
            best_loss.append(np.inf)
            for j in range(self.L):
                np.random.shuffle(guesses)
                Path[p,j,:] = guesses[:self.N] #do not repeat modules

        winner = 0
        best_path_model = utils.get_model(self.model)
        best_loss_overall=np.inf

        try:
            for g in range(self.generations):
                if np.max(lr)<self.lr_min: break

                for p in range(self.P):
                    if lr[p]<self.lr_min: continue

                    # train only the modules in the current path, minus the ones in the model.bestPath
                    self.model.unfreeze_path(t,Path[p])

                    # the optimizer trains solely the params for the current task
                    self.optimizer=self._get_optimizer(lr[p])

                    for e in range(self.nepochs):
                        # Train
                        clock0 = time.time()
                        iter_bar = tqdm(train, desc='Train Iter (loss=X.XXX)')
                        self.train_epoch(t, train, iter_bar,Path[p])
                        clock1 = time.time()
                        train_loss, train_acc, _ = self.eval(t, train,Path[p])
                        clock2 = time.time()
                        print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.1f}% |'.format(e + 1,1000 * self.sbatch * (clock1 - clock0) / len(train),
                                                                                                                    1000 * self.sbatch * (clock2 - clock1) / len(train),
                                                                                                                    train_loss,100 * train_acc),
                              end='')
                        # Valid
                        valid_loss, valid_acc, _ = self.eval(t, valid,Path[p])
                        print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss, 100 * valid_acc), end='')

                        # Save the winner
                        if valid_loss<best_loss_overall:
                            best_loss_overall=valid_loss
                            best_path_model = utils.get_model(self.model)
                            winner=p
                            print(' B',end='')

                        # Adapt lr
                        if valid_loss<best_loss[p]:
                            best_loss[p]=valid_loss
                            patience[p]=self.lr_patience
                            print(' *',end='')
                        else:
                            patience[p]-=1
                            if patience[p]<=0:
                                lr[p]/=self.lr_factor
                                print(' lr={:.1e}'.format(lr[p]),end='')
                                if lr[p]<self.lr_min:
                                    print()
                                    break
                                patience[p]=self.lr_patience
                        print()
                        # Restore best
                utils.set_model_(self.model, best_path_model)
                print('| Winning path: {:3d} | Best loss: {:.3f} |'.format(winner + 1, best_loss_overall))

                # Keep the winner and mutate it
                print('Mutating')
                probability = 1 / (self.N * self.L)  # probability to mutate
                for p in range(self.P):
                    if p != winner:
                        best_loss[p] = np.inf
                        lr[p] = lr[winner]
                        patience[p] = self.lr_patience
                        for j in range(self.L):
                            for k in range(self.N):
                                Path[p, j, k] = Path[winner, j, k]
                                if np.random.rand() < probability:
                                    Path[p, j, k] = (Path[p, j, k] + np.random.randint(-2,
                                                                                       3)) % self.M  # add int in [-2,2] to the path, this seems yet another hyperparam

        except KeyboardInterrupt:
            print()

        # save the best path into the model
        self.model.bestPath[t] = Path[winner]
        print(self.model.bestPath[t])

        return



    def train_epoch(self,t,data,iter_bar,Path):
        self.model.train()
        # Loop batches
        for step, batch in enumerate(iter_bar):
            batch = [
                bat.to(self.device) if bat is not None else None for bat in batch]
            input_ids, segment_ids, input_mask, targets= batch
            task=torch.autograd.Variable(torch.LongTensor([t]).cuda(),requires_grad=False)

            # Forward
            outputs= self.model.forward(input_ids, segment_ids, input_mask,task,Path)
            output=outputs[t]
            loss=self.criterion(output,targets)
            iter_bar.set_description('Train Iter (loss=%5.3f)' % loss.item())

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(filter(lambda p: p.requires_grad, self.model.parameters()), self.clipgrad)
            self.optimizer.step()

        return

    def eval(self,t,data,Path=None):
        total_loss=0
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

            outputs = self.model.forward(input_ids, segment_ids, input_mask, task,Path)
            output=outputs[t]
            loss = self.criterion(output, targets)

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
