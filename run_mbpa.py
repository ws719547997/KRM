import sys,os,argparse,time
import numpy as np
import torch
from config import set_args
import transformers
from tqdm import trange, tqdm
import copy
import logging
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, ConcatDataset
import os
import pickle
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
tstart=time.time()

# Arguments
'''
王松： 需要创建一下ckpt文件夹 在res里创建两个文本存储最后的结果。
--trainmode train 设置为训练模式
'''


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

args = set_args()

if args.output=='':
    args.output='./res/mbpa-result.txt'

########################################################################################################################
use_cuda = True if torch.cuda.is_available() else False
LEARNING_RATE = 3e-5
REPLAY_FREQ = 101

# Seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if use_cuda: torch.cuda.manual_seed(args.seed)

from mbpa.models.MbPAplusplus import ReplayMemory,MbPAplusplus
from dataloaders import bert_snap as dataloader

########################################################################################################################
def train_mbpa(model, memory, train_loader,args=None):
    """
    Train function
    """
    workers = 0
    if use_cuda:
        model.cuda()
        # Number of workers should be 4*num_gpu_available
        # https://discuss.pytorch.org/t/guidelines-for-assigning-num-workers-to-dataloader/813/5
        workers = 4
    # time at the start of training
    start = time.time()
    # TODO:加入dataloader
    param_optimizer = list(model.classifier.named_parameters())
    # parameters that need not be decayed
    no_decay = ['bias', 'gamma', 'beta']
    # Grouping the parameters based on whether each parameter undergoes decay or not.
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}]
    optimizer = transformers.AdamW(
        optimizer_grouped_parameters, lr=LEARNING_RATE)

    # trange is a tqdm wrapper around the normal python range
    for epoch in trange(args.nepoch, desc="Epoch"):
         # Training begins
        print("Training begins")

        # Set our model to training mode (as opposed to evaluation mode)
        model.classifier.train()
        # Tracking variables
        tr_loss = 0
        flag = 0
        nb_tr_examples, nb_tr_steps, num_curr_exs = 0, 0, 0
        # Train the data for one epoch
        for step, batch in enumerate(tqdm(train_loader)):
            # Release file descriptors which function as shared
            # memory handles otherwise it will hit the limit when
            # there are too many batches at dataloader
            batch_cp = copy.deepcopy(batch)
            del batch
            # Perform sparse experience replay after every REPLAY_FREQ steps
            if (step+1) % REPLAY_FREQ == 0:
                # sample 64 examples from memory
                content, attn_masks, labels = memory.sample(sample_size=64)
                if use_cuda:
                    content = content.cuda()
                    attn_masks = attn_masks.cuda()
                    labels = labels.cuda()
                 # Clear out the gradients (by default they accumulate)
                optimizer.zero_grad()
                # Forward pass
                loss, logits = model.classify(content, attn_masks, labels)
                # Backward pass
                loss.backward()
                # Update parameters and take a step using the computed gradient
                optimizer.step()

                # Update tracking variables
                tr_loss += loss.item()
                nb_tr_examples += content.size(0)
                nb_tr_steps += 1

                del content
                del attn_masks
                del labels
                del loss
            # Unpacking the batch items
            content, _ ,attn_masks, labels = batch_cp
            content = content.squeeze(1)
            attn_masks = attn_masks.squeeze(1)
            # number of examples in the current batch
            num_curr_exs = content.size(0)
            # Place the batch items on the appropriate device: cuda if avaliable
            if use_cuda:
                content = content.cuda()
                attn_masks = attn_masks.cuda()
                labels = labels.cuda()
            # Clear out the gradients (by default they accumulate)
            optimizer.zero_grad()
            # Forward pass
            loss, _ = model.classify(content, attn_masks, labels)
            # Get the key representation of documents
            keys = model.get_keys(content, attn_masks)
            # Push the examples into the replay memory
            memory.push(keys.cpu().numpy(), (content.cpu().numpy(),
                                             attn_masks.cpu().numpy(), labels.cpu().numpy()))
            # delete the batch data to freeup gpu memory
            del keys
            del content
            del attn_masks
            del labels
            # Backward pass
            loss.backward()
            # Update parameters and take a step using the computed gradient
            optimizer.step()
            # Update tracking variables
            tr_loss += loss.item()
            nb_tr_examples += num_curr_exs
            nb_tr_steps += 1

        now = time.time()
        print("Train loss: {}".format(tr_loss/nb_tr_steps))
        print("Time taken till now: {} hours".format((now-start)/3600))

def test_mbpa(model, memory,test_loader):
    """
    evaluate the model for accuracy
    """
    # time at the start of validation
    start = time.time()
    if use_cuda:
        model.cuda()

    # Tracking variables
    flag = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)

    print("Validation step started...")
    for batch in tqdm(test_loader):
        batch_cp = copy.deepcopy(batch)
        del batch
        contents, _ ,attn_masks, labels = batch_cp
        if use_cuda:
            contents = contents.squeeze(1).cuda()
            attn_masks = attn_masks.squeeze(1).cuda()

        if flag % args.inter == 0:
            # 更新分类器，但是非常慢

            keys = model.get_keys(contents, attn_masks)
            retrieved_batches = memory.get_neighbours(keys.cpu().numpy())
            del keys
            # Iterate over the test batch to calculate label for each document(i.e,content)
            # and store them in a list for comparision later
            for content, attn_mask, (rt_contents, rt_attn_masks, rt_labels) in tqdm(zip(contents, attn_masks, retrieved_batches), total=len(contents)):
                if use_cuda:
                    rt_contents = rt_contents.cuda()
                    rt_attn_masks = rt_attn_masks.cuda()
                    rt_labels = rt_labels.cuda()

                logits = model.infer(content, attn_mask,
                                     rt_contents, rt_attn_masks, rt_labels)
                _, pred = logits.max(1)
                predict_all = np.append(predict_all, pred.cpu().numpy())

        else:
            # 直接预测 不弄了好吧!
            for content, attn_mask in tqdm(
                    zip(contents, attn_masks), total=len(contents)):
                logits = model.test_no_kmeans(content, attn_mask)
                _, pred = logits.max(1)
                predict_all = np.append(predict_all,  pred.cpu().numpy())

        flag += 1
        labels = labels.numpy()
        labels_all = np.append(labels_all, labels)
        # del labels
    f1 = f1_score(labels_all, predict_all, pos_label=0)
    acc = accuracy_score(labels_all, predict_all)

    end = time.time()
    print("Time taken for validation {} minutes".format((end-start)/60))
    return acc,f1

def save_checkpoint(t,model_dict, memory=None):
    """
    Function to save a model checkpoint to the specified location
    """
    checkpoints_dir = 'mbpa/ckpt/'
    if not os.path.exists(checkpoints_dir):
        os.mkdir(checkpoints_dir)
    checkpoints_file = str(t)+'.pth'
    torch.save(model_dict, os.path.join(checkpoints_dir, checkpoints_file))
    if memory is not None:
        with open(checkpoints_dir+str(t)+'.pkl', 'wb') as f:
            pickle.dump(memory, f)
###################################################

if __name__ == '__main__':
    print('Load data...')
    data,taskcla=dataloader.get(logger=logger,args=args)

    print('\nTask info =',taskcla)
    print('Inits...')

    if args.trainmode == 'train':
    # 一直就用这一个模型
        model_train = MbPAplusplus(args)
        # 新建空的记忆，慢慢的往里加入。
        memory = ReplayMemory()

        # Loop tasks
        for t, ncla in taskcla:
            print('*'*100)
            print('Task {:2d} ({:s})'.format(t,data[t]['name']))
            print('*'*100)

            # Get data
            train=data[t]['train']
            num_train_steps=data[t]['num_train_steps']
            task=t

            train_sampler = RandomSampler(train)
            train_dataloader = DataLoader(train, sampler=train_sampler, batch_size=args.train_batch_size)

            print('-'*100)
            train_mbpa(model_train, memory,train_dataloader,args=args)
            save_checkpoint(t,model_train.save_state(),memory=memory.memory)
            print('save_model_finished!')
    else:
        # 先load data 按照指定的序号
        task = 0
        print(data[task]['name'])
        test = data[task]['test']
        test_sampler = SequentialSampler(test)
        test_dataloader = DataLoader(test, sampler=test_sampler, batch_size=args.eval_batch_size)
        print('load dataset successed!')

        # 完了再load moad 制定模型
        # 如果需要测评最后模型的性能
        # args.trainmode = '21'
        # 否则这样就行了
        model_test = MbPAplusplus(args,model_state=torch.load('mbpa/ckpt/'+args.trainmode+'.pth'))
        print('load '+'mbpa/ckpt/'+args.trainmode+'.pth'+' successed!')
        buffer = {}
        with open('mbpa/ckpt/'+args.trainmode+'.pkl', 'rb') as f:
            buffer = pickle.load(f)
        memory =ReplayMemory(buffer=buffer)
        print('load '+'mbpa/ckpt/'+args.trainmode+'.pkl'+' successed!')

        test_acc, test_f1 = test_mbpa(model_test, memory, test_dataloader)

        # Test

        with open(args.output,'a+') as f:
            f.write(args.trainmode + '-task (when first learned): acc, '+str(test_acc)+' f1, '+str(test_f1)+'\n')

        # appr.decode(train_dataloader)
        # break

    # Done

    print('[Elapsed time = {:.1f} h]'.format((time.time()-tstart)/(60*60)))


########################################################################################################################
