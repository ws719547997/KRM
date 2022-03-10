import time
import numpy as np
import torch
from config import set_args
import logging
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, ConcatDataset
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
tstart = time.time()

# Arguments


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

args = set_args()

if args.output == '':
    args.output = './res/' + args.experiment + '_' + args.approach + '_' + str(args.n_layers) + '_' + str(
        args.nepoch) + '.txt'

performance_output = args.output + '_performance'
performance_output_forward = args.output + '_forward_performance'

########################################################################################################################

# Seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available(): torch.cuda.manual_seed(args.seed)
else:
    print('[CUDA unavailable]')

# Args -- Experiment
if args.experiment == 'w2v_jd':
    from dataloaders import w2v_jd as dataloader
elif args.experiment == 'bert_snap':
    from dataloaders import bert_snap as dataloader
    args.bert_model += 'bert-base-uncased'
elif args.experiment == 'bert_jd':
    from dataloaders import bert_jd as dataloader
    args.bert_model += 'bert-base-chinese'
elif args.experiment == 'bert_amz':
    from dataloaders import bert_amz as dataloader
    args.bert_model +='bert-base-uncased'
elif args.experiment == 'bert':
    from dataloaders import bert_jd as dataloader

if args.approach == 'trans_qkv_turn_mlphat_ncl':
    from guaguanet import bert_trans_qkv_turn as knowledge_retain_net
if 'trans_qkv' in args.approach:
    from guaguanet import bert_trans_qkv as knowledge_retain_net
if 'trans_kv' in args.approach:
    from guaguanet import bert_trans_kv_pos as knowledge_retain_net

if 'mlphat'in args.approach:
    from guaguanet import w2v_hat as knowledge_mining_net
    from guaguanet import joint_hat_ncl as mining
elif 'hat' in args.approach:
    from guaguanet import w2v_kim_hat as knowledge_mining_net
    from guaguanet import joint_trans_hat_ncl as mining

if 'pathnet' in args.approach:
    from guaguanet import w2v_pathnet as knowledge_mining_net
    from guaguanet import joint_pathnet_ncl as mining

if 'guagua' in args.approach:
    from guaguanet import w2v_hat as knowledge_mining_net
    from guaguanet import joint_hat_ncl as mining

if 'kim' in args.approach:
    from guaguanet import w2v_kim as knowledge_mining_net
    from guaguanet import joint_bert_rnn_ncl as mining

if 'trans' in args.approach:
    from guaguanet import bert_trans_ncl as retaining


########################################################################################################################

# Load

print('Load data...')
data, taskcla = dataloader.get(logger=logger, args=args)

print('\nTask info =', taskcla)

# Inits
print('Inits...')
knowledge_retain = knowledge_retain_net.Net(taskcla, args=args).cuda()
knowledge_mining = knowledge_mining_net.Net(taskcla, args=args).cuda()
retain_appr = retaining.Appr(knowledge_retain, logger=logger, args=args, nepochs=args.nepoch, lr=0.05)
mmining_appr = mining.Appr(knowledge_mining, logger=logger, args=args)

# Loop tasks
acc = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)
lss = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)
f1 = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)

for t, ncla in taskcla:
    print('*' * 100)
    print('Task {:2d} ({:s})'.format(t, data[t]['name']))
    print('*' * 100)

    # if t>1: exit()

    if 'mtl' in args.approach:
        # Get data. We do not put it to GPU
        if t == 0:
            train = data[t]['train']
            valid = data[t]['valid']
            num_train_steps = data[t]['num_train_steps']

        else:
            train = ConcatDataset([train, data[t]['train']])
            valid = ConcatDataset([valid, data[t]['valid']])
            num_train_steps += data[t]['num_train_steps']
        task = t

        if t < len(taskcla) - 1: continue  # only want the last one

    else:
        # Get data
        train = data[t]['train']
        valid = data[t]['valid']
        num_train_steps = data[t]['num_train_steps']
        task = t

    train_sampler = RandomSampler(train)
    train_dataloader = DataLoader(train, sampler=train_sampler, batch_size=args.train_batch_size)
    valid_sampler = SequentialSampler(valid)
    valid_dataloader = DataLoader(valid, sampler=valid_sampler, batch_size=args.eval_batch_size)

    # 理论上 这里应该是先train transformer_block, 再train textcnn_hat
    print('training retained_model')
    retained_model = retain_appr.train(task, train_dataloader, valid_dataloader, args)
    print('training mining_model')
    mmining_appr.train(retained_model, task, train_dataloader, valid_dataloader, args)

    print('-' * 100)
    #

    # Test
    for u in range(t + 1):
        test = data[u]['test']
        test_sampler = SequentialSampler(test)
        test_dataloader = DataLoader(test, sampler=test_sampler, batch_size=args.eval_batch_size)

        # 测试方法在这里...

        test_loss, test_acc, test_f1 = mmining_appr.eval(retained_model, u, test_dataloader)
        print('>>> Test on task {:2d} - {:15s}: loss={:.3f}, acc={:5.1f}% <<<'.format(u, data[u]['name'], test_loss,
                                                                                      100 * test_acc))
        acc[t, u] = test_acc
        lss[t, u] = test_loss
        f1[t, u] = test_f1

    # Save
    print('Save at ' + args.output)
    np.savetxt(args.output, acc, '%.4f', delimiter='\t')
    np.savetxt(args.output + '_f1', f1, '%.4f', delimiter='\t')

    # appr.decode(train_dataloader)
    # break

# Done
print('*' * 100)
print('Accuracies =')
for i in range(acc.shape[0]):
    print('\t', end='')
    for j in range(acc.shape[1]):
        print('{:5.1f}% '.format(100 * acc[i, j]), end='')
    print()
print('*' * 100)
print('Done!')

print('[Elapsed time = {:.1f} h]'.format((time.time() - tstart) / (60 * 60)))

with open(performance_output, 'w') as file:
    if 'ncl' in args.approach or 'mtl' in args.approach:
        for j in range(acc.shape[1]):
            file.writelines(str(acc[-1][j]) + '\n')

    elif 'one' in args.approach:
        for j in range(acc.shape[1]):
            file.writelines(str(acc[j][j]) + '\n')

with open(performance_output_forward, 'w') as file:
    if 'ncl' in args.approach or 'mtl' in args.approach:
        for j in range(acc.shape[1]):
            file.writelines(str(acc[j][j]) + '\n')

########################################################################################################################
