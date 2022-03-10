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
    args.output = './res/' + args.experiment + '_' + args.approach + '_' + str(args.startpoint) + '_' + str(
        args.nepoch) + '.txt'
performance_output = args.output + '_performance'
performance_output_forward = args.output + '_forward_performance'

# Seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available(): torch.cuda.manual_seed(args.seed)
else:
    print('[CUDA unavailable]')
    exit()

# Args -- Experiment
if 'w2v' in args.experiment:
    args.bert_hidden_size = 300
elif 'bert' in args.experiment:
    args.bert_hidden_size = 768

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

# Args -- Approach
if args.approach == 'bert_lstm_ncl' or \
        args.approach == 'bert_gru_ncl' or \
        args.approach == 'bert_cnn_ncl' or\
        args.approach == 'bert_mlpmix_ncl' or\
        args.approach == 'bert_only_ncl':
    from approaches import bert_rnn_ncl as approach

elif args.approach == 'bert_lstm_kan_ncl' or args.approach == 'bert_gru_kan_ncl':
    from approaches import bert_rnn_kan_ncl as approach
    from networks import bert_lstm_kan as network

elif args.approach == 'bert_lstm_ewc_ncl' or args.approach == 'bert_gru_ewc_ncl':
    from approaches import bert_rnn_ewc_ncl as approach

elif args.approach == 'bert_cnn_ewc_ncl':
    from approaches import bert_cnn_ewc_ncl as approach
    from networks import bert_kim as network

elif args.approach == 'bert_lstm_lwf_ncl' or args.approach == 'bert_gru_lwf_ncl' or args.approach == 'bert_cnn_lwf_ncl':
    from approaches import bert_rnn_lwf_ncl as approach

elif args.approach == 'bert_lstm_lfl_ncl' or args.approach == 'bert_gru_lfl_ncl' or args.approach == 'bert_cnn_lfl_ncl' :
    from approaches import bert_rnn_lfl_ncl as approach

elif args.approach == 'bert_lstm_imm_mean_ncl' or args.approach == 'bert_gru_imm_mean_ncl' or args.approach == 'bert_cnn_imm_mean_ncl':
    from approaches import bert_rnn_imm_mean_ncl as approach
elif args.approach == 'bert_lstm_imm_mode_ncl' or args.approach == 'bert_gru_imm_mode_ncl' or args.approach == 'bert_cnn_imm_mode_ncl':
    from approaches import bert_rnn_imm_mode_ncl as approach
if args.approach == 'bert_mlp_hat_ncl':
    from approaches import bert_mlp_hat_ncl as approach
if args.approach == 'bert_mlp_pnn_ncl':
    from approaches import bert_mlp_pnn_ncl as approach
if args.approach == 'bert_mlp_pathnet_ncl':
    from approaches import bert_mlp_pathnet_ncl as approach


# # Args -- Network
if args.approach == 'bert_only_ncl':
    from networks import bert_only as network
elif 'bert_lstm_ewc' in args.approach:
    from networks import bert_lstm as network
elif 'bert_lstm_lwf' in args.approach:
    from networks import bert_lstm as network
elif 'bert_lstm_lfl' in args.approach:
    from networks import bert_lstm_lfl as network
elif 'bert_lstm' in args.approach:
    from networks import bert_lstm as network
if "bert_cnn_ncl" == args.approach:
    from networks import bert_kim as network
if 'bert_cnn_lfl' in args.approach:
    from networks import bert_textcnn_lfl as network

if 'bert_gru_kan' in args.approach:
    from networks import bert_gru_kan as network


if args.approach == 'bert_mlp_hat_ncl':
    from networks import bert_hat as network
if args.approach == 'bert_mlp_pnn_ncl':
    from networks import bert_pnn as network
if args.approach == 'bert_mlp_pathnet_ncl':
    from networks import bert_pathnet as network

if args.approach == 'bert_mlpmix_ncl':
    from networks import bert_mlpmix as network

####### new ######
if args.approach == 'bert_gru_srk_ncl':
    from approaches import bert_rnn_srk_ncl as approach
    from networks import bert_gru_srk as network

if args.approach == 'bert_cnn_hat_ncl':
    from approaches import bert_cnn_hat_ncl as approach
    from networks import bert_kim_hat as network

if args.approach == 'bert_cnn_owm_ncl':
    from approaches import bert_cnn_owm_ncl as approach
    from networks import bert_kim_owm as network

if args.approach == 'bert_cnn_ucl_ncl':
    from approaches import bert_cnn_ucl_ncl as approach
    from networks import bert_kim_ucl as network

if args.approach == 'bert_cnn_a-gem_ncl':
    from approaches import bert_cnn_agem_ncl as approach
    from networks import bert_kim as network

if args.approach == 'bert_cnn_derpp_ncl':
    from approaches import bert_cnn_derpp_ncl as approach
    from networks import bert_kim as network

if args.approach == 'bert_trans_cse':
    from krm import bert_trans_cse as approach
    from krm import bert_trans_cse as network

########################################################################################################################

# Load

print('Load data...')
data, taskcla = dataloader.get(logger=logger, args=args)

print('\nTask info =', taskcla)

# Inits
print('Inits...')
net = network.Net(taskcla, args=args).cuda()
# net=network.Net(taskcla,args=args)


appr = approach.Appr(net, logger=logger, taskcla=taskcla,args=args )

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

    if args.approach == 'bert_cnn_ewc_ncl' or args.approach =='bert_cnn_a-gem_ncl' or args.approach =='bert_cnn_derpp_ncl':
        appr.train(task, train_dataloader, valid_dataloader, args,train_data=train)
    else:
        appr.train(task, train_dataloader, valid_dataloader, args)
    print('-' * 100)
    #

    # Test
    for u in range(t + 1):
        test = data[u]['test']
        test_sampler = SequentialSampler(test)
        test_dataloader = DataLoader(test, sampler=test_sampler, batch_size=args.eval_batch_size)

        if 'kan' in args.approach:
            test_loss, test_acc, test_f1 = appr.eval(u, test_dataloader, 'mcl')
        else:
            test_loss, test_acc, test_f1 = appr.eval(u, test_dataloader)
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
