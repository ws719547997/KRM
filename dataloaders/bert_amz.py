#Coding: UTF-8
"""
作者 王松
时间 2021-05月
从英文20领域1000语料库中读取数据再转化为bert的输入格式
返回BERTtoken和label token中有 input_ids segmengt_musk attention_musk
"""
import numpy as np
import math
import torch
from torch.utils.data import TensorDataset
from transformers import BertTokenizer

dom_list = ['Sandal', 'Magazine_Subscriptions', 'RiceCooker', 'Flashlight', 'Jewelry', 'CableModem', 'GraphicsCard',
            'GPS', 'Projector', 'Keyboard', 'Video_Games', 'AlarmClock', 'HomeTheaterSystem', 'Vacuum', 'Gloves',
            'Baby', 'Bag', 'Movies_TV', 'Dumbbell', 'Headphone']



def padding_list(list, args=None):
    """
    把每个list都补长到指定维数
    """
    while len(list) < args.max_seq_length:
        list.append(0)
    return list

def build_dataset_LL(index,args):
    # 这里用的是bert 先用huggingface的transformers拿到各种index。
    tokenizer = BertTokenizer.from_pretrained(args.bert_model)  # huggingface的transformers

    def load_dataset(path, pad_size=32):
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in f:
                lin = line.strip()
                if not lin: # 如果为空就跳过
                    continue
                if len(lin.split('\t')) < 5: # 每条数据应该包含5个特征 因此如果有数据不全的话就跳过
                    continue
                lin_sp = lin.split('\t')
                content = lin_sp[4]
                if lin_sp[2] == 'NEU': # 部分分类中存在中性这个类别，但是在我们二分类的情况下，这个类别就取消了。
                    continue
                label = 0 if lin_sp[2] == 'NEG' else 1 # neg == 0 pos == 1
                token = tokenizer(content,truncation='longest_first',max_length=args.max_seq_length)
                for key in token: # 把长度补全到max_length
                    token[key] = padding_list(token[key],args)

                contents.append((token, label)) # 返回BERTtoken和label token中有 input_ids segmengt_musk attention_musk
        return contents
    _train = []
    _dev = []
    _train = load_dataset('AMZ1k/data/train/' + dom_list[index] + '.txt', args.max_seq_length)
    _dev = load_dataset('AMZ1k/data/dev/' + dom_list[index] + '.txt', args.max_seq_length)

    # only test on target domain
    _test = load_dataset('AMZ1k/data/test/' + dom_list[index] + '.txt', args.max_seq_length)

    return _train, _dev, _test


def get(logger=None,args=None):
    data={}
    taskcla=[]
    # 调整list的顺序从而设定最先开始学习的领域
    for i in range(args.startpoint):
        dom_list.append(dom_list.pop(0))

    for domain_id in range(args.ntasks):
        data[domain_id] = {}
        domain_name = dom_list[domain_id]
        data[domain_id]['name']=domain_name
        data[domain_id]['ncla']=2

        # 获得了数据
        train_data, dev_data, test_data = build_dataset_LL(domain_id,args)
        num_train_steps = int(math.ceil(len(train_data) / args.train_batch_size)) * args.num_train_epochs

        # 标签转为tensor
        train_label = torch.tensor([f[1] for f in train_data], dtype=torch.long)
        dev_label = torch.tensor([f[1] for f in dev_data], dtype=torch.long)
        test_label = torch.tensor([f[1] for f in test_data], dtype=torch.long)

        # 准备 train、Dev、test
        train_input_ids = torch.tensor([f[0].input_ids for f in train_data],dtype=torch.long)
        train_token_type_ids = torch.tensor([f[0].token_type_ids for f in train_data],dtype=torch.long)
        train_attention_mask = torch.tensor([f[0].attention_mask for f in train_data], dtype=torch.long)
        data[domain_id]['train'] = TensorDataset(train_input_ids, train_token_type_ids,train_attention_mask,train_label)

        dev_input_ids = torch.tensor([f[0].input_ids for f in dev_data], dtype=torch.long)
        dev_token_type_ids = torch.tensor([f[0].token_type_ids for f in dev_data], dtype=torch.long)
        dev_attention_mask = torch.tensor([f[0].attention_mask for f in dev_data], dtype=torch.long)
        data[domain_id]['valid'] = TensorDataset(dev_input_ids, dev_token_type_ids, dev_attention_mask,
                                                 dev_label)

        test_input_ids = torch.tensor([f[0].input_ids for f in test_data], dtype=torch.long)
        test_token_type_ids = torch.tensor([f[0].token_type_ids for f in test_data], dtype=torch.long)
        test_attention_mask = torch.tensor([f[0].attention_mask for f in test_data], dtype=torch.long)
        data[domain_id]['test'] = TensorDataset(test_input_ids, test_token_type_ids, test_attention_mask,
                                                 test_label)

        data[domain_id]['num_train_steps'] = num_train_steps
        taskcla.append((domain_id, int(data[domain_id]['ncla'])))  # [(domain,ncla),(),]

    return data,taskcla


