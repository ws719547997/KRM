#Coding: UTF-8
"""
作者 王松
时间 2021-06-04
从snap（29）中抽取2个领域10k数据 语料库中读取数据再转化为bert的输入格式
"""
import numpy as np
import math
import torch
from torch.utils.data import TensorDataset
from transformers import BertTokenizer

dom_list = ['Automotive_5', 'Electronics_5', 'Industrial_and_Scientific_5', 'Kindle_Store_5', 'Cell_Phones_and_Accessories_5', 'Musical_Instruments_5', 'Office_Products_5', 'Patio_Lawn_and_Garden_5', 'Sports_and_Outdoors_5', 'Luxury_Beauty_5', 'Grocery_and_Gourmet_Food_5', 'Digital_Music_5', 'Tools_and_Home_Improvement_5', 'Pet_Supplies_5', 'Prime_Pantry_5', 'Toys_and_Games_5', 'Movies_and_TV_5', 'Home_and_Kitchen_5', 'Arts_Crafts_and_Sewing_5', 'Video_Games_5', 'CDs_and_Vinyl_5']

def padding_list(list, args=None):
    """
    把每个list都补长到指定维数
    """
    while len(list) < args.max_seq_length:
        list.append(0)
    return list

def build_dataset_LL(index,args):
    tokenizer = BertTokenizer.from_pretrained(args.bert_model)  # huggingface的transformers

    def load_dataset(path, pad_size=32):
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in f:
                lin = line.strip()
                if not lin:
                    continue
                if len(lin.split('\t')) < 5:
                    continue
                lin_sp = lin.split('\t')
                content = lin_sp[4]
                if lin_sp[2] == 'NEU':
                    continue
                label = 0 if lin_sp[2] == 'NEG' else 1
                token = tokenizer(content,truncation='longest_first',max_length=args.max_seq_length)
                for key in token:
                    token[key] = padding_list(token[key],args)
                contents.append((token, label)) # 返回BERTtoken和label token中有 input_ids segmengt_musk attention_musk
        return contents
    _train = []
    _dev = []
    _train = load_dataset('snap10k/data/train/' + dom_list[index] + '.txt', args.max_seq_length)
    _dev = load_dataset('snap10k/data/dev/' + dom_list[index] + '.txt', args.max_seq_length)

    # only test on target domain
    _test = load_dataset('snap10k/data/test/' + dom_list[index] + '.txt', args.max_seq_length)

    return _train, _dev, _test



def get(logger=None,args=None):
    data={}
    taskcla=[]
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


