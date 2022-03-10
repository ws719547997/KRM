#Coding: UTF-8
"""
author：Wang Song
time: 2021-05-17 mid-night
"""
import numpy as np
import pickle as pkl
import math
import torch
from torch.utils.data import TensorDataset
from nltk.tokenize.treebank import TreebankWordTokenizer

MAX_VOCAB_SIZE = 80000  # 词表长度限制
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号
dom_list = ['Automotive_5', 'Electronics_5', 'Industrial_and_Scientific_5', 'Kindle_Store_5', 'Cell_Phones_and_Accessories_5', 'Musical_Instruments_5', 'Office_Products_5', 'Patio_Lawn_and_Garden_5', 'Sports_and_Outdoors_5', 'Luxury_Beauty_5', 'Grocery_and_Gourmet_Food_5', 'Digital_Music_5', 'Tools_and_Home_Improvement_5', 'Pet_Supplies_5', 'Prime_Pantry_5', 'Toys_and_Games_5', 'Movies_and_TV_5', 'Home_and_Kitchen_5', 'Arts_Crafts_and_Sewing_5', 'Video_Games_5', 'CDs_and_Vinyl_5']


def build_dataset_LL(index,args):
    tokenizer = lambda x: x.split(' ')  # 以空格隔开，word-level
    vocab = pkl.load(open('embeds/SNAP/vocab.pkl', 'rb'))

    tokenizer = TreebankWordTokenizer()

    def handle_contractions(x):
        x = tokenizer.tokenize(x)
        x = ' '.join(x)
        return x

    punct = "/-?!.,#$%()*+-/:;<=>@[\\]^_`{|}~`" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'

    def clean_special_chars(text):
        for p in punct:
            text = text.replace(p, ' ')
        return text

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
                if lin_sp[2] == 'NEG':
                    label = 0
                elif lin_sp[2] == 'POS':
                    label = 1
                words_line = []
                content = clean_special_chars(lin_sp[4])
                token = handle_contractions(content).split()
                if pad_size:
                    if len(token) < pad_size:
                        token.extend([PAD] * (pad_size - len(token)))
                    else:
                        token = token[:pad_size]
                # word to id
                for word in token:
                    words_line.append(vocab.get(word, vocab.get(UNK)))

                contents.append((words_line, label))
        return contents  # [([...], 0), ([...], 1), ...] 词向量和标签
    _train = []
    _dev = []

    # only target data
    _train += load_dataset('snap10k/data/train/' + dom_list[index] + '.txt', args.max_seq_length)
    _dev += load_dataset('snap10k/data/dev/' + dom_list[index] + '.txt', args.max_seq_length)
    # only test on target domain
    _test = load_dataset('snap10k/data/test/' + dom_list[index] + '.txt', args.max_seq_length)

    return vocab, _train, _dev, _test

def get(logger=None,args=None):
    data={}
    taskcla=[]
    # data里包含了每个领域的数据
    #每个领域的数据又包括name、分类数，训练次数，划分好的数据集


    for i in range(args.startpoint):
        dom_list.append(dom_list.pop(0))

    for domain_id in range(args.ntasks):
        data[domain_id] = {}
        domain_name = dom_list[domain_id]
        data[domain_id]['name']=domain_name
        data[domain_id]['ncla']=2

        vocab, train_data, dev_data, test_data = build_dataset_LL(domain_id,args)
        num_train_steps = int(math.ceil(len(train_data) / args.train_batch_size)) * args.num_train_epochs

        train_token =  torch.tensor([f[0] for f in train_data], dtype=torch.long)
        dev_token =    torch.tensor([f[0] for f in dev_data],dtype=torch.long)
        test_token =   torch.tensor([f[0] for f in test_data],dtype=torch.long)

        train_label = torch.tensor([f[1] for f in train_data], dtype=torch.long)
        dev_label = torch.tensor([f[1] for f in dev_data], dtype=torch.long)
        test_label = torch.tensor([f[1] for f in test_data], dtype=torch.long)
        
        data[domain_id]['train'] = TensorDataset(train_token, train_label)
        data[domain_id]['valid'] = TensorDataset(dev_token, dev_label)
        data[domain_id]['test'] = TensorDataset(test_token, test_label)
        data[domain_id]['num_train_steps'] = num_train_steps
        taskcla.append((domain_id, int(data[domain_id]['ncla'])))
    return data,taskcla


