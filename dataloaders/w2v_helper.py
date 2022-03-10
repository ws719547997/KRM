import tqdm
import operator
import time
import pickle
import string
import numpy as np
from nltk.tokenize.treebank import TreebankWordTokenizer
'''
这个程序根据语料和预训练词向量，生成一个小的词表和对应的词向量。
这个是针对英文的。
'''

def check_coverage(vocab, embeddings_index):
    a = {}
    oov = {}
    k = 0
    i = 0
    for word in vocab:
        try:
            a[word] = embeddings_index[word]
            k += vocab[word]
        except:

            oov[word] = vocab[word]
            i += vocab[word]
            pass

    print('Found embeddings for {:.2%} of vocab'.format(len(a) / len(vocab)))
    print('Found embeddings for  {:.2%} of all text'.format(k / (k + i)))
    sorted_x = sorted(oov.items(), key=operator.itemgetter(1))[::-1]

    return sorted_x


def build_vocab(sentences,vocab):
    """
    :param sentences: list of list of words
    :return: dictionary of words and their count
    """
    for sentence in sentences:
        for word in sentence:
            try:
                vocab[word] += 1
            except KeyError:
                vocab[word] = 1
    return vocab


def load_dataset(path):
    tokenizer = TreebankWordTokenizer() # 处理 't 's的情况 减少OOV

    # def handle_contractions(x):
    #     x = tokenizer.tokenize(x)
    #     x = ' '.join(x)
    #     return x

    punct = "/-?!.,#$%()*+-/:;<=>@[\\]^_`{|}~`" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'

    def clean_special_chars(text):
        for p in punct:
            text = text.replace(p, ' ')
        return text

    contents = []
    with open(path, 'r', encoding='UTF-8') as f:
        for line in f:
            lin = line.strip()
            if not lin:
                continue
            if len(lin.split('\t')) < 5:
                continue
            lin_sp = lin.split('\t')
            content = clean_special_chars(lin_sp[4])
            # token = handle_contractions(content).split()
            token = tokenizer.tokenize(content)

            # word to id
            contents.append(token)
        return contents


min_freq =1
max_size = 200000
UNK, PAD = '<UNK>', '<PAD>'
emb_dim = 300

#### 1. start here loading GLOVE 840b, 2123134 word vectors in 3 mins.
tic = time.time()
print('Loading word vectors')
GLOVE_EMBEDDING_PATH = '/Users/kuroneko/Downloads/glove.840B.300d.txt'
embeddings_idx = {}
f = open(GLOVE_EMBEDDING_PATH)
for line in f:
    values = line.split(' ')
    word = values[0]
    embedding = np.asarray(values[1:], dtype='float32')
    embeddings_idx[word] = embedding
f.close()
print(f'loaded {len(embeddings_idx)} word vectors in {time.time() - tic}s')


#### 2.read text from and preprocess

# dom_list = ['Automotive_5', 'Electronics_5', 'Industrial_and_Scientific_5', 'Kindle_Store_5',
#             'Cell_Phones_and_Accessories_5', 'Musical_Instruments_5', 'Office_Products_5', 'Patio_Lawn_and_Garden_5',
#             'Sports_and_Outdoors_5', 'Luxury_Beauty_5', 'Grocery_and_Gourmet_Food_5', 'Digital_Music_5',
#             'Tools_and_Home_Improvement_5', 'Pet_Supplies_5', 'Prime_Pantry_5', 'Toys_and_Games_5', 'Movies_and_TV_5',
#             'Home_and_Kitchen_5', 'Arts_Crafts_and_Sewing_5', 'Video_Games_5', 'CDs_and_Vinyl_5']

dom_list = ['Sandal', 'Magazine_Subscriptions', 'RiceCooker', 'Flashlight', 'Jewelry', 'CableModem', 'GraphicsCard',
            'GPS', 'Projector', 'Keyboard', 'Video_Games', 'AlarmClock', 'HomeTheaterSystem', 'Vacuum', 'Gloves',
            'Baby', 'Bag', 'Movies_TV', 'Dumbbell', 'Headphone']

vocabs = {}

# in SNAP we have 21 domains while AMZ has 20.
for index in range(20):
    vocabs = build_vocab(load_dataset('../AMZ1K/data/train/' + dom_list[index] + '.txt'),vocabs)
    vocabs = build_vocab(load_dataset('../AMZ1K/data/dev/' + dom_list[index] + '.txt'),vocabs)
    vocabs = build_vocab(load_dataset('../AMZ1K/data/test/' + dom_list[index] + '.txt'),vocabs)
print('finish loading vocabs!')


####3. see what happen...
oov = check_coverage(vocabs,embeddings_idx)
print(oov[:50])

####4. bulid small size of vocabs and embedings.
vocab_list = sorted([_ for _ in vocabs.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_size]
vocabs = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
vocabs.update({UNK: len(vocabs), PAD: len(vocabs) + 1})
pickle.dump(vocabs, open('../embeds/AMZ/vocab.pkl', 'wb'))

embeddings = np.random.rand(len(vocabs), emb_dim)
for key in vocabs:
    if key in embeddings_idx.keys():
        embeddings[vocabs[key]] = embeddings_idx[key]
np.savez_compressed('../embeds/AMZ/embedding', embeddings=embeddings)
print('saved!')