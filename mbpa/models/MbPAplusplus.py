import torch
import torch.nn as nn
import transformers
import numpy as np
from tqdm import trange
import copy
from mbpa.models.baselines.MbPA import MbPA
import random
# import pdb
'''
这份代码经过了较大的改动：
1.通过 --trainmode 设置训练模式：train：训练模型 0,1,...,ntasks：：在第几个任务上做测试。这么做的原因是因为显卡显存有限。
2.修改了dataloader。原论文代码中每个任务的数据没有明显的标识，混杂在一起。本代码使用TCL的模式，显式区分每个任务，依次学习。
3.在原代码的 infer 过程中，对每一个测试样本都重新训练网络分类器，太他妈慢了！我改成每隔n次进行一次调整(args.inter)，其余直接用调整过的分类器直接分类。
4.事先设置一下model的存储路径和结果的存储路径。

'''
class ReplayMemory(object):
    """
        Create the empty memory buffer
    """

    def __init__(self, buffer=None):

        if buffer is None:
            self.memory = {}
        else:
            self.memory = buffer
            # convert the keys from np.bytes to np.float32
            self.all_keys = np.frombuffer(
                np.asarray(list(self.memory.keys())), dtype=np.float32).reshape(-1, 768)

    def push(self, keys, examples):
        """
        Add the examples as key-value pairs to the memory dictionary with content,attention_mask,label tuple as value
        and key determined by key network
        """
        contents, attn_masks, labels = examples
        # update the memory dictionary
        for i, key in enumerate(keys):
            # numpy array cannot be used as key since it is non-hashable, hence convert it to bytes to use as key
            self.memory.update(
                {key.tobytes(): (contents[i], attn_masks[i], labels[i])})
        self.all_keys = np.frombuffer(
                np.asarray(list(self.memory.keys())), dtype=np.float32).reshape(-1, 768)

    def _prepare_batch(self, sample):
        """
        Parameter:
        sample -> list of tuple of experiences
               -> i.e, [(content_1,attn_mask_1,label_1),.....,(content_k,attn_mask_k,label_k)]
        Returns:
        batch -> tuple of list of content,attn_mask,label
              -> i.e, ([content_1,...,content_k],[attn_mask_1,...,attn_mask_k],[label_1,...,label_k])
        """
        contents = []
        attn_masks = []
        labels = []
        # Iterate over experiences
        for content, attn_mask, label in sample:
            # convert the batch elements into torch.LongTensor
            contents.append(content)
            attn_masks.append(attn_mask)
            labels.append(label)

        return (torch.LongTensor(contents), torch.LongTensor(attn_masks), torch.LongTensor(labels))

    def get_neighbours(self, keys, k=32):
        """
        Returns samples from buffer using nearest neighbour approach
        """
        samples = []
        # Iterate over all the input keys
        # to find neigbours for each of them
        for key in keys:
            # compute similarity scores based on Euclidean distance metric
            similarity_scores = np.dot(self.all_keys, key.T)
            K_neighbour_keys = self.all_keys[np.argpartition(
                similarity_scores, -k)[-k:]]
            neighbours = [self.memory[nkey.tobytes()]
                          for nkey in K_neighbour_keys]
            # converts experiences into batch
            batch = self._prepare_batch(neighbours)
            samples.append(batch)

        return samples
    
    def sample(self, sample_size):
        keys = random.sample(list(self.memory),sample_size)
        contents = np.array([self.memory[k][0] for k in keys])
        attn_masks = np.array([self.memory[k][1] for k in keys])
        labels = np.array([self.memory[k][2] for k in keys])
        return (torch.LongTensor(contents), torch.LongTensor(attn_masks), torch.LongTensor(labels))

    def update(self):
        self.all_keys = np.frombuffer(
            np.asarray(list(self.memory.keys())), dtype=np.float32).reshape(-1, 768)


class MbPAplusplus(nn.Module):
    """
    Implements Memory based Parameter Adaptation model
    """

    def __init__(self,args, model_state=None):
        super(MbPAplusplus, self).__init__()

        self.adp_classifier = None
        if model_state is None:
            # Key network to find key representation of content
            self.key_encoder = transformers.BertModel.from_pretrained(args.bert_model)
            # Bert model for text classification
            self.classifier = transformers.BertForSequenceClassification.from_pretrained(
                args.bert_model, num_labels=2)

        else:

            cls_config = transformers.BertConfig.from_pretrained(args.bert_model, num_labels=2)
            self.classifier = transformers.BertForSequenceClassification(cls_config)
            self.classifier.load_state_dict(model_state['classifier'])
            key_config = transformers.BertConfig.from_pretrained(
                args.bert_model)
            self.key_encoder = transformers.BertModel(key_config)
            self.key_encoder.load_state_dict(model_state['key_encoder'])
            # load base model weights
            # we need to detach since parameters() method returns reference to the original parameters
            self.base_weights = list(self.classifier.parameters()).copy()
        # local adaptation learning rate - 1e-3 or 5e-3
        self.loc_adapt_lr = 1e-3
        # Number of local adaptation steps
        self.L = args.mbpa

    def classify(self, content, attention_mask, labels):
        """
        Bert classification model
        """
        loss, logits = self.classifier(
            content, attention_mask=attention_mask, labels=labels)
        return loss, logits

    def get_keys(self, contents, attn_masks):
        """
        Return key representation of the documents
        """
        # Freeze the weights of the key network to prevent key
        # representations from drifting as data distribution changes
        with torch.no_grad():
            last_hidden_states, _ = self.key_encoder(
                contents, attention_mask=attn_masks)
        # Obtain key representation of every text content by selecting the its [CLS] hidden representation
        keys = last_hidden_states[:, 0, :]

        return keys

    def infer(self, content, attn_mask, K_contents, K_attn_masks, K_labels):
        """
        Function that performs inference based on memory based local adaptation
        Parameters:
        content   -> document that needs to be classified
        attn_mask -> attention mask over document
        rt_batch  -> the batch of samples retrieved from the memory using nearest neighbour approach

        Returns:
        logit -> label corresponding to the single document provided,i.e, content
        """

        # create a local copy of the classifier network
        if self.adp_classifier == None:
            self.adp_classifier = copy.deepcopy(self.classifier)
        optimizer = transformers.AdamW(
            self.adp_classifier.parameters(), lr=self.loc_adapt_lr)

        # Current model weights
        curr_weights = list(self.adp_classifier.parameters())
        # Train the adaptive classifier for L epochs with the rt_batch
        for _ in range(self.L):

            # zero out the gradients
            optimizer.zero_grad()
            likelihood_loss, _ = self.adp_classifier(
                K_contents, attention_mask=K_attn_masks, labels=K_labels)
            # Initialize diff_loss to zero and place it on the appropriate device
            diff_loss = torch.Tensor([0]).to(
                "cuda" if torch.cuda.is_available() else "cpu")
            # Iterate over base_weights and curr_weights and accumulate the euclidean norm
            # of their differences
            for base_param, curr_param in zip(self.base_weights, curr_weights):
                diff_loss += (curr_param-base_param).pow(2).sum()

            # Total loss due to log likelihood and weight restraint
            total_loss = 0.001*diff_loss + likelihood_loss
            total_loss.backward()
            optimizer.step()

        logits, = self.adp_classifier(content.unsqueeze(
            0), attention_mask=attn_mask.unsqueeze(0))
        # Note: to prevent keeping track of intermediate values which
        # can lead to cuda of memory runtime error logits should be detached
        return logits.detach()

    def test_no_kmeans(self, content, attn_mask):
        """
        妈卖批 加了k-means的infer慢成傻逼了好吗!
        """

        logits, = self.adp_classifier(content.unsqueeze(
            0), attention_mask=attn_mask.unsqueeze(0))
        # Note: to prevent keeping track of intermediate values which
        # can lead to cuda of memory runtime error logits should be detached
        return logits.detach()

    def save_state(self):
        """
        Returns model state
        """
        model_state = dict()
        model_state['classifier'] = self.classifier.state_dict()
        model_state['key_encoder'] = self.key_encoder.state_dict()

        return model_state
