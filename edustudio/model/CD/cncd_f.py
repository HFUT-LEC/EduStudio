r"""
CNCD-F
##########################################

Reference:
    Fei Wang et al. "Neural Cognitive Diagnosis for Intelligent Education Systems" in AAAI 2020.

Reference Code:
    https://github.com/bigdata-ustc/Neural_Cognitive_Diagnosis-NeuralCD

"""
from ..gd_basemodel import GDBaseModel
import torch.nn as nn
from ..utils.components import PosMLP
import torch
import torch.nn.functional as F
import numpy as np


class TextCNN(nn.Module):
    def __init__(self, batch_size, knowledge_n, embedding_dim):
        self.batch_size = batch_size
        self.embedding_len = embedding_dim
        self.sequence_len = 600
        self.output_len = knowledge_n
        self.channel_num1, self.channel_num2, self.channel_num3 = 400, 200, 100
        self.kernel_size1, self.kernel_size2, self.kernel_size3 = 3, 4, 5
        self.pool1 = 3
        self.full_in = (self.sequence_len + self.kernel_size1 - 1) // self.pool1 + self.kernel_size2 + self.kernel_size3 - 2
        super(TextCNN, self).__init__()

        self.conv1 = nn.Conv1d(self.embedding_len, self.channel_num1, kernel_size=self.kernel_size1, padding=self.kernel_size1-1, stride=1)
        self.conv2 = nn.Conv1d(self.channel_num1, self.channel_num2, kernel_size=self.kernel_size2, padding=self.kernel_size2-1, stride=1)
        self.conv3 = nn.Conv1d(self.channel_num2, self.channel_num3, kernel_size=self.kernel_size3, padding=self.kernel_size3 - 1, stride=1)
        self.full = nn.Linear(self.full_in, embedding_dim)

    def prepare_embedding(self, content_list):
        from gensim.models import Word2Vec
        from bintrees import FastRBTree
        words_all = []
        for v in content_list:
            ls = [str(i) for i in v.split(',')]
            words_all.extend(ls)
        word2vec_model = Word2Vec(words_all)
        words, self.word2id = FastRBTree(), FastRBTree()
        word_npy = []
        for exer in words_all:
            for word in exer:
                words.insert(word, True)
        word_count = 0
        word_npy.append([0.] * 100)  # index=0 is a zero-vector
        for word in words:
            if word in word2vec_model.wv:
                word_count += 1
                self.word2id.insert(word, word_count)
                word_npy.append(word2vec_model.wv[word])
            else:
                print('not found: ' + str(word))
        self.word_npy = np.array(word_npy)
        self.word_emb = nn.Embedding(len(self.word2id) + 1, self.embedding_len, padding_idx=0)
        self.word_emb.weight.data.copy_(torch.from_numpy(self.word_npy))
        return self.word2id

    def forward(self, x):
        x = self.word_emb(x)
        x = torch.transpose(x, 1, 2)
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, kernel_size=self.pool1)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.transpose(x, 1, 2)
        x = F.max_pool1d(x, self.channel_num3)
        x = torch.transpose(x, 1, 2).view(-1, self.full_in)
        ret = self.full(x)     # 使用的损失函数包含sigmoid，在预测时需在网络外加sigmoid
        return ret

class CNCD_F(GDBaseModel):
    r"""
    CNCD-F

    default_cfg:
       'dnn_units': [512, 256]  # dimension list of hidden layer in prediction layer
       'dropout_rate': 0.5      # dropout rate
       'activation': 'sigmoid'  # activation function in prediction layer
       'disc_scale': 10         # discrimination scale
       'max_len': 600,          # the maximum length of the exercise text
        'text_embedding_dim': 100 # dimension of text embedding
    """
    default_cfg = {
        'dnn_units': [512, 256],
        'dropout_rate': 0.5,
        'activation': 'sigmoid',
        'disc_scale': 10,
        'max_len': 600,
        'text_embedding_dim': 100
    }
    def __init__(self, cfg):
        super().__init__(cfg)

    def build_cfg(self):
        self.n_user = self.datatpl_cfg['dt_info']['stu_count']
        self.n_item = self.datatpl_cfg['dt_info']['exer_count']
        self.n_cpt = self.datatpl_cfg['dt_info']['cpt_count']
        self.text_dim = self.modeltpl_cfg["text_embedding_dim"]
        self.max_len = self.modeltpl_cfg["max_len"]

    def get_from_text(self):
        x = []
        for e in self.content_list:
            word_ids = []
            for word in e[:self.max_len]:  # 固定长度为self.max_len
                word_id = self.word2id.get(word)
                if word_id is not None:
                    word_ids.append(word_id)
            if len(word_ids) < self.max_len:
                word_ids += [0] * (self.max_len - len(word_ids))  # padding到self.max_len
            x.append(word_ids)
        return torch.tensor(x, device=self.traintpl_cfg['device'])

    def build_model(self):
        # prediction sub-net
        self.textcnn = TextCNN(self.traintpl_cfg["batch_size"], self.n_cpt, self.text_dim)
        self.word2id = self.textcnn.prepare_embedding(self.content_list)
        self.word_ids = self.get_from_text()
        self.out_text_factor = nn.Linear(self.text_dim, 1)
        self.student_emb = nn.Embedding(self.n_user, self.n_cpt+1)
        self.k_difficulty = nn.Embedding(self.n_item, self.n_cpt)
        self.e_difficulty = nn.Embedding(self.n_item, 1)
        self.pd_net = PosMLP(
            input_dim=self.n_cpt+1, output_dim=1, activation=self.modeltpl_cfg['activation'],
            dnn_units=self.modeltpl_cfg['dnn_units'], dropout_rate=self.modeltpl_cfg['dropout_rate']
        )

    def add_extra_data(self, **kwargs):
        cl = kwargs['content']
        self.content_list=[]
        for ls in cl:
            s = ",".join(map(str, ls))
            self.content_list.append(s)
        self.Q_mat = kwargs['Q_mat']

    def forward(self, stu_id, exer_id):
        # before prednet
        items_Q_mat = self.Q_mat[exer_id].to(self.traintpl_cfg['device'])
        items_content = self.word_ids[exer_id]
        text_embedding = self.textcnn(items_content)
        text_factor = torch.sigmoid(self.out_text_factor(text_embedding))
        stu_emb = self.student_emb(stu_id)
        stat_emb = torch.sigmoid(stu_emb)
        k_difficulty = torch.sigmoid(self.k_difficulty(exer_id))
        k_difficulty = torch.cat((k_difficulty, text_factor), dim=1)
        e_difficulty = torch.sigmoid(self.e_difficulty(exer_id)) * self.modeltpl_cfg['disc_scale']
        # prednet
        text_factor_q = torch.ones((items_Q_mat.shape[0], 1), device=self.traintpl_cfg['device'])
        input_knowledge_point = torch.cat((items_Q_mat, text_factor_q), dim=1)
        input_x = e_difficulty * (stat_emb - k_difficulty) * input_knowledge_point
        pd = self.pd_net(input_x).sigmoid()
        return pd

    @torch.no_grad()
    def predict(self, stu_id, exer_id, **kwargs):
        return {
            'y_pd': self(stu_id, exer_id).flatten(),
        }

    def get_main_loss(self, **kwargs):
        stu_id = kwargs['stu_id']
        exer_id = kwargs['exer_id']
        label = kwargs['label']
        pd = self(stu_id, exer_id).flatten()
        loss = F.binary_cross_entropy(input=pd, target=label)
        return {
            'loss_main': loss
        }

    def get_loss_dict(self, **kwargs):
        return self.get_main_loss(**kwargs)
    
    def get_stu_status(self, stu_id=None):
        if stu_id is not None:
            return self.student_emb(stu_id)
        else:
            return self.student_emb.weight
