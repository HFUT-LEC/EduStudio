from .kt_inter_datafmt_extends_q import KTInterDataFmtExtendsQ
from itertools import chain
import torch
import pandas as pd
from ..utils.pad_seq_util import PadSeqUtil
from edustudio.utils.common import tensor2npy
import numpy as np
import copy

class EERNNDataFmt(KTInterDataFmtExtendsQ):
    default_cfg = {
        'word_emb_dim': 50,
        'max_sentence_len': 100,
    }
    
    def _init_data_after_dt_info(self):
        super()._init_data_after_dt_info()
        self.construct_word_emb()

    def construct_word_emb(self):
        import gensim
        exers = tensor2npy(self.train_dict['exer_seq'].unique())
        sentences = self.df_Q[self.df_Q['exer_id'].isin(exers)]['content'].tolist()
        desc_dict = gensim.corpora.Dictionary(sentences)
        word_set = list(desc_dict.token2id.keys())
        # word2id = {w:i+1 for i,w in enumerate(word_set)}
        model = gensim.models.word2vec.Word2Vec(
            sentences, vector_size=self.datafmt_cfg['word_emb_dim'],
        )
        wv_from_bin = model.wv

        word2id = copy.deepcopy(model.wv.key_to_index)
        word2id = {k: word2id[k] + 1 for k in word2id}
        self.word_emb_dict = {word2id[key]: wv_from_bin[key] for key in word2id}
        self.word_emb_dict[0] = np.zeros(shape=(self.datafmt_cfg['word_emb_dim'], ))

        # 将训练集、验证集、测试集中未出现在word_emb中的单词，全部替换成ID为0，并进行padding
        self.df_Q['content'] = self.df_Q['content'].apply(lambda x: [word2id.get(xx, 0) for xx in x])        
        pad_mat, _, _ = PadSeqUtil.pad_sequence(
            self.df_Q['content'].tolist(), maxlen=self.datafmt_cfg['max_sentence_len'], padding='post',
            is_truncate=True, truncating='post', value=0, 
        )
        self.df_Q['content'] = [pad_mat[i].tolist() for i in range(pad_mat.shape[0])]
        
        tmp_df_Q = self.df_Q.set_index("exer_id")
        self.content_mat = torch.from_numpy(np.vstack(
            [tmp_df_Q.loc[exer_id]['content'] for exer_id in range(self.datafmt_cfg['dt_info']['exer_count'])]
        ))

        self.datafmt_cfg['dt_info']['word_count'] = len(self.word_emb_dict)
        self.datafmt_cfg['dt_info']['word_emb_dim'] = self.datafmt_cfg['word_emb_dim']

    def get_extra_data(self):
        super_dic = super().get_extra_data()
        super_dic['w2v_word_emb'] = np.vstack(
            [self.word_emb_dict[k] for k in range(self.datafmt_cfg['dt_info']['word_count'])]
        )
        super_dic['exer_content'] = self.content_mat
        return super_dic
    
    # def __getitem__(self, index):
    #     dic = super().__getitem__(index)
    #     dic['exer_text_seq'] = torch.stack(
    #         [self.content_mat[exer_seq] for exer_seq in dic['exer_seq']], dim=0
    #     )
    #     return dic

    @classmethod
    def read_Q_matrix(cls, cfg):
        file_path = f'{cfg.frame_cfg.data_folder_path}/{cfg.dataset}-Q.csv'
        sep = cfg.datafmt_cfg['seperator']
        df_Q = pd.read_csv(file_path, sep=sep, encoding='utf-8', usecols=['exer_id:token', 'cpt_seq:token_seq', 'content:token_seq'])
        feat_name2type, df_Q = cls._convert_df_to_std_fmt(df_Q)
        df_Q['cpt_seq'] = df_Q['cpt_seq'].astype(str).apply(lambda x: [int(i) for i in x.split(',')])
        df_Q['content'] = df_Q['content'].astype(str).apply(lambda x: [str(i) for i in x.split(',')]) # 注意这里，根据逗号分隔的
        return feat_name2type, df_Q
