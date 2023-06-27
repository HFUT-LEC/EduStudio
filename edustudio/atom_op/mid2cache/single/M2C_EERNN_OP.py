from ..common import BaseMid2Cache
from itertools import chain
import torch
import pandas as pd
import numpy as np
from edustudio.datatpl.utils import PadSeqUtil
import copy


class M2C_EERNN_OP(BaseMid2Cache):
    default_cfg = {
        'word_emb_dim': 50,
        'max_sentence_len': 100,
    }

    def process(self, **kwargs):
        dt_info = kwargs['dt_info']
        train_dict_list = kwargs['df_train_folds']
        df_exer = kwargs['df_exer']
        assert len(train_dict_list) > 0
        import gensim

        word_emb_dict_list,content_mat_list = [], []
        for train_dict in train_dict_list:
            exers = np.unique(train_dict['exer_seq:token_seq'])
            sentences = df_exer[df_exer['exer_id:token'].isin(exers)]['content:token_seq'].tolist()
            # desc_dict = gensim.corpora.Dictionary(sentences)
            # word_set = list(desc_dict.token2id.keys())
            # word2id = {w:i+1 for i,w in enumerate(word_set)}
            model = gensim.models.word2vec.Word2Vec(
                sentences, vector_size=self.m2c_cfg['word_emb_dim'],
            )
            wv_from_bin = model.wv

            word2id = copy.deepcopy(model.wv.key_to_index)
            word2id = {k: word2id[k] + 1 for k in word2id}
            word_emb_dict = {word2id[key]: wv_from_bin[key] for key in word2id}
            word_emb_dict[0] = np.zeros(shape=(self.m2c_cfg['word_emb_dim'], ))

            # 将训练集、验证集、测试集中未出现在word_emb中的单词，全部替换成ID为0，并进行padding
            df_exer['content:token_seq'] = df_exer['content:token_seq'].apply(lambda x: [word2id.get(xx, 0) for xx in x])        
            pad_mat, _, _ = PadSeqUtil.pad_sequence(
                df_exer['content:token_seq'].tolist(), maxlen=self.m2c_cfg['max_sentence_len'], padding='post',
                is_truncate=True, truncating='post', value=0, 
            )
            df_exer['content:token_seq'] = [pad_mat[i].tolist() for i in range(pad_mat.shape[0])]
            
            tmp_df_Q = df_exer.set_index('exer_id:token')
            content_mat = torch.from_numpy(np.vstack(
                [tmp_df_Q.loc[exer_id]['content:token_seq'] for exer_id in range(dt_info['exer_count'])]
            ))

            word_emb_dict_list.append(word_emb_dict)
            content_mat_list.append(content_mat)
        kwargs['word_emb_dict_list'] = word_emb_dict_list
        kwargs['content_mat_list'] = content_mat_list
        return kwargs

    def set_dt_info(self, dt_info, **kwargs):
        dt_info['word_emb_dim'] = self.m2c_cfg['word_emb_dim']
