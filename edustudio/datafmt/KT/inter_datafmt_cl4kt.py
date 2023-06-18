from .kt_inter_datafmt import KTInterDataFmt
import pandas as pd
import os
import torch
from torch.utils.data import DataLoader, default_collate
from collections import defaultdict
import numpy as np
from itertools import chain
from ..utils import PadSeqUtil


class KTInterDataFmtCL4KT(KTInterDataFmt):
    default_cfg = {
        'sequence_option': "recent",
    }

    def __init__(self,
                 cfg,
                 train_dict,
                 val_dict,
                 test_dict,
                 feat2type,
                 **kwargs
                 ):
        self.df_Q = kwargs['df_Q']
        self.train_easier_cpts = kwargs['train_easier_cpts']
        self.train_harder_cpts = kwargs['train_harder_cpts']
        
        super().__init__(cfg, train_dict, val_dict, test_dict, feat2type)


    @classmethod
    def from_cfg(cls, cfg):
        feat2type_, df_Q = cls.read_Q_matrix(cfg)  # df_Q:[exer_cise, [concept, ...]]
        feat2type, train_dict, val_dict, test_dict = cls.read_data(cfg, df_Q)
        
        train_easier_cpts, train_harder_cpts = train_dict['easier_cpts'], train_dict['harder_cpts']
        del train_dict['easier_cpts'], train_dict['harder_cpts']
        del val_dict['easier_cpts'], val_dict['harder_cpts'] 
        del test_dict['easier_cpts'], test_dict['harder_cpts']
        
        feat2type.update(feat2type_)
        return cls(
            cfg=cfg,
            train_dict=cls.dict2tensor(train_dict),
            test_dict=cls.dict2tensor(test_dict),
            val_dict=cls.dict2tensor(val_dict) if val_dict is not None else None,
            feat2type=feat2type, df_Q=df_Q, 
            train_easier_cpts=train_easier_cpts, train_harder_cpts=train_harder_cpts,
        )

    def _stat_dataset_info(self):
        super()._stat_dataset_info()
        exer_count = max(self.datafmt_cfg['dt_info']['exer_count'], self.df_Q['exer_id'].max() + 1)
        self.datafmt_cfg['dt_info']['exer_count'] = exer_count

        cpt_count = len(set(list(chain(*self.df_Q['cpt_seq'].to_list()))))
        self.datafmt_cfg['dt_info']['cpt_count'] = cpt_count

        self.datafmt_cfg['dt_info'].update({
            'train_easier_cpts': self.train_easier_cpts,
        })
        self.datafmt_cfg['dt_info'].update({
            'train_harder_cpts': self.train_harder_cpts,
        })

    @classmethod
    def read_data(cls, cfg, df_Q):
        if not os.path.exists(f'{cfg.frame_cfg.data_folder_path}'):
            cls.download_dataset(cfg)
        # is_dataset_divided=true: not need divided data
        if cfg.datafmt_cfg['is_dataset_divided']:
            feat2type, train_df, val_df, test_df = cls._read_data_from_divided(cfg)
            train_df, val_df, test_df = cls._unfold_dataset(train_df, val_df, test_df, df_Q)

            cfg_window_size = cfg.datafmt_cfg['window_size']
            if cfg_window_size <= 0 or cfg_window_size is None:
                cfg_window_size = np.max([
                    train_df[['stu_id', 'exer_id', 'cpt_unfold_seq']].groupby('stu_id').agg('count')['cpt_unfold_seq'].max(),
                    val_df[['stu_id', 'exer_id', 'cpt_unfold_seq']].groupby('stu_id').agg('count')['cpt_unfold_seq'].max() if not val_df.empty else 0,
                    test_df[['stu_id', 'exer_id', 'cpt_unfold_seq']].groupby('stu_id').agg('count')['cpt_unfold_seq'].max()
                ])
                cfg.logger.info(f"actual window size: {cfg_window_size}")
            
            train_dict = cls.construct_df2dict(cfg, train_df, maxlen=cfg_window_size)  
            val_dict = cls.construct_df2dict(cfg, val_df, maxlen=cfg_window_size)
            test_dict = cls.construct_df2dict(cfg, test_df, maxlen=cfg_window_size)
        else:

            feat2type, data_df = cls._read_data_from_undivided(cfg)
            train_dict, val_dict, test_dict = cls._divide_data_df(cfg, data_df)

        return feat2type, train_dict, val_dict, test_dict

    @classmethod
    def _unfold_dataset(cls,train_df, val_df, test_df, df_Q):
        # add concept to exercise_id  |  unfold operate and reranking
        unique_cpt_seq = df_Q['cpt_seq'].explode().unique()
        cpt_map = dict(zip(unique_cpt_seq, range(len(unique_cpt_seq))))  # reranking 
        df_Q_unfold = pd.DataFrame({
            'exer_id': df_Q['exer_id'].repeat(df_Q['cpt_seq'].apply(len)),
            'cpt_unfold_seq': df_Q['cpt_seq'].explode().replace(cpt_map)
        })
        train_df_unfold = pd.merge(train_df, df_Q_unfold, on=['exer_id'], how='left').reset_index(drop=True)
        val_df_unfold = pd.merge(val_df, df_Q_unfold, on=['exer_id'], how='left').reset_index(drop=True)
        test_df_unfold = pd.merge(test_df, df_Q_unfold, on=['exer_id'], how='left').reset_index(drop=True)
        return train_df_unfold, val_df_unfold, test_df_unfold

    @classmethod
    def construct_df2dict(cls, cfg, df: pd.DataFrame, maxlen=None):
        if df is None: return None
     
        if cfg.datafmt_cfg['sequence_option'] == 'recent':
            tmp_df = df[['stu_id', 'exer_id', 'cpt_unfold_seq', 'label']].groupby('stu_id').agg(
                lambda x: list(x)[-maxlen:]
            ).reset_index()

            exer_seq, idx, mask_seq = PadSeqUtil.pad_sequence(  # mask_seq corresponding to attention_mask
            tmp_df['exer_id'].to_list(), return_idx=True, return_mask=True,
            maxlen=maxlen, padding='pre'
            )
            cpt_unfold_seq, _, _ = PadSeqUtil.pad_sequence(
                tmp_df['cpt_unfold_seq'].to_list(),
                maxlen=maxlen, padding='pre'
            )
            label_seq, _, _ = PadSeqUtil.pad_sequence(
                tmp_df['label'].to_list(), dtype=np.float32,
                maxlen=maxlen, padding='pre'
            )
        else:
            tmp_df = df[['stu_id', 'exer_id', 'cpt_unfold_seq', 'label']].groupby('stu_id').agg(
                lambda x: list(x)[:maxlen]
            ).reset_index()

            exer_seq, idx, mask_seq = PadSeqUtil.pad_sequence(
                tmp_df['exer_id'].to_list(), return_idx=True, return_mask=True,
                maxlen=maxlen
            )
            cpt_unfold_seq, _, _ = PadSeqUtil.pad_sequence(
                tmp_df['cpt_unfold_seq'].to_list(),
                maxlen=maxlen
            )
            label_seq, _, _ = PadSeqUtil.pad_sequence(
                tmp_df['label'].to_list(), dtype=np.float32,
                maxlen=maxlen
            )

        stu_id = tmp_df['stu_id'].to_numpy()[idx]

        cpt_correct = defaultdict(int)
        cpt_count = defaultdict(int)
        for i, (c_list, r_list) in enumerate(zip(cpt_unfold_seq, label_seq)):
            for c, r in zip(c_list[mask_seq[i] == 1], r_list[mask_seq[i] == 1]):
                cpt_correct[c] += r
                cpt_count[c] += 1
        cpt_diff = {c: cpt_correct[c] / float(cpt_count[c]) for c in cpt_correct}  # cpt difficult
        ordered_cpts = [item[0] for item in sorted(cpt_diff.items(), key=lambda x: x[1])]
        easier_cpts, harder_cpts = defaultdict(int), defaultdict(int)
        for index, cpt in enumerate(ordered_cpts):  
            if index == 0:
                easier_cpts[cpt] = ordered_cpts[index + 1]
                harder_cpts[cpt] = cpt
            elif index == len(ordered_cpts) - 1:
                easier_cpts[cpt] = cpt
                harder_cpts[cpt] = ordered_cpts[index - 1]
            else:
                easier_cpts[cpt] = ordered_cpts[index + 1]
                harder_cpts[cpt] = ordered_cpts[index - 1]

        return {
            'stu_id': stu_id,
            'exer_seq': exer_seq,
            'cpt_unfold_seq': cpt_unfold_seq,
            'label_seq': label_seq,
            'mask_seq': mask_seq,
            'easier_cpts': easier_cpts,
            'harder_cpts': harder_cpts
        }

    @classmethod
    def read_Q_matrix(cls, cfg):
        file_path = f'{cfg.frame_cfg.data_folder_path}/{cfg.dataset}-Q.csv'
        sep = cfg.datafmt_cfg['seperator']
        df_Q = pd.read_csv(file_path, sep=sep, encoding='utf-8', usecols=['exer_id:token', 'cpt_seq:token_seq'])
        feat_name2type, df_Q = cls._convert_df_to_std_fmt(df_Q)
        df_Q['cpt_seq'] = df_Q['cpt_seq'].astype(str).apply(lambda x: [int(i) for i in x.split(',')])
        return feat_name2type, df_Q
