from .kt_inter_datafmt import KTInterDataFmt
import pandas as pd
import os
import numpy as np
from itertools import chain
from ..utils import PadSeqUtil


class KTInterDataFmtCptUnfold(KTInterDataFmt):

    def __init__(self,
                 cfg,
                 train_dict,
                 val_dict,
                 test_dict,
                 feat2type,
                 **kwargs
                 ):
        self.df_Q = kwargs['df_Q']
        super().__init__(cfg, train_dict, val_dict, test_dict, feat2type)


    @classmethod
    def from_cfg(cls, cfg):
        if not os.path.exists(f'{cfg.frame_cfg.data_folder_path}'):
            cls.download_dataset(cfg)
        feat2type_, df_Q = cls.read_Q_matrix(cfg)
        feat2type, train_dict, val_dict, test_dict = cls.read_data(cfg, df_Q)

        feat2type.update(feat2type_)
        return cls(
            cfg=cfg,
            train_dict=cls.dict2tensor(train_dict),
            test_dict=cls.dict2tensor(test_dict),
            val_dict=cls.dict2tensor(val_dict) if val_dict is not None else None,
            feat2type=feat2type, df_Q=df_Q
        )

    def _stat_dataset_info(self):
        super()._stat_dataset_info()
        exer_count = max(self.datafmt_cfg['dt_info']['exer_count'], self.df_Q['exer_id'].max() + 1)
        self.datafmt_cfg['dt_info']['exer_count'] = exer_count

        cpt_count = len(set(list(chain(*self.df_Q['cpt_seq'].to_list()))))
        self.datafmt_cfg['dt_info']['cpt_count'] = cpt_count

    @classmethod
    def read_data(cls, cfg, df_Q):
        if cfg.datafmt_cfg['is_dataset_divided']:
            feat2type, train_df, val_df, test_df = cls._read_data_from_divided(cfg)
            train_df, val_df, test_df = cls._unfold_dataset(train_df, val_df, test_df, df_Q)

            cfg_window_size = cfg.datafmt_cfg['window_size']
            if cfg_window_size <= 0 or cfg_window_size is None:
                cfg_window_size = np.max([
                    train_df[['stu_id', 'exer_id', 'cpt_unfold_seq']].groupby('stu_id').agg('count')['cpt_unfold_seq'].max(),
                    val_df[['stu_id', 'exer_id', 'cpt_unfold_seq']].groupby('stu_id').agg('count')['cpt_unfold_seq'].max() if val_df else 0,
                    test_df[['stu_id', 'exer_id', 'cpt_unfold_seq']].groupby('stu_id').agg('count')['cpt_unfold_seq'].max()
                ])
                cfg.logger.info(f"actual window size: {cfg_window_size}")
            train_dict = cls.construct_df2dict(cfg, train_df, maxlen=cfg_window_size)
            val_dict = cls.construct_df2dict(cfg, val_df, maxlen=cfg_window_size)
            test_dict = cls.construct_df2dict(cfg, test_df, maxlen=cfg_window_size)
        else:

            feat2type, data_df = cls._read_data_from_undivided(cfg)
            unique_cpt_seq = df_Q['cpt_seq'].explode().unique()
            cpt_map = dict(zip(unique_cpt_seq, range(len(unique_cpt_seq))))
            df_Q_unfold = pd.DataFrame({
                'exer_id': df_Q['exer_id'].repeat(df_Q['cpt_seq'].apply(len)),
                'cpt_unfold_seq': df_Q['cpt_seq'].explode().replace(cpt_map)
            })
            df_unfold = pd.merge(data_df, df_Q_unfold, on=['exer_id'], how='left').reset_index(drop=True)
            train_dict, val_dict, test_dict = cls._divide_data_df(cfg, df_unfold)

        return feat2type, train_dict, val_dict, test_dict

    @classmethod
    def _unfold_dataset(cls,train_df, val_df, test_df, df_Q):
        unique_cpt_seq = df_Q['cpt_seq'].explode().unique()
        cpt_map = dict(zip(unique_cpt_seq, range(len(unique_cpt_seq))))
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

        tmp_df = df[['stu_id', 'exer_id', 'cpt_unfold_seq', 'label']].groupby('stu_id').agg(
            lambda x: list(x)
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

        return {
            'stu_id': stu_id,
            'exer_seq': exer_seq,
            'cpt_unfold_seq': cpt_unfold_seq,
            'label_seq': label_seq,
            'mask_seq': mask_seq
        }

    @classmethod
    def read_Q_matrix(cls, cfg):
        file_path = f'{cfg.frame_cfg.data_folder_path}/{cfg.dataset}-Q.csv'
        sep = cfg.datafmt_cfg['seperator']
        df_Q = pd.read_csv(file_path, sep=sep, encoding='utf-8', usecols=['exer_id:token', 'cpt_seq:token_seq'])
        feat_name2type, df_Q = cls._convert_df_to_std_fmt(df_Q)
        df_Q['cpt_seq'] = df_Q['cpt_seq'].astype(str).apply(lambda x: [int(i) for i in x.split(',')])
        return feat_name2type, df_Q