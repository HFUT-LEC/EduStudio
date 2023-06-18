from .kt_inter_datafmt import KTInterDataFmt
from itertools import chain
import torch
import pandas as pd
from ..utils.pad_seq_util import PadSeqUtil


class KTInterDataFmtExtendsQ(KTInterDataFmt):
    default_cfg = {
        'cpt_seq_window_size': -1,
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
        super().__init__(cfg, train_dict, val_dict, test_dict, feat2type)
        self.construct_fields()

    def construct_fields(self):
        self.Q_mat = self._get_Q_mat_from_df_arr(
            self.df_Q, 
            self.datafmt_cfg['dt_info']['exer_count'], 
            self.datafmt_cfg['dt_info']['cpt_count']
        )
        tmp_df_Q = self.df_Q.set_index('exer_id')
        exer_count = self.datafmt_cfg['dt_info']['exer_count']
        cpt_seq_unpadding = [
            (tmp_df_Q.loc[exer_id].tolist()[0] if exer_id in tmp_df_Q.index else []) for exer_id in range(exer_count)
        ]
        cpt_seq_padding, _, cpt_seq_mask = PadSeqUtil.pad_sequence(
            cpt_seq_unpadding, maxlen=self.datafmt_cfg['cpt_seq_window_size'], return_mask=True
        )
        self.cpt_seq_padding = torch.from_numpy(cpt_seq_padding)
        self.cpt_seq_mask = torch.from_numpy(cpt_seq_mask)

    def __getitem__(self, index):
        dic = super().__getitem__(index)
        dic['cpt_seq'] = torch.stack(
            [self.cpt_seq_padding[exer_seq] for exer_seq in dic['exer_seq']], dim=0
        )
        dic['cpt_seq_mask'] = torch.stack(
            [self.cpt_seq_mask[exer_seq] for exer_seq in dic['exer_seq']], dim=0
        )
        dic['Q_mat'] = self.Q_mat[dic['exer_seq']]
        return dic
    
    @classmethod
    def from_cfg(cls, cfg):
        feat2type, train_dict, val_dict, test_dict = cls.read_data(cfg)
        feat2type_, df_Q = cls.read_Q_matrix(cfg)
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
        exer_count = max(self.datafmt_cfg['dt_info']['exer_count'], self.df_Q['exer_id'].max()+1)
        self.datafmt_cfg['dt_info']['exer_count'] = exer_count
        
        cpt_count = len(set(list(chain(*self.df_Q['cpt_seq'].to_list()))))
        self.datafmt_cfg['dt_info']['cpt_count'] = cpt_count

    def _get_Q_mat_from_df_arr(self, df_Q_arr, exer_count, cpt_count):
        Q_mat = torch.zeros((exer_count, cpt_count), dtype=torch.int64)
        for _, item in df_Q_arr.iterrows():
            for cpt_id in item['cpt_seq']: Q_mat[item['exer_id'], cpt_id] = 1
        return Q_mat

    @classmethod
    def read_Q_matrix(cls, cfg):
        file_path = f'{cfg.frame_cfg.data_folder_path}/{cfg.dataset}-Q.csv'
        sep = cfg.datafmt_cfg['seperator']
        df_Q = pd.read_csv(file_path, sep=sep, encoding='utf-8', usecols=['exer_id:token', 'cpt_seq:token_seq'])
        feat_name2type, df_Q = cls._convert_df_to_std_fmt(df_Q)
        df_Q['cpt_seq'] = df_Q['cpt_seq'].astype(str).apply(lambda x: [int(i) for i in x.split(',')])
        return feat_name2type, df_Q
