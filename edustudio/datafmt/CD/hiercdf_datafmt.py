from .inter_datafmt_extends_q import CDInterDataFmtExtendsQ
import torch
from typing import Dict
import pandas as pd


class HierCDFDataFmt(CDInterDataFmtExtendsQ):
    def __init__(self, cfg, 
                 train_dict: Dict[str, torch.tensor], 
                 test_dict: Dict[str, torch.tensor], 
                 feat_name2type: Dict[str, str], 
                 df_Q, df_cpt_relation: torch.tensor,
                 val_dict: Dict[str, torch.tensor] = None
        ):
        self.df_cpt_relation = df_cpt_relation
        super().__init__(cfg, train_dict, test_dict, feat_name2type, df_Q, val_dict)
    
    @classmethod
    def from_cfg(cls, cfg):
        feat_name2type, train_df, val_df, test_df = cls.read_data(cfg)
        name2type, df_Q = cls.read_Q_matrix(cfg)
        feat_name2type.update(name2type)
        name2type, df_cpt_relation = cls.read_cpt_relation(cfg)
        feat_name2type.update(name2type)

        return cls(
            cfg=cfg,
            train_dict=cls.df2dict(train_df),
            test_dict=cls.df2dict(test_df),
            val_dict=cls.df2dict(val_df) if val_df is not None else None,
            df_Q=df_Q, df_cpt_relation=df_cpt_relation,
            feat_name2type=feat_name2type
        )

    def get_extra_data(self):
        extra_dict = super().get_extra_data()
        extra_dict['df_cpt_relation'] = self.df_cpt_relation
        return extra_dict
        
    @classmethod
    def read_cpt_relation(cls, cfg):
        file_path = f'{cfg.frame_cfg.data_folder_path}/{cfg.dataset}-cpt_relation.csv'
        sep = cfg.datafmt_cfg['seperator']
        df_cpt_relation = pd.read_csv(file_path, sep=sep, encoding='utf-8', usecols=['cpt_head:token', 'cpt_tail:token'])
        feat_name2type, df_Q = cls._convert_df_to_std_fmt(df_cpt_relation)
        return feat_name2type, df_Q
