import os

import torch

from ..common import EduDataTPL
import pandas as pd


class ECDDataTPL(EduDataTPL):
    default_cfg = {
        'mid2cache_op_seq': ['M2C_Label2Int', 'M2C_FilterRecords4CD', 'M2C_ReMapId', 'M2C_RandomDataSplit4CD',
                             'M2C_GenQMat'],
    }

    def process_load_data_from_middata(self):
        super().process_load_data_from_middata()
        self.final_kwargs['qqq_group_list'] = self.read_QQQ_group(self.cfg)
        qqq_list = self.df_stu['qqq_seq:token_seq'].to_list()
        self.QQQ_list = torch.tensor(qqq_list)
        self.final_kwargs['qqq_list'] = self.QQQ_list
        qqq_count = torch.max(self.QQQ_list) + 1
        self.cfg['datatpl_cfg']['dt_info']['qqq_count'] = qqq_count

    def get_extra_data(self):
        return {
            'qqq_group_list': self.final_kwargs['qqq_group_list'],
            'Q_mat': self.final_kwargs['Q_mat'],
            'qqq_list': self.final_kwargs['qqq_list']
        }

    def read_QQQ_group(self, cfg):
        group_path = f'{cfg.frame_cfg.data_folder_path}/middata/{cfg.dataset}_QQQ-group.csv'
        assert os.path.exists(group_path)
        df_QQQ_group = pd.read_csv(group_path, encoding='utf-8', usecols=['qqq_id:token', 'group_id:token'])
        gp = df_QQQ_group.groupby("group_id:token")['qqq_id:token']
        gps = gp.groups
        gps_list = []
        for k, v in gps.items():
            gps_list.append(list(v))
        return gps_list
