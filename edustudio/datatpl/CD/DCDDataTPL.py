import os
from ..common.edu_datatpl import EduDataTPL
import json
from edustudio.datatpl.common.general_datatpl import DataTPLStatus
import torch


class DCDDataTPL(EduDataTPL):
    default_cfg = {
        'n_folds': 5,
        'mid2cache_op_seq': ['M2C_Label2Int', 'M2C_FilterRecords4CD', 'M2C_ReMapId', 'M2C_RandomDataSplit4CD', 'M2C_GenQMat', 'M2C_BuildMissingQ', 'M2C_FillMissingQ'],
        'cpt_relation_file_name': 'cpt_relation',
    }

    def __init__(self, cfg, df, df_train=None, df_valid=None, df_test=None, dict_cpt_relation=None, status=DataTPLStatus(), df_stu=None, df_exer=None):
        self.dict_cpt_relation = dict_cpt_relation
        super().__init__(cfg, df, df_train, df_valid, df_test, df_stu, df_exer, status)

    def _check_param(self):
        super()._check_params()
        assert 0 <= self.datatpl_cfg['Q_delete_ratio'] < 1

    @property
    def common_str2df(self):
        dic = super().common_str2df
        dic['dict_cpt_relation'] = self.dict_cpt_relation
        return dic
    

    def process_data(self):
        super().process_data()
        dt_info = self.final_kwargs['dt_info']
        user_count = dt_info['stu_count']
        item_count = dt_info['exer_count']
        self.interact_mat_list = []
        for interact_df in self.final_kwargs['df_train_folds']:
            interact_mat = torch.zeros((user_count, item_count), dtype=torch.int8)
            idx = interact_df[interact_df['label:float'] == 1][['stu_id:token','exer_id:token']].to_numpy()
            interact_mat[idx[:,0], idx[:,1]] = 1  
            idx = interact_df[interact_df['label:float'] != 1][['stu_id:token','exer_id:token']].to_numpy()
            interact_mat[idx[:,0], idx[:,1]] = -1
            self.interact_mat_list.append(interact_mat)

        self.final_kwargs['interact_mat_list'] = self.interact_mat_list

        if self.final_kwargs['dict_cpt_relation'] is None:
            self.final_kwargs['dict_cpt_relation'] = {i: [i] for i in range(self.final_kwargs['dt_info']['cpt_count'])}

    @classmethod
    def load_data(cls, cfg):
        kwargs = super().load_data(cfg)
        fph = f"{cfg.frame_cfg.data_folder_path}/middata/{cfg.datatpl_cfg['cpt_relation_file_name']}.json"
        if os.path.exists(fph):
            with open(fph, 'r', encoding='utf-8') as f:
                kwargs['dict_cpt_relation'] = json.load(f)
        else:
            cfg.logger.warning("without cpt_relation.json")
            kwargs['dict_cpt_relation'] = None
        return kwargs

    def get_extra_data(self):
        extra_dict = super().get_extra_data()
        extra_dict['filling_Q_mat'] = self.filling_Q_mat
        extra_dict['interact_mat'] = self.interact_mat
        return extra_dict

    def set_info_for_fold(self, fold_id):
        super().set_info_for_fold(fold_id)
        self.filling_Q_mat = self.final_kwargs['filling_Q_mat_list'][fold_id]
        self.interact_mat = self.final_kwargs['interact_mat_list'][fold_id]
