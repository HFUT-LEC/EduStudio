from ..common import BaseMid2Cache
import pandas as pd
import numpy as np


class M2C_MGCD_OP(BaseMid2Cache):
    default_cfg = {
        "group_id_field": "class_id:token",
        "min_inter": 5,
    }

    def __init__(self, m2c_cfg, is_dataset_divided) -> None:
        super().__init__(m2c_cfg)
        self.is_dataset_divided = is_dataset_divided
        assert self.is_dataset_divided is False

    @classmethod
    def from_cfg(cls, cfg):
        m2c_cfg = cfg.datatpl_cfg.get(cls.__name__)
        is_dataset_divided = cfg.datatpl_cfg.is_dataset_divided
        return cls(m2c_cfg, is_dataset_divided)

    def process(self, **kwargs):
        df_stu = kwargs['df_stu']
        df_inter = kwargs['df']
        df_inter_group, df_inter_stu, kwargs['df_stu']= self.get_df_group_and_df_inter(
            df_inter=df_inter, df_stu=df_stu
        )
        kwargs['df'] = df_inter_group
        kwargs['df_inter_stu'] = df_inter_stu
        kwargs['df_stu'] = kwargs['df_stu'].rename(columns={self.m2c_cfg['group_id_field']: 'group_id:token'})
        self.group_count = df_inter_group['group_id:token'].nunique()
        return kwargs

    def get_df_group_and_df_inter(self, df_stu:pd.DataFrame, df_inter:pd.DataFrame):
        df_inter = df_inter.merge(df_stu[['stu_id:token', self.m2c_cfg['group_id_field']]], on=['stu_id:token'], how='left')  # 两个表合并，这样inter里也有class_id

        df_inter_group = pd.DataFrame()
        df_inter_stu = pd.DataFrame()

        for _, inter_g in df_inter.groupby(self.m2c_cfg['group_id_field']):
            exers_list = inter_g[['stu_id:token', 'exer_id:token']].groupby('stu_id:token').agg(set)['exer_id:token'].tolist()
            inter_exer_set = None # 选择所有学生都做的题目
            for exers in exers_list: inter_exer_set = exers if inter_exer_set is None else (inter_exer_set & exers)
            inter_exer_set = np.array(list(inter_exer_set))
            if inter_exer_set.shape[0] >= self.m2c_cfg['min_inter']:
                tmp_group_df = inter_g[inter_g['exer_id:token'].isin(inter_exer_set)]
                tmp_stu_df = inter_g[~inter_g['exer_id:token'].isin(inter_exer_set)]

                df_inter_group = pd.concat([df_inter_group, tmp_group_df], ignore_index=True, axis=0)
                df_inter_stu = pd.concat([df_inter_stu, tmp_stu_df], ignore_index=True, axis=0)
            else:
                df_stu = df_stu[df_stu['class_id:token'] != inter_g['class_id:token'].values[0]]

        df_inter_group = df_inter_group[['label:float', 'exer_id:token', self.m2c_cfg['group_id_field']]].groupby(
            [self.m2c_cfg['group_id_field'], 'exer_id:token']
        ).agg('mean').reset_index().rename(columns={self.m2c_cfg['group_id_field']: 'group_id:token'})

        return df_inter_group, df_inter_stu, df_stu[df_stu['class_id:token'].isin(df_inter_group['group_id:token'].unique())]
    
    def set_dt_info(self, dt_info, **kwargs):
        dt_info['group_count'] = self.group_count