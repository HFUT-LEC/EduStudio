from ..common.base_mid2cache import BaseMid2Cache
import pandas as pd


class M2C_FilterRecords4CD(BaseMid2Cache):
    default_cfg = {
        "stu_least_records": 10,
        "exer_least_records": 0,
    }

    def process(self, **kwargs):
        df: pd.DataFrame = kwargs['df']
        assert df is not None

        # 去重，保留第一个记录
        df.drop_duplicates(
            subset=['stu_id:token', 'exer_id:token', "label:float"], 
            keep='first', inplace=True, ignore_index=True
        )
        
        stu_least_records = self.m2c_cfg['stu_least_records']
        exer_least_records = self.m2c_cfg['exer_least_records']

        # 循环删除user和item
        last_count = 0
        while last_count != df.__len__():
            last_count = df.__len__()
            gp_by_uid = df[['stu_id:token','exer_id:token']].groupby('stu_id:token').agg('count').reset_index()
            selected_users = gp_by_uid[gp_by_uid['exer_id:token'] >= stu_least_records].reset_index()['stu_id:token'].to_numpy()

            gp_by_iid = df[['stu_id:token','exer_id:token']].groupby('exer_id:token').agg('count').reset_index()
            selected_items = gp_by_iid[gp_by_iid['stu_id:token'] >= exer_least_records].reset_index()['exer_id:token'].to_numpy()

            df = df[df['stu_id:token'].isin(selected_users) & df['exer_id:token'].isin(selected_items)]


        df = df.reset_index(drop=True)
        selected_users = df['stu_id:token'].unique()
        selected_items = df['exer_id:token'].unique()

        if kwargs.get('df_exer', None) is not None:
            kwargs['df_exer'] = kwargs['df_exer'][kwargs['df_exer']['exer_id:token'].isin(selected_items)]
            kwargs['df_exer'].reset_index(drop=True, inplace=True)

        if kwargs.get('df_stu', None) is not None:
            kwargs['df_stu'] = kwargs['df_stu'][kwargs['df_stu']['stu_id:token'].isin(selected_users)]
            kwargs['df_stu'].reset_index(drop=True, inplace=True)

        kwargs['df'] = df
        return kwargs
