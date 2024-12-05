import pandas as pd
import os
import time
from .raw2mid import BaseRaw2Mid
from sklearn.preprocessing import LabelEncoder

r"""
R2M_JunyiExerAsCpt
########################
"""


class R2M_JunyiExerAsCpt(BaseRaw2Mid):
    """R2M_JunyiExerAsCpt is a class used to handle the Junyi dataset, where we consider the exercise's name as the basis for constructing the cpt_seq (concept sequence)."""

    def process(self, **kwargs):
        super().process()
        # # 处理 Junyi Problem log dataset
        data = pd.read_csv(f"{self.rawpath}/junyi_ProblemLog_original.csv", encoding='utf-8', low_memory=True)
        # # 去除花费时间小于等于0s的或者超过半年的6*30*24*60*60s = 15552000s
        # data.drop(data[(data.time_taken <= 0) | (data.time_taken >= 15552000)].index, inplace=True)
        data = data[['user_id', 'exercise', 'time_done', 'time_taken', 'correct']]
        # 确定start_timestamp:float, order_id:token, correct 和 time_taken (单位时间为ms)
        data['start_timestamp:float'] = data['time_done'] // 1000 - data['time_taken'] * 1000
        data = data.sort_values(by='start_timestamp:float', ascending=True)
        data['order_id:token'] = range(len(data))
        data['correct'] = data['correct'].astype(int)
        data['time_taken'] = data['time_taken'] * 1000
        df_inter = data.rename(columns={
            'user_id': 'stu_id:token', 'exercise': 'exer_id:token', 'time_taken': 'cost_time:float',
            'correct': 'label:float', 'time_done': 'time_done:float'
        })

        # read exer info
        raw_df_exer = pd.read_csv(
            f"{self.rawpath}/junyi_Exercise_table.csv", encoding='utf-8', low_memory=False,
            usecols=['name', 'prerequisites']
        )

        df_relation = raw_df_exer.copy()
        df_relation.dropna(inplace=True)
        df_relation.rename(columns={"name": "cpt_tail:token", 'prerequisites': "cpt_head:token"}, inplace=True)
        df_relation['cpt_head:token'] = df_relation['cpt_head:token'].apply(lambda x: x.split(","))
        df_relation = df_relation[['cpt_head:token', 'cpt_tail:token']].explode('cpt_head:token')

        df_exer = raw_df_exer.rename(columns={'name': 'exer_id:token', }) # 'topic': 'topic:token', 'area': 'area:token'

        exer_lbe = LabelEncoder()
        exer_lbe.fit(df_exer['exer_id:token'].tolist() + df_inter['exer_id:token'].tolist())
        df_inter['exer_id:token'] = exer_lbe.transform(df_inter['exer_id:token'])
        df_exer['exer_id:token'] = exer_lbe.transform(df_exer['exer_id:token'])
        df_relation["cpt_tail:token"] = exer_lbe.transform(df_relation["cpt_tail:token"])
        df_relation["cpt_head:token"] = exer_lbe.transform(df_relation["cpt_head:token"])

        df_exer.drop(columns=['prerequisites'], inplace=True)
        df_exer['cpt_seq:token_seq'] = df_exer['exer_id:token'].astype(str)
        df_exer.drop_duplicates('exer_id:token', inplace=True)
        
        # 此处将数据保存到`self.midpath`中
        df_inter.to_csv(f"{self.midpath}/{self.dt}.inter.csv", index=False, encoding='utf-8')
        # df_stu.to_csv(f"{self.midpath}/{self.dt}.stu.csv", index=False, encoding='utf-8')
        df_exer.to_csv(f"{self.midpath}/{self.dt}.exer.csv", index=False, encoding='utf-8')
        df_relation.to_csv(f"{self.midpath}/{self.dt}.cpt_relation.prerequisite.csv", index=False, encoding='utf-8')
