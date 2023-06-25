import pandas as pd
import os
import time
from .raw2mid import BaseRaw2Mid

r"""
R2M_JunyiExerAsCpt
########################
"""


class R2M_JunyiExerAsCpt(BaseRaw2Mid):
    """R2M_JunyiExerAsCpt is a class used to handle the Junyi dataset, where we consider the exercise's name as the basis for constructing the cpt_seq (concept sequence)."""

    def process(self, **kwargs):
        super().process()
        # # 处理 Junyi Problem log dataset
        raw_data = pd.read_csv(f"{self.rawpath}/junyi_ProblemLog_original.csv", encoding='utf-8', low_memory=True)
        data = raw_data.copy()
        # 去除花费时间小于等于0s的或者超过半年的6*30*24*60*60s = 15552000s
        data.drop(data[(data.time_taken <= 0) | (data.time_taken >= 15552000)].index, inplace=True)
        data = data[['user_id', 'exercise', 'time_done', 'time_taken', 'correct']]
        # 确定start_timestamp:float, order_id:token, correct 和 time_taken (单位时间为ms)
        data['start_timestamp:float'] = data['time_done'] // 1000 - data['time_taken'] * 1000
        data = data.sort_values(by='start_timestamp:float', ascending=True)
        data['order_id:token'] = range(len(data))
        data['correct'] = data['correct'].astype(int)
        data['time_taken'] = data['time_taken'] * 1000
        data = data.rename(columns={
            'user_id': 'stu_id:token', 'exercise': 'exer_name', 'time_taken': 'cost_time:float',
            'correct': 'label:float',
        })

        # # 处理知识点之间的关系，parent关系

        exercise = pd.read_csv(f"{self.rawpath}/junyi_Exercise_table.csv", encoding='utf-8', low_memory=False)
        exercise.fillna('', inplace=True)

        # 定义一个name_to_id的函数，返回值为两个dict,一个是name2id，一个是id2name
        # 主要用于将唯一标识的字符串生成自增的id
        def name_to_id(l: list()):
            if l is None:
                return None, None
            l = list(l)
            name2id = dict(zip(l, range(len(l))))
            id2name = dict(zip(range(len(l)), l))
            return name2id, id2name

        # 处理exer_parent,知识点之间的有向图
        exer_parent = exercise[['name', 'prerequisites']].copy()

        def divideExercise(x):
            if type(x) == str:
                return x.split(",")
            else:
                return None

        exer_parent['prerequisites'] = exer_parent['prerequisites'].apply(lambda x: divideExercise(x))
        # 所有的kcs
        kcs = set(exer_parent['name'])
        exer_parent['prerequisites'].apply(lambda x: kcs.update(x))
        if 'nan' in kcs:
            kcs.remove('nan')
        if '' in kcs:
            kcs.remove('')
        kc2id, id2kc = name_to_id(list(kcs))
        exer_parent['exer_id'] = exer_parent['name'].map(kc2id)

        # 处理parent_name的列表到parent_id的列表
        def process_parent(x: list()):
            if 'nan' in x:
                return None
            if '' in x:
                return None
            cpt_parent_id = [kc2id[parent] for parent in x]
            cpt_parent_id.sort()
            return cpt_parent_id

        exer_parent['parent_id'] = exer_parent['prerequisites'].apply(lambda x: process_parent(x))
        # 最终的hier文件,表示整个cpt_seq的知识点有向图的关系
        # from:token  to:token
        # cpt_head:token cpt_tail:token
        hier = []
        for index, row in exer_parent.iterrows():
            cpt_tail = row['exer_id']
            cpt_head = row['parent_id']
            # parent_id -> exer_id
            if cpt_head is not None:
                for cpt in cpt_head:
                    hier.append([cpt, cpt_tail])
        df_hier = pd.DataFrame(columns=['cpt_head:token', 'cpt_tail:token'], data=hier)

        # 处理交互信息
        data['exer_id:token'] = data['exer_name'].map(kc2id)
        df_inter = data.reindex(
            columns=['stu_id:token', 'exer_id:token', 'label:float', 'start_timestamp:float', 'cost_time:float',
                     'order_id:token', ])
        df_inter

        # 处理用户信息
        df_stu = data['stu_id:token'].unique()
        # df_stu['class_id'] = None
        # df_stu['gender'] = None

        # ## 处理习题信息
        df_exer = data[['exer_id:token']]
        df_exer = df_exer.drop_duplicates(subset=['exer_id:token'], keep='first')
        df_exer['cpt_seq:token_seq'] = data['exer_id:token']

        # 此处将数据保存到`self.midpath`中
        df_inter.to_csv(f"{self.midpath}/{self.dt}.inter.csv", index=False, encoding='utf-8')
        # df_stu.to_csv(f"{self.midpath}/{self.dt}.stu.csv", index=False, encoding='utf-8')
        df_exer.to_csv(f"{self.midpath}/{self.dt}.exer.csv", index=False, encoding='utf-8')
        df_hier.to_csv(f"{self.midpath}/{self.dt}.cpt_relation.prerequisite.csv", index=False, encoding='utf-8')

        return
