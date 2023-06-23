import pandas as pd
import os
import time
from .raw2mid import BaseRaw2Mid

r"""
R2M_Junyi_Exercise_As_Cpt
########################
"""


class R2M_Junyi_Exercise_As_Cpt(BaseRaw2Mid):
    """R2M_Junyi_Exercise_As_Cpt is a class used to handle the Junyi dataset, where we consider the exercise's KC(Exercise) as the basis for constructing the cpt_seq (concept sequence)."""

    def process(self, **kwargs):
        super().process()
        # # 处理 Junyi Problem log dataset
        raw_data = pd.read_table(f"{self.rawpath}/junyi_ProblemLog_for_PSLC.txt", sep='\t', encoding='utf-8',
                                 low_memory=True)
        data = raw_data
        data = data.drop(data[data.Outcome == 'HINT'].index)
        data = data.sort_values(by='Time', ascending=True)
        data = data[['Anon Student Id', 'Time', 'Level (Section)', 'Problem Name', 'Problem Start Time', 'Outcome',
                     'KC (Exercise)', 'KC (Topic)', 'KC (Area)']]
        data['order:token'] = range(len(data))
        data['cost_time:float'] = data['Time'] - data['Problem Start Time']

        # # 处理知识点之间的关系，parent关系
        exercise = pd.read_csv(f"{self.rawpath}/junyi_Exercise_table.csv", encoding='utf-8', low_memory=False)
        exercise.fillna('', inplace=True)

        def name_to_id(l: list()):
            r"""
             定义一个name_to_id的函数，返回值为两个dict,一个是name2id，一个是id2name
             主要用于将唯一标识的字符串生成自增的id
            """
            if l is None:
                return None, None
            l = list(l)
            name2id = dict(zip(l, range(len(l))))
            id2name = dict(zip(range(len(l)), l))
            return name2id, id2name

        # In[164]:

        exer_parent = exercise[['name', 'prerequisites']].copy()

        def divideExercise(x):
            r"""
            把前驱知识点集，分割成一个list
            """
            if type(x) == str:
                return x.split(",")
            else:
                return None

        exer_parent['prerequisites'] = exer_parent['prerequisites'].apply(lambda x: divideExercise(x))
        # 所有的kcs
        kcs = set(exer_parent['name'])
        exer_parent['prerequisites'].apply(lambda x: kcs.update(x))
        # 去除空值
        if 'nan' in kcs:
            kcs.remove('nan')
        if '' in kcs:
            kcs.remove('')
        kc2id, id2kc = name_to_id(list(kcs))
        exer_parent['exer_id'] = exer_parent['name'].map(kc2id)

        def process_parent(x: list()):
            r"""
                处理parent_name的列表到parent_id的列表
            """
            if 'nan' in x:
                return None
            if '' in x:
                return None
            cpt_parent_id = [kc2id[parent] for parent in x]
            cpt_parent_id.sort()
            return cpt_parent_id
        # 把parent_name列表转化为parent_id列表
        exer_parent['parent_id'] = exer_parent['prerequisites'].apply(lambda x: process_parent(x))
        # 最终的hier文件,表示整个cpt_seq的知识点有向图的关系
        # from:token  to:token
        # cpt_head:token cpt_tail:token
        hier = []
        for index, row in exer_parent.iterrows():
            cpt_tail = row['exer_id']
            cpt_head = row['parent_id']
            # parent_id -> exer_id 也就是 cpt_head -> cpt_tail
            if cpt_head is not None:
                for cpt in cpt_head:
                    hier.append([cpt, cpt_tail])
        df_hier = pd.DataFrame(columns=['cpt_head:token', 'cpt_tail:token'], data=hier)

        # # 处理交互信息
        inter = data.rename(columns={'Anon Student Id': 'stu_id:token', 'Problem Name': 'exer_name:token', 'Outcome': 'label:float', 'Problem Start Time': 'start_timestamp:float', }) \
            .reindex(columns=['stu_id:token', 'exer_name:token', 'label:float', 'start_timestamp:float', 'cost_time:float','order:token'])
        unique_exer = list(inter['exer_name:token'].unique())
        exer2id, id2exer = name_to_id(unique_exer)
        inter['exer_id:token'] = inter['exer_name:token'].map(exer2id)
        str2label = {'CORRECT': 1, 'INCORRECT': 0}
        inter['label:float'] = inter['label:float'].map(str2label)
        df_inter = inter[
            ['stu_id:token', 'exer_id:token', 'label:float', 'start_timestamp:float', 'cost_time:float', 'order:token']]

        #  处理用户信息
        df_stu = data['Anon Student Id'].unique()
        # df_stu['class_id'] = None
        # df_stu['gender'] = None

        #  处理习题信息
        # 'Anon Student Id','Time','Level (Section)','Problem Name','Problem Start Time','Outcome','KC (Exercise)','KC (Topic)','KC (Area)'
        exer = data[['Problem Name', 'Level (Section)', 'KC (Exercise)', 'KC (Area)', 'KC (Topic)']].copy()
        exer = exer.drop_duplicates(subset=['Problem Name'], keep='first')
        exer['cpt_seq_exercise_name'] = exer['Problem Name'].apply(lambda x: x.split("--")[0])
        exer['exer_id'] = exer['Problem Name'].map(exer2id)
        unique_assignment = list(exer['Level (Section)'].unique())
        assignment2id, id2assignment = name_to_id(unique_assignment)
        exer['assignment_id'] = exer['Level (Section)'].map(assignment2id)
        exer['cpt_seq_exercise_id'] = exer['cpt_seq_exercise_name'].map(kc2id)

        df_exer = exer[['exer_id', 'assignment_id', 'cpt_seq_exercise_id']].rename(
            columns={'exer_id': 'exer_id:token', 'assignment_id': 'assignment_id:token_seq',
                     'cpt_seq_exercise_id': 'cpt_seq:token_seq'})

        # 此处将数据保存到`self.midpath`中

        df_inter.to_csv(f"{self.midpath}/{self.dt}.inter.csv", index=False, encoding='utf-8')
        # df_stu.to_csv(f"{self.midpath}/{self.dt}.stu.csv", index=False, encoding='utf-8')
        df_exer.to_csv(f"{self.midpath}/{self.dt}.exer.csv", index=False, encoding='utf-8')
        df_hier.to_csv(f"{self.midpath}/{self.dt}.hier.csv", index=False, encoding='utf-8')

        return
