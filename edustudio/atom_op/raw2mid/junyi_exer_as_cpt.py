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

    def process(self):
        super().process()
        pd.set_option("mode.chained_assignment", None)  # ignore warning
        # 读取数据集，并展示相关数据
        raw_data = pd.read_table(f"{self.rawpath}/junyi_ProblemLog_for_PSLC.txt", sep='\t', encoding='utf-8',
                                 low_memory=True)
        data = raw_data
        # 删除hint的交互记录
        data = data.drop(data[data.Outcome == 'HINT'].index)
        data = data[['Anon Student Id', 'Time', 'Level (Section)', 'Problem Name', 'Problem Start Time', 'Outcome',
                     'KC (Exercise)', 'KC (Topic)', 'KC (Area)']]
        # 确定order字段
        data = data.sort_values(by='Time', ascending=True)
        data['order:token'] = range(len(data))
        data['cost_time:float'] = data['Time'] - data['Problem Start Time']

        inter = data.rename(columns={'Anon Student Id': 'stu_id:token', 'Problem Name': 'exer_name:token'
            , 'Outcome': 'label:float', 'Problem Start Time': 'start_timestamp:float', }) \
            .reindex(
            columns=['stu_id:token', 'exer_name:token', 'label:float', 'start_timestamp:float', 'cost_time:float',
                     'order:token'])

        # 定义一个name_to_id的函数，返回值为两个dict,一个是name2id，一个是id2name
        # 主要用于将唯一标识的字符串生成自增的id
        def name_to_id(l: list()):
            if l is None:
                return None, None
            l = list(l)
            name2id = dict(zip(l, range(len(l))))
            id2name = dict(zip(range(len(l)), l))
            return name2id, id2name

        unique_exer = list(inter['exer_name:token'].unique())
        exer2id, id2exer = name_to_id(unique_exer)
        inter['exer_id:token'] = inter['exer_name:token'].map(exer2id)
        str2label = dict()
        str2label['CORRECT'] = 1
        str2label['INCORRECT'] = 0
        inter['label:float'] = inter['label:float'].map(str2label)
        df_inter = inter[
            ['stu_id:token', 'exer_id:token', 'label:float', 'start_timestamp:float', 'cost_time:float', 'order:token']]

        # ## 处理用户信息
        df_stu = data['Anon Student Id'].unique()
        # df_stu['class_id'] = None
        # df_stu['gender'] = None

        # 处理习题信息
        # 'Anon Student Id','Time','Level (Section)','Problem Name','Problem Start Time','Outcome','KC (Exercise)','KC (Topic)','KC (Area)'
        exer = data[['Problem Name', 'Level (Section)', 'KC (Exercise)', 'KC (Area)', 'KC (Topic)']].copy()
        exer = exer.drop_duplicates(subset=['Problem Name'], keep='first')
        exer['exer_id'] = exer['Problem Name'].map(exer2id)
        unique_assignment = list(exer['Level (Section)'].unique())
        assignment2id, id2assignment = name_to_id(unique_assignment)
        exer['assignment_id'] = exer['Level (Section)'].map(assignment2id)
        # 以exercise来处理cpt_seq字段
        unique_kc_exercise = exer['KC (Exercise)']
        kc_exercise2id, id2kc_exercise = name_to_id(unique_kc_exercise)
        exer['cpt_seq_exercise'] = exer['KC (Exercise)'].map(kc_exercise2id)
        df_exer = exer[['exer_id', 'assignment_id', 'cpt_seq_exercise']].rename(
            columns={'exer_id': 'exer_id:token', 'assignment_id': 'assignment_id:token_seq',
                     'cpt_seq_exercise': 'cpt_seq:token_seq'})

        # 此处将数据保存到`self.midpath`中
        df_inter.to_csv(f"{self.midpath}/{self.dt}.inter.csv", index=False, encoding='utf-8')
        # df_stu.to_csv(f"{self.midpath}/{self.dt}.stu.csv", index=False, encoding='utf-8')
        df_exer.to_csv(f"{self.midpath}/{self.dt}.exer.csv", index=False, encoding='utf-8')
        pd.set_option("mode.chained_assignment", "warn")  # ignore warning
        return
