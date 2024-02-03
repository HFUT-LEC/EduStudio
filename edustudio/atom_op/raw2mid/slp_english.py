from edustudio.atom_op.raw2mid import BaseRaw2Mid
import pandas as pd
import numpy as np
import time

"""
    SLP Dataset: https://aic-fe.bnu.edu.cn/en/data/index.html
"""


class R2M_SLP_English(BaseRaw2Mid):
    """
        rawdata: https://aic-fe.bnu.edu.cn/en/data/index.html
    """
    def process(self):
        super().process()

        # for stu
        df_stu = pd.read_csv(f"{self.rawpath}/student.csv")
        df_stu.dropna(subset=['school_id'], inplace=True, how='any', axis=0)
        df_stu = df_stu[df_stu['school_id'] != 'n.a.']

        df_stu = df_stu.merge(
            pd.read_csv(f"{self.rawpath}/family.csv", index_col=False),
            on=['student_id'], how='inner'
        )

        df_stu = df_stu.merge(
                pd.read_csv(f"{self.rawpath}/school.csv"),
                on=['school_id'], how='inner'
        )

        df_stu.drop([
            'rate_of_higher_educated_teachers',
            "rate_of_teachers_with_master's_degree_and_above"
        ], inplace=True, axis=1)
        df_stu.rename(columns={
            'student_id': 'stu_id:token', 'gender': 'gender:token', 
            'school_id': 'sch_id:token', 'class_id': 'class_id:token', 
            'age_father': 'age_father:float', 'age_mother': 'age_mother:token',
            'edubg_father': 'edubg_father:token', 'edubg_mother':'edubg_mother:token', 
            'affiliation_father':'affiliation_father:token', 
            'affiliation_mother': 'affiliation_mother:token',
            'family_income': 'family_income:token', 'is_only_child':'is_only_child:token',
            'live_on_campus': 'live_on_campus:token', 
            'gathering_frequency_father':'gathering_frequency_father:token',
            'gathering_frequency_mother':'gathering_frequency_mother:token', 
            'family_traveling_times': "family_traveling_times:token", 
            'school_type': 'school_type:token',
            'dist_to_downtown': 'dist_to_downtown:float',
            #'rate_of_higher_educated_teachers': 'rate_of_higher_educated_teachers:float',
            #"rate_of_teachers_with_master's_degree_and_above": "rate_of_teachers_with_master's_degree_and_above:float",
        }, inplace=True)

        # for inter
        df_inter = pd.read_csv(f"{self.rawpath}/term-eng.csv", index_col=False, low_memory=False)
        df_inter = df_inter[(df_inter == 'n.a.').sum(axis=1) == 0].reset_index(drop=True)
        df_inter = df_inter[df_inter['concept'] != 'n.a.']
        df_inter['label'] = df_inter['score']/df_inter['full_score'].astype(float)
        
        df_exer = df_inter[['question_id', 'exam_id', 'subject_abbr', 'concept']]
        df_inter = df_inter[['student_id', 'question_id', 'score', 'full_score', 'time_access', 'label']]
        df_exer.drop_duplicates(subset=['question_id'], inplace=True)
        df_exer['concept'] = df_exer['concept'].apply(lambda x: x.split(";"))
        df_inter['time_access'] = df_inter['time_access'].apply(lambda x: self.convert2timestamp(x))

        df_inter.rename(columns={
            'student_id': 'stu_id:token', 'question_id': 'exer_id:token',
            'score': 'score:float', 'full_score':'full_score:float', 
            'time_access': 'start_timestamp:float', 'label':'label:float'
        }, inplace=True)

        df_exer.rename(columns={
             'question_id': 'exer_id:token',
             'exam_id': 'exam_id:token', 
             'subject_abbr': 'subject_abbr:token',
             'concept': 'cpt_seq:token_seq'
        }, inplace=True)

        df_inter['order_id:token'] = df_inter['start_timestamp:float'].astype(int)
        
        # save
        df_inter.to_csv(f"{self.midpath}/{self.dt}.inter.csv", index=False, encoding='utf-8')
        df_stu.to_csv(f"{self.midpath}/{self.dt}.stu.csv", index=False, encoding='utf-8')
        df_exer.to_csv(f"{self.midpath}/{self.dt}.exer.csv", index=False, encoding='utf-8')

    @staticmethod
    def convert2timestamp(dt):
        timeArray = time.strptime(dt, "%Y-%m-%d %H:%M:%S")
        timestamp = time.mktime(timeArray)
        return timestamp