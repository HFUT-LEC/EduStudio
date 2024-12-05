from .raw2mid import BaseRaw2Mid
import pandas as pd


class R2M_ASSIST_1213(BaseRaw2Mid):
    def process(self, **kwargs):
        use_cols = ["user_id", "skill_id", "start_time", "problem_id", "correct", "ms_first_response", 'assignment_id',
                    'attempt_count', 'student_class_id', 'teacher_id']
        df = pd.read_csv(f"{self.rawpath}/2012-2013-data-with-predictions-4-final.csv", low_memory=False,
                         usecols=use_cols)
        df = df.dropna(subset=use_cols)
        df['correct'] = df['correct'].apply(int)
        df['skill_id'] = df['skill_id'].apply(int)
        df = df[df['correct'].isin([0, 1])]  # filter responses

        from datetime import datetime
        def change2timestamp(t, hasf=True):
            if hasf:
                timeStamp = datetime.strptime(t, "%Y-%m-%d %H:%M:%S.%f").timestamp() * 1000
            else:
                timeStamp = datetime.strptime(t, "%Y-%m-%d %H:%M:%S").timestamp() * 1000
            return int(timeStamp)

        df['start_timestamp'] = df['start_time'].apply(lambda x: change2timestamp(x, hasf='.' in x))
        # df['skill_id'] = df['skill_id'].apply(lambda x:[x])
        df.rename(columns={'problem_id': 'exer_id', 'correct': 'label', 'ms_first_response': 'cost_time',
                           'skill_id': 'cpt_seq', 'student_class_id': 'class_id'}, inplace=True)
        df.drop(columns=['start_time'], inplace=True)

        grouped = df.groupby('user_id', as_index=False)

        # 定义一个函数来修改每个group的DataFrame
        def modify_group(group):
            group.sort_values(['start_timestamp'], inplace=True)
            group['order'] = range(len(group))
            return group

        # 使用apply函数来应用修改函数到每个group的DataFrame
        df_modified = grouped.apply(modify_group)
        df = df_modified.reset_index(drop=True)
        df_inter = df[['exer_id', 'user_id', 'start_timestamp', 'order', 'cost_time', 'label']]
        df_inter = df_inter.rename(
            columns={'user_id': 'stu_id:token', 'exer_id': 'exer_id:token', 'label': 'label:float',
                     'start_timestamp': 'start_timestamp:float', 'cost_time': 'cost_time:float',
                     'order': 'order_id:token'})
        df_exer = df[['exer_id', 'cpt_seq', 'assignment_id']]
        df_exer = df_exer.drop_duplicates(subset=['exer_id'])
        df_exer = df_exer.rename(
            columns={'exer_id': 'exer_id:token', 'assignment_id': 'assignment_id:token',
                     'cpt_seq': 'cpt_seq:token_seq'})
        df_stu = df[['user_id', 'class_id', 'teacher_id']]
        df_stu = df_stu.drop_duplicates(subset=['user_id'])
        df_stu = df_stu.rename(
            columns={'user_id': 'stu_id:token', 'class_id': 'class_id:token',
                     'teacher_id': 'teacher_id:token'})
        cpt_ls = []
        df['cpt_seq'].astype(str).apply(lambda x: cpt_ls.extend(x.split(',')))
        # ser_cpt = df['cpt_seq'].astype(str).apply(lambda x: [int(i) for i in x.split(',')])
        # ser_cpt.to_list()
        # cpt_set = set(cpt_ls)
        # cfg.exer_count = len(df['exer_id'].unique())
        # cfg.stu_count = len(df['user_id'].unique())
        # cfg.cpt_count = len(cpt_set)
        # cfg.interaction_count = len(df_inter)
        df_inter.to_csv(f"{self.midpath}/{self.dt}.inter.csv", index=False, encoding='utf-8')
        df_stu.to_csv(f"{self.midpath}/{self.dt}.stu.csv", index=False, encoding='utf-8')
        df_exer.to_csv(f"{self.midpath}/{self.dt}.exer.csv", index=False, encoding='utf-8')

    