from .raw2mid import BaseRaw2Mid
import pandas as pd
r"""
R2M_Eedi_20_T34
#####################################
NIPS34 dataset preprocess
"""


class R2M_Eedi_20_T34(BaseRaw2Mid):
    """R2M_Eedi_20_T34 is to preprocess NIPS 2020 challenge Task 3&4 dataset"""
    def process(self):
        super().process()
        # 读入数据 查看
        df = pd.read_csv(f"{self.rawpath}/train_data/train_task_3_4.csv", encoding='utf-8')
        df_exer = pd.read_csv(f"{self.rawpath}/metadata/answer_metadata_task_3_4.csv", encoding='utf-8')
        df_qus = pd.read_csv(f"{self.rawpath}/metadata/question_metadata_task_3_4.csv", encoding='utf-8')
        df_user = pd.read_csv(f"{self.rawpath}/metadata/student_metadata_task_3_4.csv", encoding='utf-8')
        df_rel = pd.read_csv(f"{self.rawpath}/metadata/subject_metadata.csv", encoding='utf-8')

        # 合并，获取 DateAnswered 等信息
        df = pd.merge(df, df_exer[['AnswerId', 'DateAnswered', 'GroupId', 'SchemeOfWorkId']], on='AnswerId', how='left')
        df = pd.merge(df, df_user[['UserId', 'Gender']], on='UserId', how='left')
        df = pd.merge(df, df_qus[['QuestionId', 'SubjectId']], on='QuestionId', how='left')
        df.drop(['AnswerId', 'CorrectAnswer', 'AnswerValue'], axis=1, inplace=True)

        # 处理学生多个班级的问题，保留做题数目最多的班级
        class_problem_counts = df.groupby(['GroupId', 'UserId'])['QuestionId'].nunique().reset_index()
        max_class_counts = class_problem_counts.groupby('UserId')['QuestionId'].idxmax()
        max_class_counts = class_problem_counts.loc[max_class_counts, ['UserId', 'GroupId']]
        df = df.merge(max_class_counts, on=['UserId', 'GroupId'])

        # 处理性别 不确定为空 男1 女0
        mapping = {0: None, 1: 0, 2: 1}
        df['Gender'] = df['Gender'].map(mapping)

        # 处理答题时间
        df['DateAnswered'] = pd.to_datetime(df['DateAnswered'])
        base_time = df['DateAnswered'].min()
        df['timestamp'] = (df['DateAnswered'] - base_time).dt.total_seconds() * 1000

        # 进行映射
        def sort(data, column):
            '''将原始数据对指定列进行排序，并完成0-num-1映射'''
            if (column != 'DateAnswered'):
                sorted_id = sorted(data[column].unique())
                num = len(sorted_id)
                mapping = {id: i for i, id in enumerate(sorted_id)}
                data[column] = data[column].replace('', '').map(mapping)

            # 单独处理 DateAnswered，用字典映射
            else:
                data = data.sort_values(by=['UserId', 'DateAnswered'])
                user_mapping = {}
                user_count = {}
                def generate_mapping(row):
                    user_id = row['UserId']
                    timestamp = row['DateAnswered']
                    if user_id not in user_mapping:
                        user_mapping[user_id] = {}
                        user_count[user_id] = 0
                    if timestamp not in user_mapping[user_id]:
                        user_mapping[user_id][timestamp] = user_count[user_id]
                        user_count[user_id] += 1
                    return user_mapping[user_id][timestamp]

                data['new_order_id'] = data.apply(generate_mapping, axis=1)
            return data

        df = sort(df, 'UserId')
        df = sort(df, 'QuestionId')
        df = sort(df, 'timestamp')
        df = sort(df, 'GroupId')
        df = sort(df, 'SchemeOfWorkId')
        df = sort(df, 'DateAnswered')

        # 映射cpt中的内容
        sorted_cpt_id = sorted(df_rel['SubjectId'].unique())
        num = len(sorted_cpt_id)
        mapping = {cpt_id: i for i, cpt_id in enumerate(sorted_cpt_id)}
        df_rel['SubjectId'] = df_rel['SubjectId'].map(mapping)
        df_rel['ParentId'] = df_rel['ParentId'].replace('', '').map(mapping)

        # 修改列名及顺序
        df = df.rename(columns={'UserId': 'stu_id:token', 'SchemeOfWorkId': 'assignment_id:token_seq',
                                'QuestionId': 'exer_id:token', 'IsCorrect': 'label:float', 'GroupId': 'class_id:token',
                                'timestamp': 'start_timestamp:float', 'Gender': 'gender:float',
                                'SubjectId': 'cpt_seq:token_seq',
                                'new_order_id': 'order_id:token'})
        new_column_order = ['stu_id:token', 'exer_id:token', 'label:float', 'start_timestamp:float', 'order_id:token',
                            'class_id:token', 'gender:float', 'cpt_seq:token_seq', 'assignment_id:token_seq']
        df = df.reindex(columns=new_column_order)

        # df_inter 的相关处理
        df_inter = df[['stu_id:token', 'exer_id:token', 'label:float', 'start_timestamp:float', 'order_id:token']]
        df_inter.drop_duplicates(inplace=True)
        df_inter.sort_values(by=['stu_id:token', 'order_id:token'], inplace=True)

        # df_user 相关处理
        df_user = df[['stu_id:token', 'class_id:token', 'gender:float']]
        df_user.drop_duplicates(inplace=True)
        df_user.sort_values('stu_id:token', inplace=True)

        # df_exer 相关处理
        df_exer = df[['exer_id:token', 'cpt_seq:token_seq', 'assignment_id:token_seq']]
        df_exer.drop_duplicates(inplace=True)
        # 拆分知识点列
        df_exer['cpt_seq:token_seq'] = df_exer['cpt_seq:token_seq'].str.strip('[]')
        df_exer['cpt_seq:token_seq'] = df_exer['cpt_seq:token_seq'].str.split(', ')
        df_exer = df_exer.explode('cpt_seq:token_seq')
        df_exer = sort(df_exer, 'cpt_seq:token_seq')
        # 合并 cpt_seq
        grouped_skills = df_exer[['exer_id:token', 'cpt_seq:token_seq']]
        grouped_skills.drop_duplicates(inplace=True)
        grouped_skills.sort_values(by='cpt_seq:token_seq', inplace=True)
        grouped_skills['exer_id:token'] = grouped_skills['exer_id:token'].astype(str)
        grouped_skills['cpt_seq:token_seq'] = grouped_skills['cpt_seq:token_seq'].astype(str)
        grouped_skills = grouped_skills.groupby('exer_id:token')['cpt_seq:token_seq'].agg(','.join).reset_index()
        # 合并 assignment_id
        grouped_assignments = df_exer[['exer_id:token', 'assignment_id:token_seq']]
        grouped_assignments.drop_duplicates(inplace=True)
        grouped_assignments.sort_values(by='assignment_id:token_seq', inplace=True)
        grouped_assignments['exer_id:token'] = grouped_assignments['exer_id:token'].astype(str)
        grouped_assignments['assignment_id:token_seq'] = grouped_assignments['assignment_id:token_seq'].astype(str)
        grouped_assignments = grouped_assignments.groupby('exer_id:token')['assignment_id:token_seq'].agg(
            ','.join).reset_index()
        # 结果合并
        df_exer = pd.merge(grouped_skills, grouped_assignments, on='exer_id:token', how='left')
        df_exer['exer_id:token'] = df_exer['exer_id:token'].astype(int)
        df_exer.sort_values(by='exer_id:token', inplace=True)

        # df_cpt 相关处理
        # 读取信息完成映射
        df_cpt = df_rel[['SubjectId', 'ParentId', 'Level']]
        df_cpt = df_cpt.rename(
            columns={'SubjectId': 'cpt_id:token', 'ParentId': 'parent_id:token', 'Level': 'level:float'})
        df_cpt['cpt_id:token'] = df_cpt['cpt_id:token'].astype(int)
        df_cpt.sort_values(by='cpt_id:token', inplace=True)

        # 保存数据
        df_inter.to_csv(f"{self.midpath}/{self.dt}.inter.csv", index=False, encoding='utf-8')
        df_user.to_csv(f"{self.midpath}/{self.dt}.user.csv", index=False, encoding='utf-8')
        df_exer.to_csv(f"{self.midpath}/{self.dt}.exer.csv", index=False, encoding='utf-8')
        df_cpt.to_csv(f"{self.midpath}/{self.dt}.cpt_tree.csv", index=False, encoding='utf-8')