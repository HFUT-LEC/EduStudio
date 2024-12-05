from .raw2mid import BaseRaw2Mid
import pandas as pd
r"""
R2M_ASSIST_2017
#####################################
ASSIST_2017 dataset preprocess
"""


class R2M_ASSIST_17(BaseRaw2Mid):
    """R2M_ASSIST_17 is to preprocess ASSISTment 2017 dataset"""
    def process(self):
        super().process()
        # pd.set_option("mode.chained_assignment", None)  # ingore warning


        # 读取数据 并显示
        df = pd.read_csv(f"{self.rawpath}/anonymized_full_release_competition_dataset.csv",encoding='utf-8',low_memory=False)

        # 进行映射
        df = df[['studentId','problemId','correct','startTime','timeTaken','MiddleSchoolId','InferredGender','skill','assignmentId']]

        # 处理skill，映射为skill_id
        knowledge_points = df['skill'].unique()
        knowledge_point_ids = {kp: idx for idx, kp in enumerate(knowledge_points, start=0)}
        df['skill_id'] = df['skill'].map(knowledge_point_ids)
        del df['skill']

        # 性别映射，空值保留
        gender_mapping = {'Male': 1, 'Female': 0}
        df['gender:float'] = df['InferredGender'].map(gender_mapping)
        del df['InferredGender']

        # 处理其他列
        def sort(data, column):
            '''将原始数据对指定列进行排序，并完成0-num-1映射'''
            if(column != 'startTime'):
                data .sort_values(column, inplace=True)
                value_mapping = {}
                new_value = 0
                for value in data[column].unique():
                    value_mapping[value] = new_value
                    new_value += 1
                new_column = f'new_{column}'
                data[new_column] = data[column].map(value_mapping)
                del data[column]
                
            # 单独处理 startTime，用字典映射
            else: 
                data = data.sort_values(by=['new_studentId', 'startTime'])

                user_mapping = {}
                user_count = {}

                def generate_mapping(row):
                    '''生成作答记录时间编号映射的函数'''
                    user_id = row['new_studentId']
                    timestamp = row['startTime']
                    if user_id not in user_mapping:
                        user_mapping[user_id] = {}
                        user_count[user_id] = 0
                    if timestamp not in user_mapping[user_id]:
                        user_mapping[user_id][timestamp] = user_count[user_id]
                        user_count[user_id] += 1
                    return user_mapping[user_id][timestamp]
                data['new_order_id'] = data.apply(generate_mapping, axis=1)
            return data
        df = sort(df,'studentId')
        df = sort(df,'assignmentId')
        df = sort(df,'problemId')
        df = sort(df,'MiddleSchoolId')
        df = sort(df,'startTime')
        
        # 修改列名及顺序
        df = df.rename(columns = {'new_studentId' : 'stu_id:token', 'new_assignmentId':'assignment_id:token_seq','new_problemId' : 'exer_id:token',
                                  'correct' : 'label:float','new_MiddleSchoolId':'school_id:token','new_order_id':'order_id:token',
                                  'startTime':'start_timestamp:float', 'timeTaken':'cost_time:float','skill_id':'cpt_seq:token_seq'})
        new_column_order = ['stu_id:token','exer_id:token','label:float','start_timestamp:float','cost_time:float','order_id:token',
                            'school_id:token', 'gender:float','cpt_seq:token_seq','assignment_id:token_seq']
        df = df.reindex(columns=new_column_order)

        # df_inter 的相关处理
        df_inter = df[['stu_id:token','exer_id:token','label:float','start_timestamp:float','cost_time:float','order_id:token']]
        df_inter.drop_duplicates(inplace=True)
        df_inter .sort_values('stu_id:token', inplace=True)
        
        # df_user 相关处理
        df_user = df[['stu_id:token','school_id:token','gender:float']]
        df_user.drop_duplicates(inplace=True)
        df_user .sort_values('stu_id:token', inplace=True)

        # df_exer 相关处理

        # 处理列名
        df_exer = df[['exer_id:token','cpt_seq:token_seq','assignment_id:token_seq']]
        df_exer.sort_values(by='exer_id:token', inplace=True)
        df_exer.drop_duplicates(inplace=True)

        # 合并 cpt_seq
        grouped_skills = df_exer[['exer_id:token','cpt_seq:token_seq']]
        grouped_skills.drop_duplicates(inplace=True)
        grouped_skills.sort_values(by='cpt_seq:token_seq', inplace=True)
        grouped_skills['exer_id:token'] = grouped_skills['exer_id:token'].astype(str)
        grouped_skills['cpt_seq:token_seq'] = grouped_skills['cpt_seq:token_seq'].astype(str)
        grouped_skills  = grouped_skills.groupby('exer_id:token')['cpt_seq:token_seq'].agg(','.join).reset_index()

        # 合并 assignment_id
        grouped_assignments = df_exer[['exer_id:token','assignment_id:token_seq']]
        grouped_assignments.drop_duplicates(inplace=True)
        grouped_assignments.sort_values(by='assignment_id:token_seq', inplace=True)
        grouped_assignments['exer_id:token'] = grouped_assignments['exer_id:token'].astype(str)
        grouped_assignments['assignment_id:token_seq'] = grouped_assignments['assignment_id:token_seq'].astype(str)
        grouped_assignments  = grouped_assignments.groupby('exer_id:token')['assignment_id:token_seq'].agg(','.join).reset_index()

        # 合并结果
        df_exer = pd.merge(grouped_skills, grouped_assignments, on='exer_id:token', how='left')
        df_exer['exer_id:token'] = df_exer['exer_id:token'].astype(int)
        df_exer.sort_values(by='exer_id:token', inplace=True)
        

        # # Save MidData
        # 
        # 此处将数据保存到`self.midpath`中

        df_inter.to_csv(f"{self.midpath}/{self.dt}.inter.csv", index=False, encoding='utf-8')
        df_user.to_csv(f"{self.midpath}/{self.dt}.user.csv", index=False, encoding='utf-8')
        df_exer.to_csv(f"{self.midpath}/{self.dt}.exer.csv", index=False, encoding='utf-8')
        # pd.set_option("mode.chained_assignment", "warn")  # igore warning